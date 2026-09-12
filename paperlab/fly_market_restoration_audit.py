"""Audit and plot a completed market-memory restoration diagnostic offline."""
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np

from .fly_market_restoration import ARMS, MBON07, matches, validate
from .fly_market_memory_audit import read_memory
from .fly_market_study import memory_signature

LABELS={'pristine_frozen':'Pristine / frozen','trained_frozen':'Trained / frozen',
        'restore_10704':'Restore inputs to 10704','restore_11402':'Restore inputs to 11402','restore_both':'Restore both MBON11 cells','restore_MBON07':'Restore MBON07',
        'restore_MBON07_10704':'Restore MBON07 + 10704','restore_MBON07_11402':'Restore MBON07 + 11402','restore_all':'Restore every plastic input'}


def audit(study,reference_text,envelope,artifacts=None):
    protocol,reference,plan=validate({'protocol':study['protocol'],'reference_json':reference_text,'market_plan':envelope})
    reports=study['reports'];rows=reference['phase_diagnostics'][protocol['pool']];arms=protocol['arms'];source=protocol['training_source']
    if set(reports)!=set(arms):raise ValueError('Missing restoration controls')
    if len(study['training_events'])!=3 or any(not matches(e,d['neural']) for e,d in zip(study['training_events'],rows[source]['training']['decisions'])):
        raise ValueError('Training reference differs')
    comparisons={};control=reports['pristine_frozen']['events']
    for name,arm in arms.items():
        r=reports[name];events=r['events'];restoration=r['restoration']
        if r['native_build']!=reference['native_build'] or r['plan_sha256']!=envelope['sha256'] or r['reference_report_sha256']!=protocol['reference_report_sha256']:
            raise ValueError('Restoration provenance differs')
        if r['config']!={**arm,'preset':'recorded_market','news':'none','eta':.001,'learning':False,'pulses':'none','view':'original'}:raise ValueError('Restoration configuration differs')
        if r['initial_memory_sha256']!=r['final_memory_sha256']:raise ValueError('Frozen replay changed memory')
        if r['training_memory_sha256']!=rows[source]['training']['final_memory_sha256']:raise ValueError('Restoration used different trained memory')
        edges=restoration['edge_ids']
        if restoration['post_ids']!=arm['restore_post_ids'] or restoration['edge_count']!=len(edges) or len(set(edges))!=len(edges):raise ValueError('Restoration edge metadata differs')
        if bool(edges)!=bool(arm['restore_post_ids']):raise ValueError('Restoration targets and edges disagree')
        if len(events)!=3:raise ValueError('Incomplete restoration replay')
        for e,d in zip(events,rows['trained_frozen']['test']['decisions'][:3]):
            if e['market_decision_ts']!=d['decision_ts'] or e['input_sha256']!=d['neural']['input_sha256'] or e['stimulus']!='none':raise ValueError('Restoration inputs or stimulation differ')
            if e['diagnostics']['plasticity_enabled'] or e['diagnostics']['weight_delta_l2']!=0:raise ValueError('Frozen replay updated weights')
        if name in (ARMS if protocol['schema']==2 else ('pristine_frozen','trained_frozen')):
            expected=rows[name]['test']['decisions'][:3]
            if any(not matches(e,d['neural']) for e,d in zip(events,expected)):raise ValueError('Original control did not reproduce')
            if r['initial_memory_sha256']!=rows[name]['test']['initial_memory_sha256']:raise ValueError('Original control memory differs')
        comparisons[name]={'actions':[e['side'] for e in events],'difference_hz':[e['difference_hz'] for e in events],
                           'gate_spikes':[e['gate_spikes'] for e in events],
                           'matching_pristine_spike_counts':sum(e['spike_sha256']==c['spike_sha256'] for e,c in zip(events,control)),
                           'matching_pristine_actions':sum(e['side']==c['side'] for e,c in zip(events,control)),
                           'restored_edges':len(edges)}
    left=set(reports['restore_10704']['restoration']['edge_ids']);right=set(reports['restore_11402']['restoration']['edge_ids'])
    if left&right or left|right!=set(reports['restore_both']['restoration']['edge_ids']):raise ValueError('Individual target restorations overlap or fail to cover combined restoration')
    interaction=[reports['trained_frozen']['events'][i]['difference_hz']-reports['restore_10704']['events'][i]['difference_hz']-reports['restore_11402']['events'][i]['difference_hz']+reports['restore_both']['events'][i]['difference_hz'] for i in range(3)]
    result={'comparisons':comparisons,'joint_readout_contrast_hz':interaction,
            'contrast_definition':'trained minus restore_10704 minus restore_11402 plus restore_both, separately for each observation',
            'both_restoration_matches_pristine_memory':reports['restore_both']['initial_memory_sha256']==reports['pristine_frozen']['initial_memory_sha256'],
            'verification':{'reference_controls_reproduced':True,'frozen_reported_memory_preserved':True,'restoration_masks_disjoint_and_complete':True},
            'interpretation':'Post hoc mechanism study on previously examined inputs. Restoring a neural output does not establish a trading improvement.'}
    if protocol['schema']==2:
        if artifacts is None:raise ValueError('Factorial restoration requires actual checkpoint auditing')
        root=Path(artifacts);files=['plastic-map.npz','pristine-memory.npz','trained-memory.npz']
        with np.load(root/'plastic-map.npz',allow_pickle=False) as saved:
            if set(saved.files)!={'post_ids','edge_ids'}:raise ValueError('Unexpected plastic mapping')
            posts=saved['post_ids'];edges=saved['edge_ids']
        if posts.ndim!=1 or edges.shape!=posts.shape or posts.dtype.kind not in 'iu' or edges.dtype.kind not in 'iu' or len(set(edges))!=len(edges):raise ValueError('Invalid plastic mapping')
        if any(r['graph']['plastic_edges']!=len(edges) for r in reports.values()):raise ValueError('Mapping omits declared plastic edges')
        if set(posts.astype(str))!=set(MBON07+['10704','11402']):raise ValueError('Factorial groups do not cover every plastic input')
        pristine=read_memory(root/'pristine-memory.npz');trained=read_memory(root/'trained-memory.npz')
        if memory_signature(pristine)!=rows['pristine_frozen']['test']['initial_memory_sha256'] or memory_signature(trained)!=rows[source]['training']['final_memory_sha256']:
            raise ValueError('Saved reference memory differs')
        for name,arm in arms.items():
            mask=np.isin(posts.astype(str),arm['restore_post_ids']);state=trained if arm['memory']=='trained' else pristine
            if any(state[k].shape!=posts.shape or pristine[k].shape!=posts.shape for k in state):raise ValueError('Memory and mapping dimensions differ')
            expected={k:np.where(mask,pristine[k],state[k]) for k in state}
            file=name+'/initial-memory.npz';files.append(file);actual=read_memory(root/file)
            if any(not np.array_equal(actual[k],expected[k]) for k in expected) or memory_signature(actual)!=reports[name]['initial_memory_sha256']:
                raise ValueError('Restoration changed undeclared memory or failed to reset selected inputs')
            if reports[name]['restoration']['edge_ids']!=[str(e) for e in edges[mask]]:raise ValueError('Restoration mask differs from saved mapping')
        if reports['restore_all']['initial_memory_sha256']!=reports['pristine_frozen']['initial_memory_sha256'] or comparisons['restore_all']['matching_pristine_spike_counts']!=3:
            raise ValueError('Restoring every input did not reproduce pristine memory and activity')
        contexts={'keep_MBON07_keep_10704':('trained_frozen','restore_11402'),
                  'keep_MBON07_restore_10704':('restore_10704','restore_both'),
                  'restore_MBON07_keep_10704':('restore_MBON07','restore_MBON07_11402'),
                  'restore_MBON07_restore_10704':('restore_MBON07_10704','restore_all')}
        effects={name:[comparisons[after]['difference_hz'][i]-comparisons[before]['difference_hz'][i] for i in range(3)] for name,(before,after) in contexts.items()}
        result['restoring_11402_effect_hz']=effects
        result['effect_definition']='After restoring 11402 minus before, holding the other two restoration choices fixed; one contrast per observation. Deterministic replay contrasts, not population effect estimates.'
        result['checkpoint_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
        result['verification'].update(all_memory_restorations_audited=True,all_five_reference_controls_reproduced=True,full_restoration_reproduces_pristine=True)
    return result


def figure(study):
    rows=study['reports'];height=250+58*len(rows);svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}">',
        f'<rect width="1120" height="{height}" rx="14" fill="#101a2b"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#e8edf6}.small{font-size:13px;fill:#b2bfd1}</style>']
    def text(x,y,value,size=16,css=''):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" class="{css}">{escape(str(value))}</text>')
    text(28,40,'Which retained inputs changed the directional readout?',25)
    text(28,67,'Same three historical market images; frozen inference, no reinforcement, and no recomputed fills.',13,'small')
    columns=[(28,'Starting memory / restoration'),(330,'Restored edges')]+[(510+i*180,datetime.fromtimestamp(e['market_decision_ts'],timezone.utc).strftime('%H:%M UTC')) for i,e in enumerate(rows['pristine_frozen']['events'])]
    for x,label in columns:text(x,108,label,14)
    for i,name in enumerate(study['protocol']['arms']):
        r=rows[name];y=155+i*58;svg.append(f'<path d="M28,{y+24}H1090" stroke="#35445c"/>')
        text(28,y,LABELS[name],15);text(330,y,f"{r['restoration']['edge_count']:,}")
        for j,e in enumerate(r['events']):
            text(510+j*180,y,e['side']);text(510+j*180,y+18,f"R−L {e['difference_hz']:g} Hz · gate {e['gate_spikes']}",12,'small')
    text(28,height-65,'Restoration replaces selected incoming weights and efficacy memory with pristine values before replay.',13,'small')
    text(28,height-41,'All graph connections remain. Matching controls support mechanism diagnosis, not fresh profitability.',13,'small')
    svg.append('</svg>');return '\n'.join(svg)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('study','reference','plan','out'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--artifacts',type=Path);p.add_argument('--figure',type=Path)
    p.add_argument('--recordings',type=Path,help='Directory containing each arm/view.json')
    p.add_argument('--timing-out',type=Path)
    a=p.parse_args()
    if bool(a.recordings)!=bool(a.timing_out):p.error('--recordings and --timing-out must be used together')
    study=json.loads(a.study.read_text())
    result=audit(study,a.reference.read_text(),json.loads(a.plan.read_text()),a.artifacts)
    result['study_sha256']=hashlib.sha256(a.study.read_bytes()).hexdigest()
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    if a.figure:a.figure.write_text(figure(study))
    if a.recordings:
        from .fly_pulse_audit import gate_timing
        files={name:(a.recordings/name/'view.json').read_bytes() for name in study['protocol']['arms']}
        timing=gate_timing(study,{name:json.loads(raw) for name,raw in files.items()},study['protocol']['arms'])
        timing['view_sha256']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
        a.timing_out.write_text(json.dumps(timing,indent=2)+'\n')
    print(json.dumps(result['comparisons'],indent=2))


if __name__=='__main__':main()
