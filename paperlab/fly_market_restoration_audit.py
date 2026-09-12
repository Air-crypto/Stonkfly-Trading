"""Audit and plot a completed market-memory restoration diagnostic offline."""
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

from .fly_market_restoration import ARMS, matches, validate

LABELS={'pristine_frozen':'Pristine / frozen','trained_frozen':'Trained / frozen',
        'restore_10704':'Restore inputs to 10704','restore_11402':'Restore inputs to 11402','restore_both':'Restore both MBON11 cells'}


def audit(study,reference_text,envelope):
    protocol,reference,plan=validate({'protocol':study['protocol'],'reference_json':reference_text,'market_plan':envelope})
    reports=study['reports'];rows=reference['phase_diagnostics'][protocol['pool']]
    if set(reports)!=set(ARMS):raise ValueError('Missing restoration controls')
    if len(study['training_events'])!=3 or any(not matches(e,d['neural']) for e,d in zip(study['training_events'],rows['online_original']['training']['decisions'])):
        raise ValueError('Training reference differs')
    comparisons={};control=reports['pristine_frozen']['events']
    for name,arm in ARMS.items():
        r=reports[name];events=r['events'];restoration=r['restoration']
        if r['native_build']!=reference['native_build'] or r['plan_sha256']!=envelope['sha256'] or r['reference_report_sha256']!=protocol['reference_report_sha256']:
            raise ValueError('Restoration provenance differs')
        if r['config']!={**arm,'preset':'recorded_market','news':'none','eta':.001,'learning':False,'pulses':'none','view':'original'}:raise ValueError('Restoration configuration differs')
        if r['initial_memory_sha256']!=r['final_memory_sha256']:raise ValueError('Frozen replay changed memory')
        if r['training_memory_sha256']!=rows['online_original']['training']['final_memory_sha256']:raise ValueError('Restoration used different trained memory')
        edges=restoration['edge_ids']
        if restoration['post_ids']!=arm['restore_post_ids'] or restoration['edge_count']!=len(edges) or len(set(edges))!=len(edges):raise ValueError('Restoration edge metadata differs')
        if bool(edges)!=bool(arm['restore_post_ids']):raise ValueError('Restoration targets and edges disagree')
        if len(events)!=3:raise ValueError('Incomplete restoration replay')
        for e,d in zip(events,rows['trained_frozen']['test']['decisions'][:3]):
            if e['market_decision_ts']!=d['decision_ts'] or e['input_sha256']!=d['neural']['input_sha256'] or e['stimulus']!='none':raise ValueError('Restoration inputs or stimulation differ')
            if e['diagnostics']['plasticity_enabled'] or e['diagnostics']['weight_delta_l2']!=0:raise ValueError('Frozen replay updated weights')
        if name in ('pristine_frozen','trained_frozen'):
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
    return {'comparisons':comparisons,'joint_readout_contrast_hz':interaction,
            'contrast_definition':'trained minus restore_10704 minus restore_11402 plus restore_both, separately for each observation',
            'both_restoration_matches_pristine_memory':reports['restore_both']['initial_memory_sha256']==reports['pristine_frozen']['initial_memory_sha256'],
            'verification':{'reference_controls_reproduced':True,'frozen_reported_memory_preserved':True,'restoration_masks_disjoint_and_complete':True},
            'interpretation':'Post hoc mechanism study on previously examined inputs. Restoring a neural output does not establish a trading improvement.'}


def figure(study):
    rows=study['reports'];svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="540" viewBox="0 0 1120 540">',
        '<rect width="1120" height="540" rx="14" fill="#101a2b"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#e8edf6}.small{font-size:13px;fill:#b2bfd1}</style>']
    def text(x,y,value,size=16,css=''):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" class="{css}">{escape(str(value))}</text>')
    text(28,40,'Which retained inputs changed the directional readout?',25)
    text(28,67,'Same three historical market images; frozen inference, no reinforcement, and no recomputed fills.',13,'small')
    columns=[(28,'Starting memory / restoration'),(330,'Restored edges')]+[(510+i*180,datetime.fromtimestamp(e['market_decision_ts'],timezone.utc).strftime('%H:%M UTC')) for i,e in enumerate(rows['pristine_frozen']['events'])]
    for x,label in columns:text(x,108,label,14)
    for i,name in enumerate(ARMS):
        r=rows[name];y=155+i*58;svg.append(f'<path d="M28,{y+24}H1090" stroke="#35445c"/>')
        text(28,y,LABELS[name],15);text(330,y,f"{r['restoration']['edge_count']:,}")
        for j,e in enumerate(r['events']):
            text(510+j*180,y,e['side']);text(510+j*180,y+18,f"R−L {e['difference_hz']:g} Hz · gate {e['gate_spikes']}",12,'small')
    text(28,475,'Restoration replaces selected incoming weights and efficacy memory with pristine values before replay.',13,'small')
    text(28,499,'All graph connections remain. Matching controls support mechanism diagnosis, not fresh profitability.',13,'small')
    svg.append('</svg>');return '\n'.join(svg)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('study','reference','plan','out'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--figure',type=Path);a=p.parse_args();study=json.loads(a.study.read_text())
    result=audit(study,a.reference.read_text(),json.loads(a.plan.read_text()))
    result['study_sha256']=hashlib.sha256(a.study.read_bytes()).hexdigest()
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    if a.figure:a.figure.write_text(figure(study))
    print(json.dumps(result['comparisons'],indent=2))


if __name__=='__main__':main()
