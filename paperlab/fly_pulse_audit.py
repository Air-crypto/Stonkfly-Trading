"""Audit and visualize the registered factorial neural replay without model compute."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path

from .fly_market_pulse import ARMS


def audit(study, reference_text):
    protocol=study['protocol'];reports=study['reports']
    if hashlib.sha256(reference_text.encode()).hexdigest()!=protocol['reference_report_sha256']:
        raise ValueError('Reference hash mismatch')
    reference=json.loads(reference_text)
    if protocol['arms']!=ARMS or set(reports)!=set(ARMS):raise ValueError('Missing factorial controls')
    rows=reference['phase_diagnostics'][protocol['pool']]
    expected=rows['online_original']['test']['decisions'][:3]
    for name,arm in ARMS.items():
        r=reports[name];events=r['events']
        if {k:r['config'][k] for k in arm}!=arm or r['native_build']!=reference['native_build'] or len(events)!=3:
            raise ValueError('Changed configuration, native build, or horizon')
        if len({e['memory']['sha256'] for e in events})!=1 and not arm['learning']:
            raise ValueError('Frozen weights changed')
        for e,ref in zip(events,expected):
            stimulus=ref['neural']['stimulus'] if arm['pulses']=='recorded' else 'none'
            if e['input_sha256']!=ref['neural']['input_sha256'] or e['market_decision_ts']!=ref['decision_ts'] or e['stimulus']!=stimulus:
                raise ValueError('Changed image, timestamp, or pulse schedule')
            if not arm['learning'] and (e['diagnostics']['plasticity_enabled'] or e['diagnostics']['weight_delta_l2']):
                raise ValueError('Frozen arm reported an update')
        if name in ('pristine_frozen_none','trained_online_recorded'):
            ref_name='pristine_frozen' if name=='pristine_frozen_none' else 'online_original'
            if [e['spike_sha256'] for e in events]!=[d['neural']['spike_sha256'] for d in rows[ref_name]['test']['decisions'][:3]]:
                raise ValueError('Reference spike control differs')
    comparisons=[]
    for i,(left,a) in enumerate(ARMS.items()):
        for right,b in list(ARMS.items())[i+1:]:
            changed=[k for k in a if a[k]!=b[k]]
            if len(changed)!=1:continue
            x,y=reports[left],reports[right]
            if changed[0]!='memory' and x['initial_memory_sha256']!=y['initial_memory_sha256']:
                raise ValueError('Single-factor comparison changed initial memory')
            comparisons.append({'factor':changed[0],'first':left,'second':right,
                'different_actions':sum(u['side']!=v['side'] for u,v in zip(x['events'],y['events'])),
                'different_spike_counts':sum(u['spike_sha256']!=v['spike_sha256'] for u,v in zip(x['events'],y['events']))})
    return {'comparisons':comparisons,'verification':{'all_eight_arms':True,'reference_controls_reproduced':True,
            'same_images_and_native_build':True,'frozen_reported_weights_stable':True},
            'interpretation':'Post hoc neural intervention on three already examined market images. No new trading return or policy-selection result.'}


def figure(study):
    rows=study['reports'];out=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="690" viewBox="0 0 1100 690">',
        '<rect width="1100" height="690" rx="14" fill="#101a2b"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#e8edf6}.small{fill:#b2bfd1;font-size:13px}</style>']
    def text(x,y,value,size=16,css=''):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" class="{css}">{escape(str(value))}</text>')
    text(28,40,'What produced the extra BUY?',26)
    text(28,67,'Same recorded market images. Independent controls for starting memory, updates, and stimulation.',13,'small')
    columns=[(28,'Starting memory'),(200,'Plasticity'),(350,'Pulses'),(490,'10:05'),(620,'10:10'),(750,'10:15'),(880,'Final gate'),(990,'Final Δ L2')]
    for x,label in columns:text(x,110,label,14)
    for i,(name,arm) in enumerate(ARMS.items()):
        y=153+i*55;events=rows[name]['events']
        out.append(f'<path d="M28,{y+18}H1070" stroke="#35445c"/>')
        values=[arm['memory'].title(),'Online' if arm['learning'] else 'Frozen',arm['pulses'].title(),
                *[e['side'] for e in events],str(events[-1]['gate_spikes']),f"{events[-1]['diagnostics']['weight_delta_l2']:.4f}"]
        for (x,_),value in zip(columns,values):text(x,y,value,16)
    text(28,622,'Both original reference controls reproduced. All frozen arms preserved synaptic memory during execution.',13,'small')
    text(28,648,'Gate = DNpe017 spikes in the final 500 ms observation. Δ L2 = final-observation weight update.',13,'small')
    text(28,671,'Mechanism diagnostic on previously examined inputs; no fills or returns were recomputed.',13,'small')
    out.append('</svg>');return '\n'.join(out)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--figure',type=Path)
    a=p.parse_args();study=json.loads(a.study.read_text());result=audit(study,a.reference.read_text())
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    if a.figure:a.figure.write_text(figure(study))
    print(json.dumps(result['comparisons'],indent=2))


if __name__=='__main__':main()
