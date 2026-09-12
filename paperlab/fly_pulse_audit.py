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



def gate_timing(study, recordings, expected_arms=ARMS):
    """Locate decoder gate spikes in the published 10 ms recordings.

    These are bin boundaries, not exact spike timestamps or a causal path.
    Match each compact recording to its original study report before use.
    """
    result={}
    if set(recordings)!=set(expected_arms):raise ValueError('Missing timing controls')
    for name,view in recordings.items():
        if view['report']!=study['reports'][name]:
            raise ValueError('Timing recording differs from study report')
        frames=view['frames'];events=view['report']['events']
        if len(frames)!=len(events):raise ValueError('Missing timing observations')
        ids=[str(n['id']) for n in view['nodes']]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate displayed neuron')
        observations=[]
        for frame,event in zip(frames,events):
            if frame['event']!=event:raise ValueError('Timing event differs')
            times=frame['times_ms'];counts=frame['counts']
            if len(times)!=50 or times[-1]!=event['brain_ms'] or any(b-a!=10 for a,b in zip(times,times[1:])):
                raise ValueError('Expected 50 consecutive 10 ms bins')
            if len(counts)!=len(times) or any(len(row)!=len(ids) for row in counts):
                raise ValueError('Timing count dimensions differ')
            gates={}
            for neuron in event['cell_ids']['gate']:
                if neuron not in ids:raise ValueError('Decoder gate absent from displayed subset')
                column=ids.index(neuron);bins=[]
                for i,(end,row) in enumerate(zip(times,counts)):
                    count=row[column]
                    if not isinstance(count,int) or count<0:raise ValueError('Invalid spike count')
                    if count:bins.append({'bin':i,'start_ms':end-10,'end_ms':end,'spikes':count})
                gates[neuron]=bins
            total=sum(b['spikes'] for bins in gates.values() for b in bins)
            if total!=event['gate_spikes']:raise ValueError('Binned gate count differs from decoder total')
            observations.append({'market_decision_ts':event['market_decision_ts'],
                'side':event['side'],'gate_spikes':total,'neurons':gates})
        result[name]=observations
    return {'resolution_ms':10,'arms':result,
            'interpretation':'Recorded gate-spike bins, not exact spike times or evidence of which upstream connection caused a spike. No trading returns recomputed.'}


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
    p.add_argument('--recordings',type=Path,help='Directory containing pulse01-{arm}/view.json')
    p.add_argument('--timing-out',type=Path,help='Optional gate spike-bin report; requires --recordings')
    a=p.parse_args()
    if bool(a.recordings)!=bool(a.timing_out):p.error('--recordings and --timing-out must be used together')
    study=json.loads(a.study.read_text());result=audit(study,a.reference.read_text())
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    if a.figure:a.figure.write_text(figure(study))
    if a.recordings:
        files={name:(a.recordings/f'pulse01-{name}'/'view.json').read_bytes() for name in ARMS}
        timing=gate_timing(study,{name:json.loads(raw) for name,raw in files.items()})
        timing['view_sha256']={name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
        a.timing_out.write_text(json.dumps(timing,indent=2)+'\n')
    print(json.dumps(result['comparisons'],indent=2))


if __name__=='__main__':main()
