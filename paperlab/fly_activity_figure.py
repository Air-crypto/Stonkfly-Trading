"""Render audited activity-reset outputs without resimulating the model."""
import argparse
from datetime import datetime,timezone
from html import escape
import json
from pathlib import Path

from .fly_market_activity import catalog

LABELS={'carry':'Carry activity','full':'Full dynamics reset','visual_filters':'Visual filters reset','adaptation':'Intrinsic adaptation reset','voltage':'Voltage reset','conductance':'Synaptic current reset','voltage_conductance':'Voltage + current reset','gate_voltage_conductance':'Gate voltage + current reset'}


def render(study,audit):
    arms=catalog(study['protocol']);height=1030+(len(arms)-8)*98
    if study['protocol']!=audit['protocol'] or set(study['reports'])!=set(arms) or set(audit['arms'])!=set(arms):raise ValueError('Figure inputs do not describe the same audited study')
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}"><rect width="1120" height="{height}" rx="14" fill="#0e1727"/><style>text{{font-family:Arial,sans-serif;fill:#e8edf6}}.small{{fill:#b2bfd1}}</style>']
    def text(x,y,value,size=14,cls=''):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" class="{cls}">{escape(str(value))}</text>')
    text(28,40,'What carries the gate dropout between images?' if study['protocol']['schema']==1 else 'Activity resets on the same market inputs',25)
    text(28,67,'Same three market images · frozen synaptic memory · full retained graph · unchanged decoder',14,'small')
    text(28,91,'Intervene before images 2 and 3. All first-image outputs must match their carried-state control.',13,'small')
    first=study['reports']['pristine_carry']['events']
    for j,e in enumerate(first):text(322+260*j,125,f"Image {j+1} · {datetime.fromtimestamp(e['market_decision_ts'],timezone.utc):%H:%M} UTC",15)
    for i,(name,arm) in enumerate(arms.items()):
        s=audit['arms'][name];reported=study['summary'][name]
        if any(s[k]!=reported[k] for k in ('actions','difference_hz','gate_spikes')):raise ValueError('Figure summary differs from audited events')
        y=148+i*98;text(28,y+30,arm['memory'].capitalize()+' memory',16);text(28,y+53,LABELS[arm['state_reset']],14,'small')
        control=audit['arms'][arm['memory']+'_carry']
        for j in range(3):
            changed=any(s[k][j]!=control[k][j] for k in ('actions','difference_hz','gate_spikes'))
            x=306+260*j;svg.append(f'<rect x="{x}" y="{y}" width="242" height="80" rx="7" fill="#172237" stroke="{"#c4a1ff" if changed else "#35445c"}"/>')
            text(x+16,y+27,s['actions'][j],20);text(x+16,y+52,f"Gate: {s['gate_spikes'][j]} spikes",14)
            text(x+16,y+70,f"Right − left: {s['difference_hz'][j]:g} Hz",13,'small')
    text(28,height-69,'Purple border: a displayed output differs from its carried-state control. All eight conditions shown.' if study['protocol']['schema']==1 else f'Purple border: a displayed output differs from its carried-state control. All {len(arms)} conditions shown.',13,'small')
    text(28,height-45,'Actual boundary arrays verify the reset scope and preserved memory. No fills, P&L or promotion computed.',13,'small')
    text(28,height-21,'A full reset changes multiple dynamic fields and neural time; a recovered gate alone does not validate a trading fix.',13,'small')
    return '\n'.join(svg+['</svg>'])+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('study','audit','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(render(json.loads(a.study.read_text()),json.loads(a.audit.read_text())))


if __name__=='__main__':main()
