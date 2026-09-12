"""Render phase-separated market equity from the published decision ledger."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

PHASES = ("training", "development", "test")
LABELS = {"trained_frozen":"Trained / frozen inference", "pristine_frozen":"Original / frozen", "online_original":"Original / learning",
          "fixed_returns_frozen":"Fixed returns / frozen", "fixed_returns_online":"Fixed returns / learning",
          "reinforcement_gated":"Reinforcement gated"}


def phase_series(report, arm, phase):
    pools=list(report["phase_diagnostics"].values())
    traces=[p[arm][phase]["decisions"] for p in pools]
    times=[r["decision_ts"] for r in traces[0]]
    if any([r["decision_ts"] for r in rows]!=times for rows in traces):
        raise ValueError("Cannot aggregate unaligned decision ledgers")
    idle=report["initial_capital"]-len(pools)*report["costs"]["capital"]
    equity=[idle+sum(rows[i]["equity"] for rows in traces) for i in range(len(times))]
    if abs(equity[-1]-report["total_equity"][arm][phase])>1e-7:
        raise ValueError("Decision ledger does not reconcile to final equity")
    unavailable=[any(not rows[i]["available"] for rows in traces) for i in range(len(times))]
    return times,equity,unavailable


def render(report, title="Sealed market replay"):
    arms=list(report["total_equity"])
    series={(arm,phase):phase_series(report,arm,phase) for arm in arms for phase in PHASES}
    values=[v for _,equity,_ in series.values() for v in equity]
    low=min(min(values),1000);high=max(max(values),1000);span=max(high-low,1)
    low-=span*.12;high+=span*.12
    width=1120;height=170+len(arms)*155+100
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" rx="14" fill="#0e1728"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#e8edf6}.small{font-size:12px;fill:#aebcd0}</style>']
    def text(x,y,value,size=15,css=""):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" class="{css}">{escape(str(value))}</text>')
    text(28,40,title,25)
    text(28,65,'Each phase starts with $1,000. Learned state comes from training; development is not carried into test.',13,'small')
    text(28,87,'All panels share one dollar scale. Red points include an unavailable pool quote; cash may still be fully priced.',13,'small')
    for j,phase in enumerate(PHASES):
        times=series[arms[0],phase][0]
        stamp=lambda t:datetime.fromtimestamp(t,timezone.utc).strftime('%H:%M')
        text(260+j*280,123,f'{phase.title()}  {stamp(times[0])}–{stamp(times[-1])} UTC',15)
    for i,arm in enumerate(arms):
        top=145+i*155;bottom=top+86
        text(28,top+30,LABELS.get(arm,arm),15)
        text(28,top+52,f'${low:.2f}–${high:.2f}',12,'small')
        for j,phase in enumerate(PHASES):
            ts,equity,unavailable=series[arm,phase];left=260+j*280;right=left+238
            X=lambda k:left+k*(right-left)/(len(ts)-1)
            Y=lambda v:bottom-(v-low)/(high-low)*(bottom-top)
            svg.append(f'<rect x="{left}" y="{top}" width="238" height="86" fill="#172237" stroke="#35445c"/>')
            svg.append(f'<path d="M{left},{Y(1000)}H{right}" stroke="#75859c" stroke-dasharray="4 4"/>')
            points=' '.join(f'{X(k)},{Y(v)}' for k,v in enumerate(equity))
            svg.append(f'<polyline points="{points}" fill="none" stroke="#67e8cf" stroke-width="2"/>')
            for k,v in enumerate(equity):
                color='#ff8e8e' if unavailable[k] else '#67e8cf'
                svg.append(f'<circle cx="{X(k)}" cy="{Y(v)}" r="3.5" fill="{color}"/>')
            totals=[p[arm][phase] for p in report['phase_diagnostics'].values()]
            fills=sum(p['fills'] for p in totals);fees=sum(p['fees'] for p in totals)
            decisions=[d for p in totals for d in p['decisions'] if not d['terminal']]
            observed=sum(d['neural'] is not None for d in decisions)
            text(left,bottom+22,f'${equity[-1]:.2f} · {fills} fills · ${fees:.3f} fees',13)
            text(left,bottom+40,f'{observed}/{len(decisions)} decisions observed',12,'small')
    y=170+len(arms)*155
    selected=report['selection']['selected']
    text(28,y,'Development selection: '+(LABELS.get(selected,selected) if selected else 'No arm passed the cash and frozen-baseline gate.'),15)
    text(28,y+24,'Simulated DEX fees/slippage included; hosting excluded. Unavailable inventory uses stress valuation. News disabled.',13,'small')
    text(28,y+45,'Short retrospective experiment; this is not a monthly return estimate or evidence of executable live profit.',13,'small')
    svg.append('</svg>')
    return '\n'.join(svg)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--title',default='Sealed market replay')
    a=p.parse_args();a.out.write_text(render(json.loads(a.report.read_text()),a.title))


if __name__=='__main__':main()
