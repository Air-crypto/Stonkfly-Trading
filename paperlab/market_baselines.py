"""Descriptive cash and exposure references on sealed paper-study prices.

These are not registered candidate arms and never alter study selection.
No neural model, external order, training or cloud submission is used.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from .core import Broker, Costs, Tick, atomic_json, digest
from .fly_market_study import quote_at

POLICIES={
    'cash':'Keep all capital as cash; no orders.',
    'one_entry_hold':'At each new usable nonterminal quote, request the maximum allowed exposure until the first filled purchase. Then hold that quantity through the terminal mark. The first purchase remains subject to the original per-order cap, so this is a low-exposure reference.',
    'constant_target':'At every new usable nonterminal quote, request the maximum allowed exposure. Rebalance at the subsequent quote, including sales when above target. Apply the original broker risk limits and per-order cap.',
}


def simulate(ticks,start,steps,costs,policy,decision_seconds=300):
    if policy not in POLICIES:raise ValueError('Unknown baseline policy')
    if type(steps) is not int or steps<1 or decision_seconds<=0:raise ValueError('Invalid phase grid')
    if not ticks or any(b.ts<=a.ts for a,b in zip(ticks,ticks[1:])):
        raise ValueError('Baseline quotes must be strictly chronological')
    broker=Broker(costs);pending=None;entered=False;last_quote=0;rows=[]
    for slot in range(steps+1):
        stamp=start+slot*decision_seconds;_,tick=quote_at(ticks,stamp)
        fill=broker.execute(*pending,tick) if pending else {'status':'hold'};pending=None
        if fill['status']=='filled' and fill['side']=='BUY':entered=True
        eligible=slot<steps and tick.available and tick.ts>last_quote
        target=None
        if eligible:
            if policy=='constant_target' or (policy=='one_entry_hold' and not entered):target=costs.max_exposure
            if target is not None:pending=(target,stamp)
            last_quote=tick.ts
        rows.append({'decision_ts':stamp,'quote_ts':tick.ts,'available':tick.available,
            'terminal':slot==steps,'eligible_decision':eligible,'target_exposure':target,
            'fill':fill,'equity':broker.equity(tick),'broker':broker.state(),
            'inventory_midpoint_usd':float(broker.qty)*tick.mid if tick.available else None})
    filled=[r['fill'] for r in rows if r['fill']['status']=='filled']
    peak=costs.capital;drawdown=0
    for row in rows:
        peak=max(peak,row['equity']);drawdown=max(drawdown,1-row['equity']/peak)
    return {'policy':policy,'definition':POLICIES[policy],'rows':rows,'equity':rows[-1]['equity'],
        'fees_usd':float(broker.fees),'fills':len(filled),'fill_sides':dict(Counter(f['side'] for f in filled)),
        'turnover_usd':sum(float(f['quantity'])*float(f['price']) for f in filled),
        'eligible_decisions':sum(r['eligible_decision'] for r in rows),
        'unavailable_marks':sum(not r['available'] for r in rows),
        'maximum_marked_drawdown_pct':100*drawdown,
        'terminal_quote_usable':rows[-1]['available'],
        'note':'Equity includes paid fees and remaining inventory marked at bid less exit fee/slippage. Unavailable inventory is stress-marked at zero, not sold. Exposure differs across policies; this is not a risk-matched causal comparison. Hosting is excluded.'}


def build(root,output):
    root,output=Path(root),Path(output)
    if output.exists():raise ValueError('Preserve existing benchmark evidence')
    envelope=json.loads((root/'plan.json').read_text());study=envelope['plan']['registration']['study']
    if study=='11':
        from .fly_online_figure import evidence
        report,_,_,fixture=evidence(root)
        p=envelope['plan'];provenance={'completed_study_report_sha256':digest(root/'report.json'),
                                    'price_audit':report['price_audit']}
    elif study=='12':
        from .fly_rate_inputs import validate
        from .fly_rate_price_audit import audit_prices
        p=validate(envelope);prices=audit_prices(envelope,root/'universe.db')
        fixture=any('synthetic' in t['source'] for series in p['series'].values() for t in series)
        provenance={'price_audit':prices}
    else:raise ValueError('Only sealed studies 11 and 12 are supported')
    r=p['registration'];costs=Costs(**r['costs'])
    if len(r['cohort'])!=2 or costs.capital!=250 or r['phase_steps']!=24:
        raise ValueError('Expected two $250 sleeves and 24 decisions per phase')
    results={};aggregate={}
    for stage in ('development','test'):
        results[stage]={};aggregate[stage]={}
        for policy in POLICIES:
            pair=[simulate([Tick(**t) for t in p['series'][key]],r[stage+'_start'],r['phase_steps'],costs,policy)
                  for key in r['cohort']]
            results[stage][policy]=dict(zip(r['cohort'],pair))
            equities=[500+sum(v['rows'][slot]['equity'] for v in pair) for slot in range(25)]
            aggregate[stage][policy]={'initial_capital_usd':1000,'idle_cash_usd':500,
                'equity':equities[-1],'net_pnl_usd':equities[-1]-1000,'equity_path':equities,
                'fills':sum(v['fills'] for v in pair),'fees_usd':sum(v['fees_usd'] for v in pair),
                'turnover_usd':sum(v['turnover_usd'] for v in pair),
                'coverage':[v['eligible_decisions'] for v in pair],
                'terminal_quotes_usable':all(v['terminal_quote_usable'] for v in pair)}
    result={'status':'descriptive_market_baselines','study':study,'validation_only':fixture,
        'plan_sha256':envelope['sha256'],'policies':POLICIES,'costs':asdict(costs),'cohort':r['cohort'],
        'aggregate':aggregate,'sleeves':results,'provenance':provenance,
        'analysis_source_sha256':digest(__file__),'broker_source_sha256':digest(Path(__file__).with_name('core.py')),
        'quote_source_sha256':digest(Path(__file__).with_name('fly_market_study.py')),
        'new_neural_observations':0,'cloud_submissions':0,'selection_modified':False,
        'interpretation':'Descriptive references added after study 11 completion and after study 12 collection began. These are not preregistered selection arms. All use fresh phase accounts and the same quote eligibility and delayed execution assumptions. One-entry holding uses only one capped purchase per sleeve; constant target can rebalance repeatedly. Differing exposure and turnover prevent treating return differences as isolated learning effects. No policy is selected or promoted; hosting is excluded.'}
    output.mkdir(parents=True);atomic_json(output/'report.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=build(a.root,a.out)
    print(json.dumps({k:r[k] for k in ('status','study','validation_only','aggregate')},indent=2))


if __name__=='__main__':main()
