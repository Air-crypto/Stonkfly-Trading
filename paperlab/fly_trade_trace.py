"""Connect recorded decoder decisions to later fills and their execution costs.

Read-only study 11 postmortem. No model construction or alternative-policy replay.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode

from .core import Costs, Tick, atomic_json, digest
from .fly_loss_attribution import decompose
from .fly_online_figure import evidence
from .fly_online_protocol import ARMS, chunk_name, chunk_order
from .fly_rate_metrics import replay_ledger


def trace(rows, ticks, costs):
    accounting = decompose(rows, ticks, costs)
    decisions = []; observation = 0
    for slot, row in enumerate(rows):
        event = row.get('event')
        if not event:
            continue
        if slot + 1 >= len(rows):
            raise ValueError('An observed decision needs its subsequent execution slot')
        difference = event['right_hz'] - event['left_hz']
        expected = 'HOLD' if event['gate_spikes'] == 0 or abs(difference) < 2 else 'BUY' if difference > 0 else 'SELL'
        if event['difference_hz'] != difference or event['side'] != expected:
            raise ValueError('Recorded decoder does not match its output rates and gate')
        following = rows[slot+1]; fill = following['fill']
        if fill['status'] == 'filled':
            if (event['side'] == 'HOLD' or fill['decision_ts'] != row['decision_ts']
                    or (event['side'] == 'SELL' and fill['side'] != 'SELL')):
                raise ValueError('Fill is not owned by the preceding target decision')
        elif event['side'] == 'HOLD' and fill['status'] != 'hold':
            raise ValueError('HOLD unexpectedly submitted an execution attempt')
        before, after = accounting['rows'][slot:slot+2]
        costs_at_fill = {field:after[field]-before[field] for field in (
            'paid_fees_usd','execution_spread_usd','execution_slippage_usd')}
        decisions.append({'slot':slot,'observation':observation,'decision_ts':row['decision_ts'],
            'decision':event['side'],'target_exposure':None if expected=='HOLD' else .5 if expected=='BUY' else 0,
            'gate_spikes':event['gate_spikes'],'difference_hz':difference,
            'left_hz':event['left_hz'],'right_hz':event['right_hz'],
            'feedback_before_decision_usd':event['equity_reward_usd'],
            'weight_delta_l2':event['weight_delta_l2'],
            'execution_slot':slot+1,'execution_mark_ts':following['decision_ts'],
            'execution_quote_ts':following['quote_ts'],'fill':fill,
            'execution_costs':costs_at_fill,'execution_cost_usd':sum(costs_at_fill.values()),
            'buy_target_rebalanced_sell':expected=='BUY' and fill.get('side')=='SELL'})
        observation += 1
    filled = [d for d in decisions if d['fill']['status']=='filled']
    if len(filled) != sum(row['fill']['status']=='filled' for row in rows):
        raise ValueError('Not every fill has a unique prior decision')
    end=accounting['terminal']
    execution_cost=sum(end[k] for k in ('paid_fees_usd','execution_spread_usd','execution_slippage_usd'))
    if abs(sum(d['execution_cost_usd'] for d in decisions)-execution_cost)>1e-8:
        raise ValueError('Per-decision execution costs do not reconcile')
    return {'decisions':decisions,'fills':len(filled),
        'fill_routes':dict(Counter(d['decision']+'->'+d['fill']['side'] for d in filled)),
        'single_gate_fills':sum(d['gate_spikes']==1 for d in filled),
        'execution_cost_usd':execution_cost,
        'buy_target_rebalanced_sells':sum(d['buy_target_rebalanced_sell'] for d in filled)}


def build(root, output):
    root,output=Path(root),Path(output)
    if output.exists():raise ValueError('Preserve an existing trade trace')
    report,_,_,fixture=evidence(root)
    plan=json.loads((root/'plan.json').read_text())['plan'];r=plan['registration'];costs=Costs(**r['costs'])
    chunks={}
    for stage,pool,arm in chunk_order():
        name=chunk_name(stage,pool,arm);path=root/'chunks'/name/'summary.json'
        s=json.loads(path.read_text());raw=plan['series'][r['cohort'][pool]]
        replay_ledger(s['outcome'],raw,r[stage+'_start'],ARMS[arm])
        item=trace(s['outcome']['rows'],[Tick(**t) for t in raw],costs)
        item.update(summary_sha256=digest(path),stage=stage,pool=pool,arm=arm)
        chunks[name]=item
    for name,item in chunks.items():
        for decision in item['decisions']:
            comparisons={}
            for arm in ('pristine_frozen','trained_online_carry'):
                other=chunks[chunk_name(item['stage'],item['pool'],arm)]
                match=next((d for d in other['decisions'] if d['decision_ts']==decision['decision_ts']),None)
                if match is None or match['observation']!=decision['observation']:
                    raise ValueError('Comparison timestamp or displayed observation differs')
                comparisons[arm]={k:match[k] for k in ('decision','gate_spikes','difference_hz','execution_cost_usd')}
            decision['comparisons']=comparisons
            query={'run':name,'compare':chunk_name(item['stage'],item['pool'],'trained_online_carry'),
                   'step':decision['observation'],'bin':49,'neuron':10527}
            decision['viewer_url']='http://127.0.0.1:8767/?'+urlencode(query)
    aggregate={}
    for stage in ('development','test'):
        aggregate[stage]={}
        for arm in ARMS:
            pair=[chunks[chunk_name(stage,pool,arm)] for pool in range(2)]
            totals={k:sum(v[k] for v in pair) for k in ('fills','single_gate_fills','execution_cost_usd','buy_target_rebalanced_sells')}
            totals['fill_routes']=dict(sum((Counter(v['fill_routes']) for v in pair),Counter()))
            unmatched=[d for v in pair for d in v['decisions'] if d['fill']['status']=='filled'
                       and d['comparisons']['pristine_frozen']['decision']=='HOLD']
            totals['fills_where_pristine_held']=len(unmatched)
            totals['execution_cost_where_pristine_held_usd']=sum(d['execution_cost_usd'] for d in unmatched)
            aggregate[stage][arm]=totals
    result={'status':'audited_decision_fill_trace','study':'11','validation_only':fixture,
            'source_report_sha256':digest(root/'report.json'),'plan_sha256':report['plan_sha256'],
            'chunks':chunks,'aggregate':aggregate,'new_neural_observations':0,'cloud_submissions':0,
            'verification':{'replayed_ledgers':len(chunks),'mapped_decisions':sum(len(v['decisions']) for v in chunks.values()),
                            'mapped_fills':sum(v['fills'] for v in chunks.values())},
            'source_sha256':{name:digest(Path(__file__).with_name(name)) for name in (
                'fly_trade_trace.py','fly_loss_attribution.py','fly_online_figure.py','fly_rate_metrics.py')},
            'interpretation':'Each fill belongs to the preceding observed target decision. BUY targets 50% exposure and can rebalance by selling. Costs are paid fees plus spread and slippage at execution, excluding terminal exit allowance and hosting. Feedback shown precedes this decision and does not measure this new trade. Comparison decisions arise from their own account and neural histories; differences are not isolated causal effects. No performance estimate for a changed policy.'}
    output.mkdir(parents=True);atomic_json(output/'report.json',result)
    return result


def markdown(report, path):
    lines=['# Recorded decisions and later fills','',report['interpretation'],'',
        'All times UTC. Start the study 11 debugger on port 8767 to use the links.',
        'Single-gate counts describe the original decoder; they are not a validated confidence score.','']
    for stage in ('development','test'):
        for pool in range(2):
            name=chunk_name(stage,pool,'trained_online_reset_rates')
            lines += [f'## {stage} / pool {pool} / online reset','',
                '| Decision UTC | Reset decision | Gate / R−L Hz | Carry / pristine decision | Later fill | Execution costs | Trace |',
                '| --- | --- | --- | --- | --- | ---: | --- |']
            for d in report['chunks'][name]['decisions']:
                fill=d['fill'];stamp=lambda x:datetime.fromtimestamp(x,timezone.utc).strftime('%m-%d %H:%M:%S')
                outcome=fill['status'] if fill['status']!='filled' else fill['side']+' at '+stamp(fill['fill_ts'])
                comparisons=d['comparisons'];cost=d['execution_cost_usd']
                lines.append(f"| {stamp(d['decision_ts'])} | {d['decision']} | {d['gate_spikes']} / {d['difference_hz']:g} | {comparisons['trained_online_carry']['decision']} / {comparisons['pristine_frozen']['decision']} | {outcome} | ${cost:.4f} | [Inspect]({d['viewer_url']}) |")
            lines.append('')
    Path(path).write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    report=build(args.root,args.out);markdown(report,args.out/'decisions.md')
    print(json.dumps(report['aggregate'],indent=2))


if __name__=='__main__':main()
