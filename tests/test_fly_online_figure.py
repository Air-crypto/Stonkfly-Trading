"""Figure wiring uses synthetic diagnostics and a real indicative-price ledger.

The native audit itself has separate full-graph tests. These explicitly labeled
fixtures are never market evidence, even when rendering or consistency passes.
"""
import json
from pathlib import Path

import pytest

from paperlab.core import Broker, Tick, atomic_json, digest
from paperlab.fly_market_study import quote_at, signature
from paperlab.fly_online_figure import CHECKS, evidence, render
from paperlab.fly_online_protocol import ARMS, chunk_name, chunk_order
from paperlab.fly_online_schedule import aggregate
from paperlab.fly_online_study import select_development
from paperlab.fly_paper_price_audit import audit_prices
from paperlab.multi import DEX_COSTS
from test_fly_online import sealed


@pytest.fixture
def figure_evidence(sealed):
    env,base=sealed;r=env['plan']['registration'];root=base/'figure-evidence'
    atomic_json(root/'plan.json',env);prices=audit_prices(env,base/'universe.db')
    atomic_json(root/'price-audit.json',prices);summaries={};audits={};receipts={};selected=None
    for i,(stage,pool,arm) in enumerate(chunk_order()):
        if i==8:selected=select_development(env,summaries.copy())
        name=chunk_name(stage,pool,arm);broker=Broker(DEX_COSTS);pending=None;rows=[];observed=[]
        ticks=[Tick(**x) for x in env['plan']['series'][r['cohort'][pool]]]
        last_quote=0;anchor=250;unpriced=False
        for slot in range(25):
            stamp=r[stage+'_start']+slot*300;_,t=quote_at(ticks,stamp)
            fill=broker.execute(*pending,t) if pending else {'status':'hold'};pending=None
            equity=broker.equity(t);event=None
            if slot<24 and t.available and t.ts>last_quote:
                learning=ARMS[arm]['learning']
                # A reset candidate that remains in cash must fail development
                # even if all other conditions lose money after costs.
                side='HOLD' if arm=='trained_online_reset_rates' else 'BUY'
                gate=0 if side=='HOLD' else (slot%5)+1
                norm=(.1+slot*.01) if learning else 0
                reward=(0 if unpriced else equity-anchor) if learning else 0
                event={'side':side,'gate_spikes':gate,'weight_delta_l2':norm,
                    'plasticity_enabled':learning,'equity_reward_usd':reward,'synthetic_figure_diagnostic':True}
                earlier=0 if arm=='trained_online_reset_rates' or slot==0 else slot/10
                current=0 if slot==0 else 1
                observed.append({'decision_ts':stamp,'side':side,'gate_spikes':gate,
                    'weight_norm_audit':{'reported':norm},'equity_reward_usd':reward,
                    'rule':{'integrated_absolute_drive':{'earlier':earlier,'current':current}}})
                if side!='HOLD':pending=(.5 if side=='BUY' else 0,stamp)
                last_quote=t.ts;anchor=equity
            unpriced=not t.available
            rows.append({'decision_ts':stamp,'quote_ts':t.ts,'equity':equity,'available':t.available,
                'fill':fill,'event':event,'broker':broker.state(),'terminal':slot==24})
        outcome={'rows':rows,'equity':rows[-1]['equity'],'fees':float(broker.fees),
            'fills':sum(x['fill']['status']=='filled' for x in rows)}
        summary={'status':'paper_online_chunk_completed','chunk':name,'plan_sha256':env['sha256'],'registration':r,
            'stage':stage,'pool_index':pool,'arm':arm,'outcome':outcome,'selection':selected,
            'artifact_sha256':{},'code_sha256':{},'synthetic_figure_fixture':True}
        atomic_json(root/f'chunks/{name}/summary.json',summary);summaries[name]=summary
        audit={'status':'paper_online_chunk_audited','chunk':name,'plan_sha256':env['sha256'],
            'observations':observed,'artifact_sha256':{},'executed_source_sha256':{},'synthetic_figure_fixture':True,
            'verification':{'decision_slots':25,'observations':len(observed),'native_bins':50*len(observed),
                'ledger_replayed':True,'feedback_reconstructed':True,'all_boundaries_verified':True,'all_weights_exact':True}}
        atomic_json(root/f'audits/{name}.json',audit);audits[name]=digest(root/f'audits/{name}.json')
        receipt={'call_id':f'fixture-{i}','input_id':f'fixture-input-{i}',
            'summary_sha256':digest(root/f'chunks/{name}/summary.json'),
            'completed_at':r['end']+i*300+120,'claimed_at':r['end']+i*300}
        atomic_json(root/f'receipts/{name}-completed.json',receipt);receipts[name]=receipt
    report=aggregate(env,summaries,selected)
    report.update(status='paper_online_study_audited',audited=True,audit_sha256=audits,price_audit=prices,
        verification={**dict.fromkeys(CHECKS,True),'distinct_completed_worker_calls':16},
        selection_receipt={'call_id':'fixture-8','input_id':'fixture-input-8','selected_at':r['end']+8*300,
            'development_chunk_sha256':selected['development_chunk_sha256']},
        phase_diagnostics={key:{arm:{stage:summaries[chunk_name(stage,pool,arm)]['outcome'] for stage in ('development','test')}
            for arm in ARMS} for pool,key in enumerate(r['cohort'])},synthetic_figure_fixture=True)
    atomic_json(root/'report.json',report)
    return root


def test_figure_reconciles_all_slots_and_keeps_gaps_distinct_from_hold(figure_evidence):
    root=figure_evidence;report,lanes,panels,fixture=evidence(root)
    assert fixture and len(lanes)==16 and len(panels)==8
    assert report['selection']['selected'] is None
    reset=lanes['development-pool0-trained_online_reset_rates']['rows']
    assert len(reset)==25 and reset[-1]['terminal'] and reset[-1]['side'] is None
    assert any(row['side'] is None and not row['terminal'] for row in reset)
    assert any(row['side']=='HOLD' and row['gate_spikes']==0 for row in reset)
    assert all(row['gate_spikes'] is None for row in reset if row['side'] is None)
    assert reset[0]['earlier_component_share_pct'] is None  # No drive is undefined, not zero.
    assert any(row['earlier_component_share_pct']==0 for row in reset[1:] if row['side'])
    assert panels['development','trained_online_reset_rates']['equity']==[1000]*25
    # Target BUY can cause an actual SELL as exposure is rebalanced after a rise.
    rebalances=[row for lane in lanes.values() for row in lane['rows']
        if row['fill']['status']=='filled' and row.get('fill_origin_side')=='BUY' and row['fill']['side']=='SELL']
    assert rebalances and all(row['fill']['decision_ts']<row['quote_ts'] for row in rebalances)


@pytest.mark.parametrize('change',['incomplete','audit_hash','summary_hash','selection','aggregate','timeline','same_time_fill','duplicate_call'])
def test_figure_rejects_inconsistent_or_incomplete_evidence(figure_evidence,change):
    root=figure_evidence;read=lambda path:json.loads((root/path).read_text());report=read('report.json')
    name='development-pool0-trained_online_carry'
    if change=='incomplete':report['audited']=False
    if change=='audit_hash':report['audit_sha256'][name]='f'*64
    if change=='summary_hash':report['chunk_sha256'][name]='f'*64
    if change=='selection':report['selection']['selected']='trained_online_reset_rates'
    if change=='aggregate':report['total_equity']['trained_online_reset_rates']['test']=10001
    if change=='duplicate_call':
        path=f'receipts/{name}-completed.json';receipt=read(path);receipt['call_id']='fixture-0';atomic_json(root/path,receipt)
    if change in ('timeline','same_time_fill'):
        path=f'chunks/{name}/summary.json';summary=read(path);rows=summary['outcome']['rows']
        if change=='timeline':rows[1]['decision_ts']+=1
        else:
            row=next(x for x in rows if x['fill']['status']=='filled')
            row['fill']['decision_ts']=row['decision_ts']
        atomic_json(root/path,summary);report['chunk_sha256'][name]=signature(summary)
        report['phase_diagnostics'][report['registration']['cohort'][0]]['trained_online_carry']['development']=summary['outcome']
        rp=f'receipts/{name}-completed.json';receipt=read(rp);receipt['summary_sha256']=digest(root/path);atomic_json(root/rp,receipt)
    atomic_json(root/'report.json',report)
    with pytest.raises(ValueError):evidence(root)


def test_render_produces_all_five_labeled_figures_and_preserves_input(figure_evidence):
    root=figure_evidence;before=digest(root/'report.json');out=root.parent/'figures'
    result=render(root,out)
    assert result['validation_only'] and len(result['figure_sha256'])==10
    assert digest(root/'report.json')==before
    assert len(result['paired_steps'])==80
    after_gap=next(x for x in result['paired_steps'] if x['phase']=='development' and x['pool']==0 and x['slot']==6)
    assert '&step=4&' in after_gap['viewer_url']  # Observed frame index, not wall-clock slot.
    assert 'Earlier fills' in (out/'diagnostics.md').read_text()
    for name in ('comparison','development-pool0','development-pool1','test-pool0','test-pool1'):
        svg=(out/(name+'.svg')).read_text()
        assert 'SYNTHETIC FIXTURE' in svg and 'no market result' in svg
        assert (out/(name+'.png')).stat().st_size>10000
    with pytest.raises(ValueError,match='Preserve'):render(root,out)
