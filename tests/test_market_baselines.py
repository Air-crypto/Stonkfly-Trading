"""Reference policy mechanics, delayed prices and risk limits; no neural model."""
from dataclasses import replace

import pytest

from paperlab.core import Costs,Tick
from paperlab.market_baselines import simulate
from test_fly_rate_pipeline import sealed


COSTS=Costs(capital=250,fee_bps=100,slippage_bps=200,max_order=103.02,max_spread_bps=500)


def run(policy,ticks,steps=2):
    return simulate(ticks,100,steps,COSTS,policy,decision_seconds=100)


def test_cash_preserves_capital_through_missing_quotes():
    ticks=[Tick(100,99,101),Tick(200,99,101,available=False),Tick(300,119,121)]
    r=run('cash',ticks)
    assert r['equity']==250 and r['fills']==0 and r['fees_usd']==0
    assert r['unavailable_marks']==1
    assert all(row['target_exposure'] is None for row in r['rows'])


def test_one_entry_has_delayed_fill_and_hand_calculated_terminal_value():
    r=run('one_entry_hold',[Tick(100,99,101),Tick(200,99,101),Tick(300,119,121)])
    assert r['rows'][0]['fill']=={'status':'hold'}
    fill=r['rows'][1]['fill']
    assert fill['decision_ts']==100 and fill['fill_ts']==200
    assert float(fill['quantity'])==1
    assert r['equity']==pytest.approx(261.4036)
    assert r['fees_usd']==pytest.approx(1.0302)
    assert r['fills']==1 and r['rows'][1]['target_exposure'] is None
    assert r['rows'][-1]['terminal'] and r['rows'][-1]['target_exposure'] is None


def test_constant_target_rebalances_but_holding_does_not():
    ticks=[Tick(100,99,101),Tick(200,99,101),Tick(300,990,1010)]
    target=run('constant_target',ticks);hold=run('one_entry_hold',ticks)
    assert target['fill_sides']=={'BUY':1,'SELL':1}
    assert hold['fill_sides']=={'BUY':1}
    assert target['rows'][-1]['fill']['decision_ts']==200


def test_rejected_first_entry_retries_only_after_a_new_usable_quote():
    ticks=[Tick(100,99,101),Tick(200,99,101,available=False),Tick(300,99,101),Tick(400,99,101),Tick(500,99,101)]
    r=run('one_entry_hold',ticks,steps=4)
    assert r['rows'][1]['fill']['reason']=='unavailable_market'
    assert r['rows'][1]['target_exposure'] is None
    assert r['rows'][2]['fill']['status']=='hold'
    assert r['rows'][3]['fill']['decision_ts']==300
    assert r['fills']==1


def test_repeated_quote_is_not_a_new_observation_or_fill():
    r=run('one_entry_hold',[Tick(100,99,101),Tick(300,99,101)])
    assert r['rows'][1]['fill']['reason']=='not_after_decision'
    assert not r['rows'][1]['eligible_decision']
    assert r['fills']==0


def test_appending_future_prices_cannot_change_a_finished_phase():
    ticks=[Tick(100,99,101),Tick(200,99,101),Tick(300,119,121)]
    assert run('constant_target',ticks)==run('constant_target',ticks+[Tick(400,999,1001)])


def test_original_loss_stop_blocks_more_risk():
    ticks=[Tick(100,99,101),Tick(200,99,101),Tick(300,.995,1.005)]
    r=run('constant_target',ticks)
    assert r['rows'][-1]['fill']['reason']=='loss_stop'
    assert r['fills']==1 and r['rows'][-1]['broker']['halted']


def test_unavailable_terminal_inventory_is_not_sold():
    ticks=[Tick(100,99,101),Tick(200,99,101),Tick(300,119,121,available=False)]
    r=run('one_entry_hold',ticks)
    assert r['equity']==pytest.approx(145.9498)
    assert r['rows'][-1]['broker']['qty']=='1'
    assert r['rows'][-1]['fill']=={'status':'hold'}
    assert not r['terminal_quote_usable']


def test_order_limit_and_exposure_limit_are_preserved():
    ticks=[Tick(100,99,101),Tick(200,99,101),Tick(300,99,101)]
    c=replace(COSTS,max_order=25,max_exposure=.25)
    r=simulate(ticks,100,2,c,'constant_target',decision_seconds=100)
    assert all(row['target_exposure'] in (None,.25) for row in r['rows'])
    assert all(float(row['fill']['quantity'])*float(row['fill']['price'])<=25+1e-9
               for row in r['rows'] if row['fill']['status']=='filled')


def test_sealed_study12_baselines_keep_gaps_and_original_plan(sealed):
    from paperlab.market_baselines import build
    from paperlab.core import digest
    _,root,_=sealed;before=digest(root/'plan.json')
    report=build(root,root/'baselines')
    assert report['validation_only'] and report['study']=='12'
    assert report['provenance']['price_audit']['prices_reconstructed_from_snapshot']
    assert report['aggregate']['test']['cash']['equity']==1000
    assert not report['aggregate']['test']['one_entry_hold']['terminal_quotes_usable']
    assert report['aggregate']['test']['one_entry_hold']['coverage'][1]<18
    assert not report['selection_modified'] and report['cloud_submissions']==0
    assert digest(root/'plan.json')==before
    with pytest.raises(ValueError,match='Preserve'):build(root,root/'baselines')


def test_changed_price_archive_cannot_publish_baselines(sealed):
    from paperlab.market_baselines import build
    _,root,_=sealed
    with (root/'universe.db').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='snapshot hash'):build(root,root/'bad-baselines')
    assert not (root/'bad-baselines').exists()
