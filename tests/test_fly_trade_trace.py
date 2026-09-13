"""Delayed target execution and decoder provenance, without a neural model."""
from copy import deepcopy

import pytest

from paperlab.core import Broker, Costs, Tick
from paperlab.fly_trade_trace import trace


def fixture(gap=False):
    costs=Costs(capital=250,fee_bps=100,slippage_bps=200,max_order=100,max_spread_bps=500)
    ticks=[Tick(100,9.9,10.1),Tick(200,9.9,10.1),Tick(300,99,101,available=not gap),Tick(400,99,101)]
    sides=['BUY','BUY',None if gap else 'HOLD',None]
    if gap:
        ticks.append(Tick(500,99,101));sides=['BUY','BUY',None,'BUY',None]
    broker=Broker(costs);pending=None;rows=[]
    for i,(tick,side) in enumerate(zip(ticks,sides)):
        fill=broker.execute(*pending,tick) if pending else {'status':'hold'}
        event=None if side is None else {'side':side,'gate_spikes':1 if side=='BUY' else 0,
            'left_hz':2,'right_hz':4,'difference_hz':2,'equity_reward_usd':i+3,'weight_delta_l2':i+.1}
        rows.append({'decision_ts':tick.ts,'quote_ts':tick.ts,'available':tick.available,
                     'fill':fill,'event':event,'broker':broker.state(),'equity':broker.equity(tick)})
        pending=(.5,tick.ts) if side=='BUY' else None
    return rows,ticks,costs


def test_buy_target_can_own_a_later_sell():
    result=trace(*fixture());first,second,third=result['decisions']
    assert first['decision_ts']==100 and first['fill']['fill_ts']==200
    assert first['fill']['side']=='BUY'
    assert second['decision_ts']==200 and second['fill']['fill_ts']==300
    assert second['decision']=='BUY' and second['fill']['side']=='SELL'
    assert second['buy_target_rebalanced_sell']
    assert result['fill_routes']=={'BUY->BUY':1,'BUY->SELL':1}
    assert result['single_gate_fills']==2
    assert first['feedback_before_decision_usd']==3
    assert second['feedback_before_decision_usd']==4
    assert first['execution_cost_usd']>0 and second['execution_cost_usd']>0
    assert third['execution_cost_usd']==0
    assert sum(d['execution_cost_usd'] for d in result['decisions'])==pytest.approx(result['execution_cost_usd'])


def test_quote_gap_keeps_rejection_and_observation_index():
    result=trace(*fixture(gap=True))
    assert result['decisions'][1]['fill']=={'status':'rejected','reason':'unavailable_market'}
    assert result['decisions'][1]['execution_cost_usd']==0
    assert result['decisions'][2]['slot']==3
    assert result['decisions'][2]['observation']==2
    assert result['decisions'][2]['fill']['decision_ts']==400


@pytest.mark.parametrize('damage', ['owner','decoder','terminal','orphan'])
def test_inconsistent_provenance_rejected(damage):
    rows,ticks,costs=fixture();rows=deepcopy(rows)
    if damage=='owner':rows[1]['fill']['decision_ts']=99
    elif damage=='decoder':rows[0]['event']['gate_spikes']=0
    elif damage=='terminal':rows[-1]['event']=deepcopy(rows[0]['event'])
    else:rows[0]['event']=None
    with pytest.raises(ValueError):trace(rows,ticks,costs)
