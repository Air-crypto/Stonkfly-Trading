"""Cost and turnover accounting from a phase's independently replayable ledger."""
from decimal import Decimal
import math


def replay_ledger(outcome, raw_ticks, start, arm):
    """Reconstruct account execution and feedback before using selection equity."""
    from .core import Broker, Tick
    from .fly_market_study import quote_at
    from .multi import DEX_COSTS
    rows=outcome['rows'];ticks=[Tick(**t) for t in raw_ticks]
    if len(rows)!=25:raise ValueError('Incomplete phase ledger')
    if any(type(outcome.get(k)) not in (int,float) or not math.isfinite(outcome[k]) for k in ('equity','fees')):
        raise ValueError('Nonfinite phase outcome')
    broker=Broker(DEX_COSTS);pending=None;anchor=250.;unpriced=False;last_quote=0;coverage=[]
    for slot,row in enumerate(rows):
        stamp=start+slot*300;_,t=quote_at(ticks,stamp)
        if (row['decision_ts']!=stamp or row['quote_ts']!=t.ts or row['available']!=t.available
                or row['terminal']!=(slot==24)):
            raise ValueError('Ledger quote or decision timeline differs')
        fill=broker.execute(*pending,t) if pending else {'status':'hold'};pending=None
        equity=broker.equity(t)
        if type(row['equity']) not in (int,float) or not math.isfinite(row['equity']):
            raise ValueError('Nonfinite account equity')
        if row['fill']!=fill or row['broker']!=broker.state() or abs(row['equity']-equity)>1e-8:
            raise ValueError('Account fills or equity do not replay')
        event=row['event'];expected=slot<24 and t.available and t.ts>last_quote
        if (event is not None)!=expected:raise ValueError('Ledger observation coverage differs')
        if event is not None:
            if type(event['equity_reward_usd']) not in (int,float) or not math.isfinite(event['equity_reward_usd']):
                raise ValueError('Nonfinite feedback')
            reward=(0 if unpriced else equity-anchor) if arm['learning'] else 0
            stimulus='reward' if reward>.01 else 'aversive' if reward<-.01 else 'none'
            if (event['side'] not in ('BUY','SELL','HOLD') or event['market_decision_ts']!=stamp
                    or event['plasticity_enabled']!=arm['learning'] or event['stimulus']!=stimulus
                    or abs(event['equity_reward_usd']-reward)>1e-8):
                raise ValueError('Ledger decision or feedback differs')
            if event['side']!='HOLD':pending=(.5 if event['side']=='BUY' else 0,stamp)
            last_quote=t.ts;anchor=equity;coverage.append(stamp)
        unpriced=not t.available
    metrics=trading_metrics(rows)
    if (abs(outcome['equity']-rows[-1]['equity'])>1e-8 or outcome['fills']!=metrics['fills']
            or abs(outcome['fees']-metrics['fees_usd'])>1e-8
            or outcome['unavailable_marks']!=metrics['unavailable_marks']):
        raise ValueError('Phase outcome does not reconcile with its replayed ledger')
    return {'coverage':coverage,'metrics':metrics,'equity':rows[-1]['equity']}


def trading_metrics(rows):
    if len(rows)!=25 or not rows[-1]['terminal'] or any(r['terminal'] for r in rows[:-1]):
        raise ValueError('Require all 24 decisions and the terminal mark')
    notionals={'BUY':Decimal(0),'SELL':Decimal(0)};fees=Decimal(0);fills=0
    signals={side:0 for side in ('BUY','SELL','HOLD')};peak=250.;drawdown=0.
    for row in rows:
        equity=float(row['equity'])
        if not math.isfinite(equity) or equity<0:raise ValueError('Invalid marked equity')
        peak=max(peak,equity);drawdown=max(drawdown,1-equity/peak)
        event=row['event']
        if event is not None:signals[event['side']]+=1
        fill=row['fill']
        if fill['status']=='filled':
            qty,price,fee=(Decimal(str(fill[k])) for k in ('quantity','price','fee'))
            if not all(x.is_finite() for x in (qty,price,fee)) or qty<=0 or price<=0 or fee<0:
                raise ValueError('Invalid fill accounting')
            notionals[fill['side']]+=qty*price;fees+=fee;fills+=1
    total=sum(notionals.values())
    return {'fills':fills,'fees_usd':float(fees),'buy_notional_usd':float(notionals['BUY']),
            'sell_notional_usd':float(notionals['SELL']),'turnover_usd':float(total),
            'turnover_over_initial_capital':float(total/Decimal(250)),
            'signals':signals,'maximum_marked_drawdown_pct':100*drawdown,
            'unavailable_marks':sum(not row['available'] for row in rows),
            'marking_note':'Missing inventory is stress-marked at zero; this is not an observed market crash.',
            'equity_includes_execution_costs':True,'hosting_included_in_equity':False}
