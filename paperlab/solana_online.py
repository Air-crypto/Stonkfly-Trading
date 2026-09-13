"""Shared paper portfolio and cross-launch online learning, cloud propagation only."""
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import json
import time
import requests

from .core import Broker, Tick, atomic_json
from .solana_events import Feed
from .solana_paper import COSTS, Readout, inputs, prior_state, tick_for

MAX_ACTIVE = 8
MAX_RISK_USD = Decimal('100')
MAX_ENTRY_DEBIT_USD = Decimal('2.50')
CASH_FLOOR_USD = Decimal('902')  # Original $900 loss stop plus a $2 execution buffer.
MIN_ENTRY_DEBIT_USD = Decimal('1.05')


class Portfolio:
    """One cash balance, per-mint inventory and cash flows; no free capital on admission."""
    def __init__(self, state):
        self.cash = Decimal(state['cash'])
        self.fees = Decimal(state['fees'])
        self.halted = bool(state.get('halted', False))
        self.positions = {m: {k: Decimal(v) for k, v in p.items()} for m, p in state['positions'].items()}
        self.quarantined = set(state.get('quarantined', []))
        if not self.quarantined <= self.positions.keys(): raise ValueError('Unknown quarantined inventory')
        if self.cash < 0 or any(p['qty'] < 0 or p['basis'] < 0 for p in self.positions.values()):
            raise ValueError('Invalid portfolio')

    def position(self, mint):
        return self.positions.setdefault(mint, dict(qty=Decimal(0), basis=Decimal(0), cash_flow=Decimal(0)))

    def state(self):
        return dict(cash=str(self.cash), fees=str(self.fees), halted=self.halted, quarantined=sorted(self.quarantined),
                    positions={m: {k: str(v) for k, v in p.items()} for m, p in self.positions.items()})

    def view(self, mint):
        return Broker(COSTS, state=dict(cash=str(self.cash), qty=str(self.position(mint)['qty']),
                                       fees=str(self.fees), halted=self.halted))

    def value(self, mint, tick):
        p = self.position(mint)
        return float(p['qty']) * tick.bid * .99 * .9875 if tick and tick.available else 0.

    def contribution(self, mint, tick):
        return float(self.position(mint)['cash_flow']) + self.value(mint, tick)

    def equity(self, quotes):
        return float(self.cash) + sum(self.value(m, quotes.get(m)) for m in self.positions)

    def entry_budget(self, mint=None):
        """Quarantine changes allocation only; it never restores cash, basis, or equity."""
        active_basis = sum(p['basis'] for m,p in self.positions.items() if m not in self.quarantined)
        room = min(MAX_RISK_USD-active_basis, self.cash-CASH_FLOOR_USD)
        if mint is not None: room = min(room, MAX_ENTRY_DEBIT_USD-self.position(mint)['basis'])
        return max(Decimal(0), room) if not self.halted else Decimal(0)

    def allowed_actions(self, mint):
        return (0,1) if self.position(mint)['qty'] or self.entry_budget(mint)>=MIN_ENTRY_DEBIT_USD else (0,)

    def target(self, mint, action, tick):
        if action == 0: return 0.
        # LONG holds an existing lot; it does not repeatedly request more blocked exposure.
        if self.position(mint)['qty']:
            return min(COSTS.max_exposure, float(self.position(mint)['qty'])*tick.mid/self.view(mint).equity(tick))
        return COSTS.max_exposure

    def execute(self, mint, target, issued, tick, quotes):
        if self.equity(quotes) <= 900: self.halted = True
        b = self.view(mint); before = b.cash; before_qty = b.qty
        requested = Decimal(str(b.equity(tick)*target/tick.mid))-b.qty
        buying = requested*Decimal(str(tick.mid)) >= 1
        budget = self.entry_budget(mint)
        if buying and not self.halted:
            if budget < MIN_ENTRY_DEBIT_USD:
                return dict(status='rejected', reason='entry_risk_budget', entry_budget_usd=str(budget))
            # Re-size on the actual later receipt, with fees inside the debit allowance.
            b.c = replace(COSTS, max_order=float(budget/Decimal('1.0125'))*.999999)
        fill = b.execute(target, issued, tick)
        if fill['status'] != 'filled': return fill
        p = self.position(mint); debit = before - b.cash
        if fill['side'] == 'BUY' and debit > budget:
            return dict(status='rejected', reason='entry_risk_budget')
        p['basis'] = p['basis'] + debit if fill['side'] == 'BUY' else p['basis'] * b.qty / before_qty
        p['qty'] = b.qty; p['cash_flow'] += b.cash - before
        self.cash = b.cash; self.fees = b.fees; self.halted |= b.halted
        return fill


def load_parent(path):
    path = Path(path)
    if (path/'online-state.json').exists():
        completed = json.loads((path/'completed.json').read_text())
        if completed['status'] != 'completed' or not completed['paper_only']:
            raise ValueError('Resume only completed paper windows')
        return json.loads((path/'online-state.json').read_text())
    saved = prior_state(path); b = saved['broker']; mint = saved['selected_mint']
    # Legacy records have one cash account. Preserve its inventory and all net losses.
    debit = Decimal(1000) - Decimal(b['cash'])
    return dict(portfolio=dict(cash=b['cash'], fees=b['fees'], halted=b['halted'], positions={mint:
        dict(qty=b['qty'], basis=str(max(Decimal(0), debit)), cash_flow=str(-debit))}),
        active={mint: saved['created']}, retired=[], native_checkpoint=saved['native_checkpoint'],
        head_checkpoint=saved['head_checkpoint'], native_observations_total=0,
        migration='Legacy acquisition-cost risk reserve uses net cash spent; old ledger remains immutable.')


def choose(candidates, current, burst, visits):
    """Three observations per burst, then least-recently served; no return ranking."""
    if current in candidates and burst < 3: return current, burst + 1
    remaining = [m for m in candidates if m != current] or list(candidates)
    if not remaining: return None, 0
    return min(remaining, key=lambda m: (visits.get(m, 0), m)), 1



def update_active(active, watched, retired, inactive, portfolio, snapshots, quotes, now, connected=True):
    """Quarantine unavailable holdings after healthy-feed observation; retain the entire ledger."""
    removed = []; admitted = []
    if not connected:
        inactive.clear(); return admitted, removed
    for m in quotes: portfolio.quarantined.discard(m)
    # Include parked/watched inventory on subsequent windows, not just active tokens.
    for m in watched:
        if m in quotes: inactive.pop(m, None)
        elif portfolio.position(m)['qty']:
            inactive.setdefault(m, now)
            if now-inactive[m] >= 120: portfolio.quarantined.add(m)
    for m in list(active):
        qty = portfolio.position(m)['qty']; t = quotes.get(m)
        if t: inactive.pop(m, None)
        else: inactive.setdefault(m, now)
        idle = m in inactive and now-inactive[m] >= 120
        dust = qty > 0 and t is not None and float(qty)*t.mid < 1
        expired_flat = qty == 0 and now-active[m]['timestamp'] >= 1200
        if idle or dust or expired_flat:
            active.pop(m); removed.append(m)
            if qty == 0: retired.add(m); watched.pop(m, None)
    candidates = []
    for m, t in quotes.items():
        if m in active or m in retired: continue
        qty = portfolio.positions.get(m, {}).get('qty', Decimal(0))
        if qty and float(qty)*t.mid < 1: continue
        candidates.append((0 if qty else 1, snapshots[m]['created']['received'], m))
    for _, _, m in sorted(candidates)[:max(0, MAX_ACTIVE-len(active))]:
        active[m] = snapshots[m]['created']; watched[m] = active[m]
        portfolio.position(m); admitted.append(m)
    return admitted, removed

def run(root, data, parent, *, seconds=900, commit=lambda: None, fly_factory=None,
        feed_factory=Feed, clock=time.time, sleep=time.sleep, fx_fetch=None):
    if not 60 <= seconds <= 900: raise ValueError('Bounded window required')
    if fly_factory is None:
        import modal
        if modal.is_local(): raise RuntimeError('Native fly must run in Modal')
        from .fly import Fly
        fly_factory = Fly
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root/'started.json').exists(): raise ValueError('Immutable window')
    saved = load_parent(parent); portfolio = Portfolio(saved['portfolio'])
    active = dict(saved['active']); watched = dict(saved.get('watched', active)); retired = set(saved['retired'])
    head = Readout(); head.restore(saved['head_checkpoint']); opening_updates = head.updates
    feed = feed_factory(root)
    for created in watched.values(): feed.accept(created)
    feed.pinned_mints = set(watched); feed.start()
    started = clock(); deadline = started + seconds
    atomic_json(root/'opening.json', saved)
    atomic_json(root/'started.json', dict(at=started, deadline=deadline, paper_only=True,
        max_active_tokens=MAX_ACTIVE, shared_capital=1000, interval_seconds=5, parent=str(parent),
        coverage='Pump launches received during this window; bounded admission, not every token',
        risk_policy='quarantined_inventory_cash_floor_v1', max_tradable_acquisition_cost_usd=100,
        max_new_position_debit_usd=2.5, minimum_cash_after_buy_usd=902, news_enabled=False))
    commit()
    fly = None; current = None; burst = 0; visits = {}; contexts = {}; inactive = {}
    rows = trained = fills = overruns = nonzero_rewards = rejected_credit = 0
    status = 'collecting'; next_step = started; next_fx = 0.
    fx = fx_seen = 0.; reasons = Counter(); snapshots = {}; quotes = {}
    def fetch_fx():
        r = requests.get('https://api.coinbase.com/v2/prices/SOL-USD/spot', timeout=4)
        r.raise_for_status(); return float(r.json()['data']['amount'])
    fx_fetch = fx_fetch or fetch_fx
    def context(m):
        return contexts.setdefault(m, dict(ticks=[], signature=None, previous=None, pending=None,
                                          anchor=None, last_ts=None))
    ledger = open(root/'decisions.jsonl', 'a'); fxlog = open(root/'fx.jsonl', 'a')
    try:
        while clock() < deadline:
            now = clock()
            if now < next_step: sleep(min(1, next_step-now)); continue
            step_start = now; next_step = now + 5
            if now >= next_fx:
                next_fx = now + 60
                try:
                    fx = fx_fetch(); fx_seen = clock()
                    if not 0 < fx < 1e7: raise ValueError('Invalid FX')
                    fxlog.write(json.dumps(dict(at=fx_seen, sol_usd=fx))+'\n'); fxlog.flush()
                except Exception as exc:
                    fx = 0.; print(json.dumps(dict(event='fx_error', error=type(exc).__name__)), flush=True)
            now = clock(); snapshots = feed.snapshot(); health = feed.health()
            if health['status'] in ('failed', 'capacity_stopped'): status = health['status']; break
            quotes = {}; rejected = {}; admissions = []; removals = []
            connected = health['status'] == 'connected' and 0 <= now-health['last_message'] <= 10
            for m, token in snapshots.items():
                t, reason = tick_for(token, now, fx, fx_seen, selected=m in watched)
                if not connected: reason = 'disconnected_feed'
                if reason is None and t: quotes[m] = t
                else: rejected[m] = reason or 'missing_quote'
            admissions, removals = update_active(active, watched, retired, inactive,
                                                  portfolio, snapshots, quotes, now, connected=connected)
            for m in removals: contexts.pop(m, None)  # Pending orders and credit expire; inventory stays.
            with feed.lock: feed.pinned_mints = set(watched)
            if len(watched) > 128: status = 'capacity_stopped'; break
            if len(retired) + len(active) > 10000: status = 'capacity_stopped'; break
            fresh = []; executions = []
            for m in active:
                c = context(m); t = quotes.get(m)
                if t is None:
                    reasons[rejected.get(m, 'missing_token')] += 1
                    c['previous'] = None; c['last_ts'] = None
                    if c['pending'] and now-c['pending']['issued'] > 15: c['pending'] = None
                    continue
                last = snapshots[m]['trades'][-1]; sig = (last['signature'], last['log_index'])
                if sig == c['signature']: continue
                c['signature'] = sig
                if c['pending']:
                    fill = portfolio.execute(m, c['pending']['target'], c['pending']['issued'], t, quotes)
                    executions.append(dict(mint=m, tick=t.__dict__, fill=fill))
                    fills += fill['status'] == 'filled'
                    if fill['status']=='rejected' and fill.get('reason')!='not_after_decision':
                        c['previous']=None; rejected_credit+=1
                    if fill.get('reason') != 'not_after_decision': c['pending'] = None
                c['ticks'].append(t); c['ticks'] = c['ticks'][-100:]
                if len(c['ticks']) >= 12 and c['pending'] is None: fresh.append(m)
            neural = None; learning = None; decision = None
            picked, next_burst = choose(fresh, current, burst, visits)
            if picked and deadline-clock() > 30:
                m = picked; c = context(m); t = quotes[m]
                if fly is None:
                    fly = fly_factory(data, learning=True, checkpoint=saved['native_checkpoint'])
                    fly.controller.brain.reset(keep_memory=True)
                if clock()-t.received_at <= 10:
                    switched = m != current
                    if switched: fly.controller.brain.reset(keep_memory=True)
                    contribution = portfolio.contribution(m, t)
                    continuous = c['last_ts'] is not None and 0 < t.ts-c['last_ts'] <= 60
                    delta = contribution-c['anchor'] if continuous else 0.
                    if not continuous: c['previous'] = None
                    # Native eligibility belongs to the preceding token. Clear it on switching;
                    # only the head may assign delayed, mint-specific credit across other tokens.
                    from .news import News
                    native_path = root/'neural'/f'{trained:05d}' if trained % 12 == 0 else None
                    neural = fly.observe(c['ticks'], len(c['ticks'])-1, News(enabled=False),
                                         delta if not switched and continuous else 0., native_path)
                    x = inputs(c['ticks'], snapshots[m], portfolio.view(m), neural, now)
                    x[10] = max(-5., min(5., contribution/25))
                    allowed = portfolio.allowed_actions(m)
                    if c['previous'] is not None:
                        learning = head.update(c['previous'], x, delta, t.ts-c['previous']['ts'], allowed_actions=allowed)
                        nonzero_rewards += abs(delta)>1e-12
                    action, prediction = head.choose(x, allowed_actions=allowed); issued = clock()
                    if issued-t.received_at <= 15 and c['pending'] is None:
                        c['pending'] = dict(target=portfolio.target(m,action,t), issued=issued)
                        c['previous'] = dict(x=x, action=action, ts=t.ts)
                        decision = dict(mint=m, action=action, issued=issued, target=c['pending']['target'])
                    else: c['previous'] = None
                    c['anchor'] = contribution; c['last_ts'] = t.ts
                    visits[m] = rows+1; current = m; burst = next_burst; trained += 1
                    neural.update(mint=m, tick=t.__dict__, contribution_usd=contribution, switched_token=switched, head_training=learning,
                                  readout=prediction, input_features=x.tolist(),
                                  diagnostic_path=str(native_path) if native_path else None)
                    status = 'training'
            elapsed = clock()-step_start; overruns += elapsed > 5
            row = dict(at=clock(), step=rows, paper_only=True, status=status, admissions=admissions,
                retired=removals, active_tokens=len(active), tracked_launches=len(snapshots),
                eligible_tokens=len(quotes), executions=executions, decision=decision, neural=neural,
                portfolio=portfolio.state(), equity_stress_usd=portfolio.equity(quotes),
                marks={m:t.__dict__ for m,t in quotes.items() if portfolio.positions.get(m, {}).get('qty', 0)},
                unavailable_positions=[m for m,p in portfolio.positions.items() if p['qty'] and m not in quotes],
                feed=health, neural_observations=trained, readout_updates=head.updates,
                new_readout_updates=head.updates-opening_updates, fills=fills, overruns=overruns,
                risk_policy='quarantined_inventory_cash_floor_v1', entry_budget_usd=str(portfolio.entry_budget()),
                nonzero_reward_updates=nonzero_rewards, rejected_order_credit_dropped=rejected_credit,
                step_seconds=elapsed, skip_reasons=dict(reasons))
            ledger.write(json.dumps(row, allow_nan=False)+'\n'); ledger.flush()
            atomic_json(root/'latest.json', row); rows += 1
            print(json.dumps({k:v for k,v in row.items() if k not in ('neural','portfolio','feed')}, allow_nan=False), flush=True)
            if neural: print(json.dumps(dict(event='online_learning', mint=current,
                native=neural['learning_diagnostics'], head=learning)), flush=True)
            if rows % 12 == 0: commit()
        if clock() >= deadline: status = 'completed'
    finally:
        feed.close(); ledger.close(); fxlog.close()
    if fly:
        fly.save(root/'fly-final.npz'); head.save(root/'head-final.pt')
    atomic_json(root/'online-state.json', dict(portfolio=portfolio.state(), active=active,
        watched=watched, retired=sorted(retired), native_checkpoint=str(root/'fly-final.npz') if fly else saved['native_checkpoint'],
        head_checkpoint=str(root/'head-final.pt') if fly else saved['head_checkpoint'],
        native_observations_total=saved.get('native_observations_total', 0)+trained,
        readout_updates=head.updates))
    result = dict(status=status, started=started, ended=clock(), steps=rows, paper_only=True,
        neural_observations=trained, readout_updates=head.updates, new_readout_updates=head.updates-opening_updates,
        fills=fills, active_tokens=len(active), portfolio=portfolio.state(),
        risk_policy='quarantined_inventory_cash_floor_v1', entry_budget_usd=str(portfolio.entry_budget()),
        nonzero_reward_updates=nonzero_rewards, rejected_order_credit_dropped=rejected_credit,
        tradable_positions=[m for m,p in portfolio.positions.items() if p['qty'] and portfolio.value(m,quotes.get(m))>=1],
        unavailable_positions=[m for m,p in portfolio.positions.items() if p['qty'] and m not in quotes],
        equity_stress_usd=portfolio.equity(quotes), skip_reasons=dict(reasons), feed=feed.health(),
        profitable_learning_proven=False)
    atomic_json(root/'result.json', result); commit(); return result
