# Versioned experimental fork. Baseline and its sealed source are unchanged.
# Only cadence and injectable replay head differ; account/credit audit is reused.
# Baseline source SHA256: a316c7ed00eecf067e5d4234fe11ee2c9a0f3dd0ad233bcbab0984063303b5aa
"""Shared paper portfolio and cross-launch online learning, cloud propagation only."""
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import json
import time
import requests

from paperlab.core import Broker, Tick, atomic_json
from paperlab.solana_events import Feed
from paperlab.solana_paper import COSTS, Readout, inputs, prior_state, tick_for
from paperlab.solana_quotes import QUOTE_PROTOCOL, quote_for

MAX_ACTIVE = 8
MAX_RISK_USD = Decimal('100')
MAX_ENTRY_DEBIT_USD = Decimal('2.50')
CASH_FLOOR_USD = Decimal('902')  # Original $900 loss stop plus a $2 execution buffer.
MIN_ENTRY_DEBIT_USD = Decimal('1.05')
ACCOUNT_MODES = ('continuous', 'fresh_training_episode')
EPISODE_TOKEN_CAP_USD = Decimal('250')
EPISODE_POLICY = 'fresh_training_full_cash_v1'
REWARD_PROTOCOL = 'mint_credit_terminal_v2'


class Portfolio:
    """One cash balance, per-mint inventory and cash flows; no free capital on admission."""
    def __init__(self, state):
        self.episodic = state.get('risk_policy') == EPISODE_POLICY
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
        result = dict(cash=str(self.cash), fees=str(self.fees), halted=self.halted, quarantined=sorted(self.quarantined),
                    positions={m: {k: str(v) for k, v in p.items()} for m, p in self.positions.items()})
        if self.episodic: result['risk_policy'] = EPISODE_POLICY
        return result

    def view(self, mint):
        costs=replace(COSTS,max_exposure=1.,max_order=1000.,loss_stop=1.) if self.episodic else COSTS
        return Broker(costs, state=dict(cash=str(self.cash), qty=str(self.position(mint)['qty']),
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
        if self.episodic:
            room=self.cash
            if mint is not None:room=min(room,EPISODE_TOKEN_CAP_USD-self.position(mint)['basis'])
            return max(Decimal(0),room) if not self.halted else Decimal(0)
        active_basis = sum(p['basis'] for m,p in self.positions.items() if m not in self.quarantined)
        room = min(MAX_RISK_USD-active_basis, self.cash-CASH_FLOOR_USD)
        if mint is not None: room = min(room, MAX_ENTRY_DEBIT_USD-self.position(mint)['basis'])
        return max(Decimal(0), room) if not self.halted else Decimal(0)

    def allowed_actions(self, mint):
        return (0,1) if self.position(mint)['qty'] or self.entry_budget(mint)>=MIN_ENTRY_DEBIT_USD else (0,)

    def target(self, mint, action, tick):
        if action == 0: return 0.
        if self.episodic and self.entry_budget(mint)>=MIN_ENTRY_DEBIT_USD:
            return 1.
        # LONG holds an existing lot; it does not repeatedly request more blocked exposure.
        if self.position(mint)['qty']:
            return min(1. if self.episodic else COSTS.max_exposure, float(self.position(mint)['qty'])*tick.mid/self.view(mint).equity(tick))
        return 1. if self.episodic else COSTS.max_exposure

    def execute(self, mint, target, issued, tick, quotes, *, liquidity_notional_usd=None):
        if self.equity(quotes) <= (0 if self.episodic else 900): self.halted = True
        b = self.view(mint); before = b.cash; before_qty = b.qty
        requested = Decimal(str(b.equity(tick)*target/tick.mid))-b.qty
        buying = requested*Decimal(str(tick.mid)) >= 1
        budget = self.entry_budget(mint)
        if self.episodic:
            import math
            if liquidity_notional_usd is None or not math.isfinite(liquidity_notional_usd) or liquidity_notional_usd<1:
                return dict(status='rejected',reason='insufficient_observed_liquidity')
            b.c=replace(b.c,max_order=min(1000.,liquidity_notional_usd))
        if buying and not self.halted:
            if budget < MIN_ENTRY_DEBIT_USD:
                return dict(status='rejected', reason='entry_risk_budget', entry_budget_usd=str(budget))
            # Re-size on the actual later receipt, with fees inside the debit allowance.
            b.c = replace(b.c, max_order=min(b.c.max_order,float(budget/Decimal('1.0125'))*.999999))
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


def opening_state(parent, account_mode, episode_id):
    """Carry learning across independent episodes without rewriting the parent account."""
    from paperlab.core import digest
    if account_mode not in ACCOUNT_MODES:
        raise ValueError('Unknown account mode')
    saved = load_parent(parent)
    if account_mode == 'continuous':
        if saved.get('account_mode')=='fresh_training_episode':
            raise ValueError('Do not relabel training episodes as a continuous account')
        return saved
    parent = Path(parent)
    completed = json.loads((parent/'completed.json').read_text())
    reset = dict(account_mode=account_mode, episode_id=episode_id, training_only=True,
        risk_policy=EPISODE_POLICY,max_token_acquisition_usd=float(EPISODE_TOKEN_CAP_USD),
        cash_floor_usd=0.,liquidity_fraction=.01,
        initial_cash_usd=1000., parent_archive=str(parent),
        parent_state_sha256=digest(parent/'online-state.json'),
        parent_completed_sha256=digest(parent/'completed.json'),
        parent_portfolio=saved['portfolio'], parent_end_equity_usd=completed.get('equity_stress_usd'),
        native_checkpoint_sha256=digest(saved['native_checkpoint']),
        head_checkpoint_sha256=digest(saved['head_checkpoint']),
        semantics='Independent training episode; no account deposit or continuous portfolio return')
    return {**saved, 'portfolio':dict(cash='1000',fees='0',halted=False,positions={},quarantined=[],risk_policy=EPISODE_POLICY),
            'active':{},'watched':{},'retired':[], 'account_mode':account_mode,'episode':reset}


def choose(candidates, current, burst, visits):
    """Three observations per burst, then least-recently served; no return ranking."""
    if current in candidates and burst < 3: return current, burst + 1
    remaining = [m for m in candidates if m != current] or list(candidates)
    if not remaining: return None, 0
    return min(remaining, key=lambda m: (visits.get(m, 0), m)), 1



def update_active(active, watched, retired, inactive, portfolio, snapshots, quotes, now, connected=True, entry_quotes=None):
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
        if not qty and entry_quotes is not None and m not in entry_quotes:continue
        if qty and float(qty)*t.mid < 1: continue
        candidates.append((0 if qty else 1, snapshots[m]['created']['received'], m))
    for _, _, m in sorted(candidates)[:max(0, MAX_ACTIVE-len(active))]:
        active[m] = snapshots[m]['created']; watched[m] = active[m]
        portfolio.position(m); admitted.append(m)
    return admitted, removed

def run(root, data, parent, *, seconds=900, commit=lambda: None, fly_factory=None,
        feed_factory=Feed, clock=time.time, sleep=time.sleep, fx_fetch=None, account_mode='continuous',
        all_observed=False,quote_fx_fetch=None,interval_seconds=2.5,head_factory=None):
    if interval_seconds not in (2.5,5):raise ValueError('Only benchmarked cadence choices')
    if not 60 <= seconds <= 900: raise ValueError('Bounded window required')
    if fly_factory is None:
        import modal
        if modal.is_local(): raise RuntimeError('Native fly must run in Modal')
        from paperlab.fly import Fly
        fly_factory = Fly
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root/'started.json').exists(): raise ValueError('Immutable window')
    saved = opening_state(parent, account_mode, root.name); portfolio = Portfolio(saved['portfolio'])
    risk_policy=EPISODE_POLICY if portfolio.episodic else 'quarantined_inventory_cash_floor_v1'
    active = dict(saved['active']); watched = dict(saved.get('watched', active)); retired = set(saved['retired'])
    head = (head_factory or Readout)(); head.restore(saved['head_checkpoint']); opening_updates = head.updates; opening_optimizer_steps=getattr(head,'optimizer_steps',head.updates)
    quote_protocol=QUOTE_PROTOCOL
    if all_observed:
        if account_mode!='fresh_training_episode':raise ValueError('Broad universe requires versioned fresh episodes')
        from paperlab.solana_universe import AllObservedFeed,prepare_snapshots,update_all_active,USDC
        from paperlab.solana_universe import QUOTE_PROTOCOL as quote_protocol
        if feed_factory is Feed:feed_factory=AllObservedFeed
    feed = feed_factory(root)
    for created in watched.values(): feed.accept(created)
    feed.pinned_mints = set(watched); feed.start()
    started = clock(); deadline = started + seconds
    atomic_json(root/'opening.json', saved)
    atomic_json(root/'started.json', dict(at=started, deadline=deadline, paper_only=True,
        reward_protocol=REWARD_PROTOCOL,quote_protocol=quote_protocol if portfolio.episodic else 'legacy_tick_for',
        all_observed=all_observed,max_active_tokens=None if all_observed else MAX_ACTIVE, shared_capital=1000, interval_seconds=interval_seconds, parent=str(parent),
        account_mode=account_mode, episode=saved.get('episode'),
        coverage=('All observed Pump/PumpSwap markets; no token admission cap, public RPC gaps and unresolved/unpriced markets remain visible'
                  if all_observed else 'Pump launches received during this window; bounded admission, not every token'),
        risk_policy=risk_policy, max_tradable_acquisition_cost_usd=None if portfolio.episodic else 100,
        max_new_position_debit_usd=float(EPISODE_TOKEN_CAP_USD) if portfolio.episodic else 2.5,
        minimum_cash_after_buy_usd=0 if portfolio.episodic else 902, news_enabled=False))
    commit()
    fly = None; current = None; burst = 0; visits = {}; contexts = {}; inactive = {}
    rows = trained = fills = overruns = nonzero_rewards = rejected_credit = rejected_orders = 0
    status = 'collecting'; next_step = started; next_fx = 0.
    fx = fx_seen = 0.; quote_usd={}; reasons = Counter(); snapshots = {}; quotes = {}; quote_states = {}
    reward_total = 0.; native_reward_total = 0.; terminal_settlements = []
    def fetch_fx():
        r = requests.get('https://api.coinbase.com/v2/prices/SOL-USD/spot', timeout=4)
        r.raise_for_status(); return float(r.json()['data']['amount'])
    fx_fetch = fx_fetch or fetch_fx
    def fetch_quote_fx():
        r=requests.get('https://api.coinbase.com/v2/prices/USDC-USD/spot',timeout=4)
        r.raise_for_status();return float(r.json()['data']['amount'])
    quote_fx_fetch=quote_fx_fetch or fetch_quote_fx
    def context(m):
        return contexts.setdefault(m, dict(ticks=[], signature=None, previous=None, pending=None,
                                          anchor=None, last_ts=None, credit_anchor=0. if portfolio.episodic else None,
                                          reward_usd=0., quote_gap=False))
    def credit(m, c, contribution, x, at, *, terminal=False, quote_available=True):
        """Each marked dollar enters exactly one mint-specific Q transition.

        Missing intermediate quotes do not turn holdings into cash and back. Keep
        the last observable anchor; only the final stress mark settles an absent
        endpoint, explicitly labelled as a conservative valuation assumption.
        """
        nonlocal reward_total, nonzero_rewards, rejected_credit
        previous=c['previous']
        if previous is None:return None
        delta=contribution-c['credit_anchor']
        skip=(previous.get('execution_rejected') and not portfolio.position(m)['qty'] and abs(delta)<1e-12)
        if skip:
            metric=None;rejected_credit+=1
        else:
            metric=head.update(previous,x,delta,max(0.,at-previous['ts']),
                allowed_actions=portfolio.allowed_actions(m),terminal=terminal)
            metric.update(reward_protocol=REWARD_PROTOCOL,mint=m,
                credit_start_contribution_usd=c['credit_anchor'],credit_end_contribution_usd=contribution,
                quote_gap=c['quote_gap'],quote_available=quote_available,
                valuation='terminal_missing_quote_stress' if terminal and not quote_available else 'observed_indicative_mark')
            reward_total+=delta;c['reward_usd']+=delta;nonzero_rewards+=abs(delta)>1e-12
        c['credit_anchor']=contribution;c['previous']=None;c['quote_gap']=False
        return metric
    ledger = open(root/'decisions.jsonl', 'a'); fxlog = open(root/'fx.jsonl', 'a')
    try:
        while clock() < deadline:
            now = clock()
            if now < next_step: sleep(min(1, next_step-now)); continue
            step_start = now; next_step = now + interval_seconds
            if now >= next_fx:
                next_fx = now + 60
                try:
                    fx = fx_fetch(); fx_seen = clock()
                    if not 0 < fx < 1e7: raise ValueError('Invalid FX')
                    if all_observed:
                        try:
                            usdc=quote_fx_fetch()
                            if not 0<usdc<1e7:raise ValueError('Invalid quote FX')
                        except Exception:usdc=0.
                        quote_usd={USDC:usdc};fx_seen=clock()
                    fxlog.write(json.dumps(dict(at=fx_seen, sol_usd=fx,quote_usd=quote_usd))+'\n'); fxlog.flush()
                except Exception as exc:
                    fx = 0.; print(json.dumps(dict(event='fx_error', error=type(exc).__name__)), flush=True)
            if hasattr(feed,'snapshot_at'):snapshots,now,health=feed.snapshot_at(clock)
            else:snapshots=feed.snapshot();now=clock();health=feed.health()
            observation_at=now
            if health['status'] in ('failed', 'capacity_stopped'): status = health['status']; break
            if all_observed:snapshots=prepare_snapshots(snapshots,now,fx,fx_seen,quote_usd)
            quotes = {}; rejected = {}; admissions = []; removals = [];entry_quotes={};quote_states={}
            connected = health['status'] == 'connected' and 0 <= now-health['last_message'] <= 10
            for m, token in snapshots.items():
                if portfolio.episodic:
                    q=quote_for(token,now,fx,fx_seen,selected=m in watched,connected=connected,unrestricted=all_observed)
                    quote_states[m]=q;t=q.observed_tick;reason=q.observation_reason
                    if q.entry_tick:entry_quotes[m]=q.entry_tick
                else:
                    t, reason = tick_for(token, now, fx, fx_seen, selected=m in watched)
                    if not connected: reason = 'disconnected_feed'
                if reason is None and t: quotes[m] = t
                else: rejected[m] = reason or 'missing_quote'
            admissions, removals = (update_all_active if all_observed else update_active)(active, watched, retired, inactive,
                portfolio,snapshots,quotes,now,connected=connected,
                entry_quotes=entry_quotes if portfolio.episodic else None)
            for m in removals:
                # Park the execution intent, but retain credit for inventory already bought.
                c=context(m);c['pending']=None;c['quote_gap']=True
            with feed.lock: feed.pinned_mints = set(watched)
            if not all_observed and len(watched) > 128: status = 'capacity_stopped'; break
            if not all_observed and len(retired) + len(active) > 10000: status = 'capacity_stopped'; break
            fresh = []; executions = []
            for m in active:
                c = context(m); t = quotes.get(m)
                if t is None:
                    reasons[rejected.get(m, 'missing_token')] += 1
                    c['quote_gap'] = True; c['last_ts'] = None
                    if c['pending'] and now-c['pending']['issued'] > 15: c['pending'] = None
                    continue
                last = snapshots[m]['trades'][-1]; sig = (last['signature'], last['log_index'])
                if sig == c['signature']: continue
                c['signature'] = sig
                if c['pending']:
                    liquidity=last['real_sol_reserves']/1e9*fx*.01
                    kwargs={'liquidity_notional_usd':liquidity} if portfolio.episodic else {}
                    q=quote_states.get(m)
                    execution_tick,execution_reason=(q.for_target(c['pending']['target'],portfolio.cash,
                        portfolio.position(m)['qty']) if q else (t,None))
                    if execution_tick is None:
                        fill=dict(status='rejected',reason=execution_reason)
                    else:
                        fill = portfolio.execute(m, c['pending']['target'], c['pending']['issued'], execution_tick, quotes, **kwargs)
                    executions.append(dict(mint=m, tick=t.__dict__, fill=fill,
                        liquidity_notional_usd=liquidity,real_sol_reserves=last['real_sol_reserves'],sol_usd=fx))
                    fills += fill['status'] == 'filled'
                    if fill['status']=='rejected' and fill.get('reason')!='not_after_decision':
                        if c['previous']:c['previous']['execution_rejected']=True
                        rejected_orders+=1
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
                    contribution = portfolio.contribution(m, t)
                    if c['credit_anchor'] is None:c['credit_anchor']=contribution
                    continuous = c['last_ts'] is not None and 0 < t.ts-c['last_ts'] <= 60
                    if switched or not continuous:fly.controller.brain.reset(keep_memory=True)
                    delta = contribution-c['anchor'] if continuous else 0.
                    # Native eligibility belongs to the preceding token. Clear it on switching;
                    # only the head may assign delayed, mint-specific credit across other tokens.
                    from paperlab.news import News
                    native_path = root/'neural'/f'{trained:05d}' if trained % 12 == 0 else None
                    neural = fly.observe(c['ticks'], len(c['ticks'])-1, News(enabled=False),
                                         delta if not switched and continuous else 0., native_path)
                    native_reward_total+=delta if not switched and continuous else 0.
                    x = inputs(c['ticks'], snapshots[m], portfolio.view(m), neural, now)
                    x[10] = max(-5., min(5., contribution/25))
                    allowed = portfolio.allowed_actions(m)
                    # Select from the current policy, then learn the preceding
                    # transition. A stale inference cannot erase unresolved credit.
                    action, prediction = head.choose(x, allowed_actions=allowed); issued = clock()
                    if issued-t.received_at <= 15 and c['pending'] is None:
                        learning=credit(m,c,contribution,x,t.ts)
                        c['pending'] = dict(target=portfolio.target(m,action,t), issued=issued)
                        c['previous'] = dict(x=x, action=action, ts=t.ts)
                        decision = dict(mint=m, action=action, issued=issued, target=c['pending']['target'])
                    c['anchor'] = contribution; c['last_ts'] = t.ts
                    visits[m] = rows+1; current = m; burst = next_burst; trained += 1
                    neural.update(mint=m, tick=t.__dict__, contribution_usd=contribution, switched_token=switched,
                                  native_context_reset=switched or not continuous,head_training=learning,
                                  readout=prediction, input_features=x.tolist(),
                                  diagnostic_path=str(native_path) if native_path else None)
                    status = 'training'
            elapsed = clock()-step_start; overruns += elapsed > interval_seconds
            row = dict(at=clock(), step=rows, paper_only=True, status=status, admissions=admissions,
                observation_at=observation_at,reward_protocol=REWARD_PROTOCOL,
                event_cursor=health.get('event_cursor'),
                fx_state=dict(sol_usd=fx,seen_at=fx_seen,quote_usd=quote_usd),
                all_observed=all_observed,quote_protocol=quote_protocol if portfolio.episodic else 'legacy_tick_for',
                account_mode=account_mode, episode_id=root.name if account_mode=='fresh_training_episode' else None,
                retired=removals, active_tokens=len(active), tracked_launches=len(snapshots),
                eligible_tokens=len(quotes), executions=executions, decision=decision, neural=neural,
                coverage=dict(observed_tokens=len(snapshots),observable_tokens=len(quotes),
                    unpriced_or_unavailable=dict(Counter(rejected.values())),
                    ready_for_native=len(fresh),never_served_active=sum(m not in visits for m in active),
                    all_tokens_guaranteed=False),
                portfolio=portfolio.state(), equity_stress_usd=portfolio.equity(quotes),
                marks={m:t.__dict__ for m,t in quotes.items() if portfolio.positions.get(m, {}).get('qty', 0)},
                quote_diagnostics={m:dict(observation_reason=q.observation_reason,entry_reason=q.entry_reason,
                    exit_reason=q.exit_reason,**q.position_values(portfolio.position(m)['qty']))
                    for m,q in quote_states.items() if (m==current if all_observed else m in active) or portfolio.positions.get(m,{}).get('qty',0)},
                unavailable_positions=[m for m,p in portfolio.positions.items() if p['qty'] and m not in quotes],
                feed=health, neural_observations=trained, readout_updates=head.updates,
                new_readout_updates=head.updates-opening_updates,
                optimizer_steps=getattr(head,'optimizer_steps',head.updates),new_optimizer_steps=getattr(head,'optimizer_steps',head.updates)-opening_optimizer_steps, fills=fills, overruns=overruns,
                risk_policy=risk_policy, entry_budget_usd=str(portfolio.entry_budget()),
                nonzero_reward_updates=nonzero_rewards, rejected_order_credit_dropped=rejected_credit,
                rejected_orders=rejected_orders,zero_reward_rejected_transitions_skipped=rejected_credit,
                raw_q_reward_usd=reward_total,native_reward_usd=native_reward_total,
                step_seconds=elapsed, skip_reasons=dict(reasons))
            ledger.write(json.dumps(row, allow_nan=False)+'\n'); ledger.flush()
            atomic_json(root/'latest.json', row); rows += 1
            print(json.dumps({k:v for k,v in row.items() if k not in ('neural','portfolio','feed')}, allow_nan=False), flush=True)
            if neural: print(json.dumps(dict(event='online_learning', mint=current,
                native=neural['learning_diagnostics'], head=learning)), flush=True)
            if rows % 12 == 0: commit()
        if clock() >= deadline: status = 'completed'
        if status=='completed':
            for m,c in contexts.items():
                if c['previous'] is None:continue
                metric=credit(m,c,portfolio.contribution(m,quotes.get(m)),c['previous']['x'],
                    max(c['previous']['ts'],observation_at),terminal=True,quote_available=m in quotes)
                if metric:terminal_settlements.append(metric)
            if rows:
                terminal_row={**row,'at':clock(),'step':rows,'status':status,'terminal':True,
                    'decision':None,'neural':None,'executions':[],'admissions':[],'retired':[],
                    'reward_settlements':terminal_settlements,'readout_updates':head.updates,
                    'new_readout_updates':head.updates-opening_updates,
                    'optimizer_steps':getattr(head,'optimizer_steps',head.updates),'new_optimizer_steps':getattr(head,'optimizer_steps',head.updates)-opening_optimizer_steps,
                    'nonzero_reward_updates':nonzero_rewards,'raw_q_reward_usd':reward_total,
                    'rejected_order_credit_dropped':rejected_credit,
                    'zero_reward_rejected_transitions_skipped':rejected_credit}
                ledger.write(json.dumps(terminal_row,allow_nan=False)+'\n');ledger.flush()
                atomic_json(root/'latest.json',terminal_row);rows+=1
    finally:
        feed.close(); ledger.close(); fxlog.close()
    if all_observed:
        atomic_json(root/'observed-universe.json',dict(complete_global_inventory=False,
            tokens={m:dict(created=t['created'],last_trade=t['trades'][-1] if t['trades'] else None)
                    for m,t in snapshots.items()},coverage=feed.health()))
    if fly:
        fly.save(root/'fly-final.npz'); head.save(root/'head-final.pt')
    atomic_json(root/'online-state.json', dict(portfolio=portfolio.state(), active=active,
        account_mode=account_mode, episode=saved.get('episode'),
        watched=watched, retired=sorted(retired), native_checkpoint=str(root/'fly-final.npz') if fly else saved['native_checkpoint'],
        head_checkpoint=str(root/'head-final.pt') if fly else saved['head_checkpoint'],
        native_observations_total=saved.get('native_observations_total', 0)+trained,
        readout_updates=head.updates))
    result = dict(status=status, started=started, ended=clock(), steps=rows, paper_only=True,
        all_observed=all_observed,reward_protocol=REWARD_PROTOCOL,quote_protocol=quote_protocol if portfolio.episodic else 'legacy_tick_for',
        equity_semantics='Indicative full-inventory liquidation mark, not a guaranteed executable sale; unavailable marks are zero.',
        raw_q_reward_usd=reward_total,native_reward_usd=native_reward_total,
        terminal_reward_settlements=terminal_settlements,
        reward_reconciliation=dict(expected_episode_pnl_usd=portfolio.equity(quotes)-1000 if portfolio.episodic else None,
            raw_q_reward_usd=reward_total,
            residual_usd=portfolio.equity(quotes)-1000-reward_total if portfolio.episodic else None,
            per_mint={m:dict(reward_usd=c['reward_usd'],end_contribution_usd=portfolio.contribution(m,quotes.get(m)))
                      for m,c in contexts.items()},
            native_uncredited_difference_usd=reward_total-native_reward_total,
            native_limitation='Native feedback is available only within uninterrupted same-mint bursts; Q receives delayed and terminal credit.'),
        account_mode=account_mode, episode=saved.get('episode'),
        episode_pnl_usd=portfolio.equity(quotes)-1000 if account_mode=='fresh_training_episode' else None,
        neural_observations=trained, readout_updates=head.updates, new_readout_updates=head.updates-opening_updates,
                optimizer_steps=getattr(head,'optimizer_steps',head.updates),new_optimizer_steps=getattr(head,'optimizer_steps',head.updates)-opening_optimizer_steps,
        fills=fills, active_tokens=len(active), portfolio=portfolio.state(),
        risk_policy=risk_policy, entry_budget_usd=str(portfolio.entry_budget()),
        nonzero_reward_updates=nonzero_rewards, rejected_order_credit_dropped=rejected_credit,
        rejected_orders=rejected_orders,zero_reward_rejected_transitions_skipped=rejected_credit,
        tradable_positions=[m for m,p in portfolio.positions.items() if p['qty'] and
            ((m in quote_states and quote_states[m].exit_tick is not None and
              quote_states[m].position_values(p['qty'])['next_fill_proceeds_usd']>0)
             if portfolio.episodic else portfolio.value(m,quotes.get(m))>=1)],
        inventory_indicative_value_usd=portfolio.equity(quotes)-float(portfolio.cash),
        inventory_next_fill_proceeds_usd=sum(quote_states[m].position_values(p['qty'])['next_fill_proceeds_usd']
            for m,p in portfolio.positions.items() if m in quote_states) if portfolio.episodic else None,
        final_mark_observation_at=observation_at,
        unavailable_positions=[m for m,p in portfolio.positions.items() if p['qty'] and m not in quotes],
        equity_stress_usd=portfolio.equity(quotes), skip_reasons=dict(reasons), feed=feed.health(),
        profitable_learning_proven=False)
    atomic_json(root/'result.json', result); commit(); return result
