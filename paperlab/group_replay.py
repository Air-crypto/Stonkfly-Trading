"""Bounded GSPO-inspired, paper-only trajectory replay. No network or order API.

The frozen fly is an encoder; only a stochastic actor head receives gradients.
Twelve branches share market observations, never portfolio state or rewards.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch

from .core import Broker, Costs, Tick, atomic_json, digest
from .solana_events import Feed
from .solana_paper import inputs, tick_for

GROUP_SIZE = 12
INTERVAL = 5
HORIZON = 60
PASSES = 8
FEATURES = 23
ACTION_NAMES = ('hold', 'exit', 'target_0.15_percent', 'target_0.25_percent')
TARGETS = (None, 0., .0015, .0025)
COSTS = Costs(capital=1000, fee_bps=125, slippage_bps=100,
              max_exposure=.0025, max_order=2.5, max_spread_bps=110,
              max_delay=15, loss_stop=.10)
SOURCES = {'train': 'solana-online-20260913-230259',
           'test': 'solana-online-20260913-232717'}
ZERO_EVENT = dict(left_hz=0, right_hz=0, gate_spikes=0, total_spikes=0,
                  KC_spikes=0, reward_spikes=0, aversive_spikes=0, difference_hz=0,
                  learning_diagnostics=dict(weight_delta_l2=0, kc_trace_mean_hz=0))


def extract(source, scratch, excluded=(), limit=4, horizon=HORIZON):
    """First eligible launches by receipt time, chosen without looking at outcomes.

    Only completed archives are accepted. Every selected episode retains the full
    fixed horizon, including missing/stale quotes and disconnected feed periods.
    No SQLite writes or websocket reader are opened on the source archive.
    """
    source = Path(source)
    if not (source/'completed.json').exists():
        raise ValueError('Use completed, immutable source windows')
    db = sqlite3.connect(f'file:{source / "events.db"}?mode=ro', uri=True)
    try:
        bounds = db.execute('SELECT min(received),max(received) FROM events').fetchone()
        if bounds[0] is None or bounds[1]-bounds[0] < horizon:
            raise ValueError('Archive too short')
        all_mints = {r[0] for r in db.execute('SELECT DISTINCT mint FROM events')}
        receipts = iter(db.execute('SELECT received,body FROM events ORDER BY received,rowid'))
        current = next(receipts, None)
        health = iter(db.execute('SELECT at,kind FROM health ORDER BY at,rowid'))
        health_next = next(health, None)
        connected = False
        fx_rows = sorted((json.loads(s) for s in (source/'fx.jsonl').read_text().splitlines()),
                         key=lambda r: r.get('at', r.get('received', 0)))
        fx_index, fx, fx_seen = 0, 0., 0.
        feed = Feed(scratch)
        episodes = {}
        histories = {}
        excluded = set(excluded)
        for now in range(math.ceil(bounds[0]/INTERVAL)*INTERVAL, int(bounds[1])+1, INTERVAL):
            while current is not None and current[0] <= now:
                feed.accept(json.loads(current[1]))
                current = next(receipts, None)
            while health_next and health_next[0] <= now:
                connected = health_next[1] == 'connected'
                health_next = next(health, None)
            while fx_index < len(fx_rows):
                row = fx_rows[fx_index]
                seen = row.get('at', row.get('received', 0))
                if seen > now:
                    break
                fx, fx_seen = row['sol_usd'], seen
                fx_index += 1
            for mint, token in sorted(feed.tokens.items()):
                t, reason = tick_for(token, now, fx, fx_seen, selected=mint in episodes)
                if not connected:
                    reason = 'feed_disconnected'
                if t is not None and reason:
                    t = replace(t, available=False)
                if mint not in episodes:
                    if (len(episodes) >= limit or mint in excluded or reason or t is None
                            or now+horizon > bounds[1]):
                        continue
                    episodes[mint] = dict(mint=mint, start=now, end=now+horizon,
                                          source=source.name, frames=[])
                    histories[mint] = []
                    feed.pinned_mints.add(mint)
                episode = episodes[mint]
                if now > episode['end']:
                    continue
                history = histories[mint]
                # A placeholder carries a past price, but remains unexecutable and
                # receives zero inventory value. No future quote fills a data gap.
                if t is None:
                    t = replace(history[-1], available=False)
                history.append(t)
                base = inputs(history, token, Broker(COSTS), ZERO_EVENT, now).tolist()
                episode['frames'].append(dict(now=now, tick=asdict(t), reason=reason,
                                              base=base))
        result = list(episodes.values())
        if not result or any(len(e['frames']) != horizon//INTERVAL+1 for e in result):
            raise ValueError('Insufficient complete fixed-horizon episodes')
        return result, all_mints, bounds
    finally:
        db.close()


def cache_fly(episodes, data, output, commit=lambda: None):
    """Native propagation is explicitly restricted to Modal."""
    import modal
    if modal.is_local():
        raise RuntimeError('Full fly encoding must run in Modal, never locally')
    from .fly import Fly
    from .news import News
    fly = Fly(data, learning=False)  # pristine, no future-trained checkpoint
    brain = fly.controller.brain
    before = hashlib.sha256(brain.weight.tobytes()).hexdigest()
    news = News(enabled=False)
    observations, neural_seconds = 0, 0.
    try:
        for episode in episodes:
            brain.reset(keep_memory=False)
            ticks = []
            last_neural = [0.] * 10
            for frame in episode['frames']:
                t = Tick(**frame['tick'])
                ticks.append(t)
                if t.available:
                    event = fly.observe(ticks, len(ticks)-1, news, delta=0.)
                    if event['learning_diagnostics']['changed_this_step']:
                        raise RuntimeError('Frozen fly weights changed')
                    last_neural = [event['left_hz']/100, event['right_hz']/100,
                        event['gate_spikes']/10, math.log1p(event['total_spikes'])/15,
                        math.log1p(event['KC_spikes'])/15, math.log1p(event['reward_spikes'])/15,
                        math.log1p(event['aversive_spikes'])/15, event['difference_hz']/100,
                        0., event['learning_diagnostics']['kc_trace_mean_hz']/100]
                    frame['neural'] = event
                    observations += 1
                    neural_seconds += event['compute_seconds']
                frame['base'][12:22] = np.clip(last_neural, -5, 5).tolist()
            if hashlib.sha256(brain.weight.tobytes()).hexdigest() != before:
                raise RuntimeError('Native encoder weight hash changed')
            atomic_json(Path(output)/'cache-progress.json', dict(mint=episode['mint'],
                        native_observations=observations, neural_seconds=neural_seconds))
            commit()
            print(json.dumps(dict(event='group_replay_encoder', mint=episode['mint'],
                                  native_observations=observations)), flush=True)
    finally:
        news.db.close()
    return dict(native_observations=observations, neural_seconds=neural_seconds,
                native_weights_before=before, native_weights_after=before,
                native_weight_delta_l2=0., native_gradients=None,
                encoder='pristine frozen full fly; zero reward; reset per mint; no news')


class Actor(torch.nn.Module):
    def __init__(self, seed=71):
        super().__init__()
        # Constructor must not alter the caller's rollout random stream.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.net = torch.nn.Sequential(torch.nn.Linear(FEATURES, 32), torch.nn.Tanh(),
                                            torch.nn.Linear(32, len(TARGETS)))

    def distribution(self, x, mask):
        logits = self.net(torch.as_tensor(x, dtype=torch.float32))
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError('Empty action mask')
        return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def branch_inputs(frame, broker, remaining):
    t = Tick(**frame['tick'])
    x = list(frame['base'])
    x[8] = float(broker.qty)*t.mid/2.5
    x[9] = float(broker.cash)/1000
    x[10] = (broker.equity(t)-1000)/2.5
    return np.clip(x+[remaining], -5, 5).astype(np.float32).tolist()


def action_mask(broker, tick, pending):
    mask = [True, False, False, False]
    if pending or not tick.available:
        return mask
    equity = broker.equity(tick)
    if float(broker.qty)*tick.mid >= 1:
        mask[1] = True
    for action in (2, 3):
        delta = equity*TARGETS[action]-float(broker.qty)*tick.mid
        # Exclude sub-dollar/no-op targets and insufficient cash at decision time.
        debit = delta*(tick.ask/tick.mid)*(1+COSTS.slippage_bps/10000)*(1+COSTS.fee_bps/10000)
        mask[action] = (abs(delta) >= 1.05 and
                        (delta < 0 or (not broker.halted and equity > 900 and float(broker.cash) >= debit)))
    return mask


def rollout(actor, episode, seed, baseline=None):
    broker = Broker(COSTS)
    rng = np.random.default_rng(seed)
    pending = None
    rows, exposures = [], []
    peak, drawdown = 1000., 0.
    for index, frame in enumerate(episode['frames']):
        now, tick = frame['now'], Tick(**frame['tick'])
        fill = {'status': 'hold'}
        if pending:
            target, issued = pending
            if now-issued > COSTS.max_delay:
                fill = {'status': 'expired', 'decision_ts': issued}
                pending = None
            elif tick.available and tick.received_at > issued:
                fill = broker.execute(target, issued, tick)
                pending = None
        equity = broker.equity(tick)
        peak = max(peak, equity)
        drawdown = max(drawdown, peak-equity)
        exposures.append(float(broker.qty)*tick.mid)
        final = index == len(episode['frames'])-1
        row = dict(now=now, fill=fill, equity=equity, state=broker.state(),
                   quote_available=tick.available, terminal=final)
        if not final:
            x = branch_inputs(frame, broker, 1-index/(len(episode['frames'])-1))
            mask = action_mask(broker, tick, pending)
            with torch.no_grad():
                distribution = actor.distribution(x, mask)
                probs = distribution.probs.numpy().astype(float)
            if baseline == 'cash':
                action = 0
            elif baseline == 'buy_hold':
                action = 3 if index == 0 and mask[3] else 0
            else:
                action = int(rng.choice(len(TARGETS), p=probs/probs.sum()))
            row.update(x=x, mask=mask, action=action, action_name=ACTION_NAMES[action],
                       old_log_prob=float(math.log(probs[action])), old_probs=probs.tolist())
            if TARGETS[action] is not None:
                pending = (TARGETS[action], now)
        rows.append(row)
    pnl = rows[-1]['equity']-1000
    exposure_penalty = .001*float(np.mean(exposures))
    reward = pnl-.25*drawdown-exposure_penalty
    return dict(mint=episode['mint'], seed=seed, baseline=baseline, rows=rows, pnl_usd=pnl,
                reward=reward, drawdown_usd=drawdown, exposure_penalty_usd=exposure_penalty,
                fees_usd=float(broker.fees), fills=sum(r['fill']['status']=='filled' for r in rows),
                terminal_pending=pending, terminal_inventory=broker.state(),
                terminal_quote_available=episode['frames'][-1]['tick']['available'])


def advantages(rewards):
    values = np.asarray(rewards, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite rewards')
    deviation = float(values.std())
    return None if deviation < 1e-8 else (values-values.mean())/deviation


def clipped_objective(log_ratios, advantage, epsilon=.2):
    ratio = log_ratios.exp()
    return -torch.minimum(ratio*advantage, ratio.clamp(1-epsilon, 1+epsilon)*advantage).mean()


def update(actor, optimizer, group, epochs=2):
    if len(group) != GROUP_SIZE:
        raise ValueError('Expected twelve complete trajectories')
    adv = advantages([r['reward'] for r in group])
    if adv is None:
        return [dict(status='skipped_zero_reward_variance', reward_std=0.)]
    advantages_tensor = torch.tensor(adv, dtype=torch.float32)
    metrics = []
    for epoch in range(epochs):
        ratios, entropies, kls = [], [], []
        for trajectory in group:
            rows = trajectory['rows'][:-1]
            d = actor.distribution([r['x'] for r in rows], [r['mask'] for r in rows])
            new = d.log_prob(torch.tensor([r['action'] for r in rows]))
            old = torch.tensor([r['old_log_prob'] for r in rows])
            # All timesteps, including masked HOLD steps, belong to the sequence.
            ratios.append((new-old).mean())
            entropies.append(d.entropy().mean())
            old_probs = torch.tensor([r['old_probs'] for r in rows])
            kls.append((old_probs*(old_probs.clamp_min(1e-30).log()-d.logits)).sum(-1).mean())
        log_ratios = torch.stack(ratios)
        entropy = torch.stack(entropies).mean()
        kl = torch.stack(kls).mean()
        if float(kl.detach()) > .03:
            metrics.append(dict(status='skipped_kl_limit', epoch=epoch, kl=float(kl.detach())))
            break
        policy_loss = clipped_objective(log_ratios, advantages_tensor)
        loss = policy_loss-.001*entropy+.01*kl
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite policy loss')
        before = torch.cat([p.detach().flatten() for p in actor.parameters()]).clone()
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        gradient_by_layer = {name: float(p.grad.norm()) for name, p in actor.named_parameters()}
        optimizer.step()
        change = torch.cat([p.detach().flatten() for p in actor.parameters()])-before
        ratios_np = log_ratios.detach().exp().numpy()
        metrics.append(dict(status='updated', epoch=epoch, loss=float(loss.detach()),
            policy_loss=float(policy_loss.detach()), entropy=float(entropy.detach()), kl=float(kl.detach()),
            gradient_norm=float(gradient_norm), gradients_after_clip=gradient_by_layer,
            weight_delta_l2=float(change.norm()), weight_delta_max=float(change.abs().max()),
            sequence_ratios=ratios_np.tolist(), clipped_fraction=float(np.mean(abs(ratios_np-1)>.2)),
            reward_mean=float(np.mean([r['reward'] for r in group])),
            reward_std=float(np.std([r['reward'] for r in group])), advantages=adv.tolist()))
    return metrics


def experiment(state, output, commit=lambda: None):
    """One fixed pilot, no hyperparameter search, promotion, or recurring schedule."""
    state, output = Path(state), Path(output)
    started = time.monotonic()
    torch.set_num_threads(2)
    train, seen, train_bounds = extract(state/'solana-live'/SOURCES['train'], output/'feed-train')
    test, _, test_bounds = extract(state/'solana-live'/SOURCES['test'], output/'feed-test', excluded=seen)
    if train_bounds[1] >= test_bounds[0] or set(e['mint'] for e in train) & set(e['mint'] for e in test):
        raise ValueError('Temporal or mint split leakage')
    manifest = dict(algorithm='GSPO-inspired sequence actor', group_size=GROUP_SIZE,
        horizon_seconds=HORIZON, interval_seconds=INTERVAL, passes=PASSES,
        gradient_epochs_per_group=2, actor_seed=71, optimizer='Adam lr=0.001',
        costs=asdict(COSTS), actions=dict(zip(ACTION_NAMES, TARGETS)),
        reward='net liquidation PnL USD - 0.25*max drawdown USD - 0.001*mean marked exposure USD',
        selection='first four eligible mints per source by receipt grid then lexical mint; fixed horizon',
        exclusions='test excludes every mint appearing in training archive',
        sources={split: dict(run_id=name, hashes={n: digest(state/'solana-live'/name/n)
                    for n in ('events.db', 'fx.jsonl', 'completed.json')}) for split, name in SOURCES.items()},
        cohorts={split: [dict(mint=e['mint'], start=e['start'], end=e['end']) for e in episodes]
                 for split, episodes in (('train', train), ('test', test))},
        limitations=['indicative reserves, not executable exchange quotes',
            'correlated branches do not add independent market data',
            'short same-day pilot; no evidence of generalization or profitability',
            'no native synapse updates, news, live promotion or current Q-head comparison',
            'each token is an isolated $1000 account, not a combined live portfolio'])
    atomic_json(output/'manifest.json', manifest)
    commit()  # Cohorts and training choices are durable before outcomes/updates.
    encoder = cache_fly(train+test, state/'fly-data', output, commit)
    atomic_json(output/'features.json', dict(train=train, test=test))
    atomic_json(output/'encoder.json', encoder)
    commit()
    actor = Actor()
    frozen = Actor()
    torch.save(actor.state_dict(), output/'actor-initial.pt')
    optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
    updates, group_index = [], 0
    with (output/'training-trajectories.jsonl').open('w') as ledger:
        for repeat in range(PASSES):
            for episode in train:
                group = [rollout(actor, episode, seed=10000+group_index*GROUP_SIZE+i) for i in range(GROUP_SIZE)]
                for branch, trajectory in enumerate(group):
                    ledger.write(json.dumps(dict(group=group_index, branch=branch, repeat=repeat, **trajectory))+'\n')
                ledger.flush()
                metrics = update(actor, optimizer, group)
                updates.extend(dict(group=group_index, repeat=repeat, mint=episode['mint'], **r) for r in metrics)
                group_index += 1
            atomic_json(output/'updates.json', updates)
            print(json.dumps(dict(event='group_replay_training', completed_passes=repeat+1,
                                  groups=group_index, last_update=updates[-1])), flush=True)
    torch.save(dict(model=actor.state_dict(), optimizer=optimizer.state_dict()), output/'actor-trained.pt')
    evaluation = []
    for split, episodes in (('train', train), ('test', test)):
        for episode in episodes:
            for name, model, baseline, count in (('trained', actor, None, GROUP_SIZE),
                    ('initial', frozen, None, GROUP_SIZE), ('cash', frozen, 'cash', 1),
                    ('buy_hold', frozen, 'buy_hold', 1)):
                for i in range(count):
                    evaluation.append(dict(split=split, policy=name,
                                           **rollout(model, episode, 90000+i, baseline)))
    with (output/'evaluation-trajectories.jsonl').open('w') as ledger:
        for row in evaluation:
            ledger.write(json.dumps(row)+'\n')
    summary = []
    for split in ('train', 'test'):
        for policy in ('trained', 'initial', 'cash', 'buy_hold'):
            rows = [r for r in evaluation if r['split']==split and r['policy']==policy]
            summary.append(dict(split=split, policy=policy, trajectories=len(rows),
                independent_mints=len({r['mint'] for r in rows}),
                mean_pnl_usd=float(np.mean([r['pnl_usd'] for r in rows])),
                mean_reward=float(np.mean([r['reward'] for r in rows])),
                mean_fees_usd=float(np.mean([r['fees_usd'] for r in rows])),
                mean_fills=float(np.mean([r['fills'] for r in rows])),
                worst_pnl_usd=min(r['pnl_usd'] for r in rows)))
    paired = []
    for episode in test:
        by_policy = {p: np.mean([r['pnl_usd'] for r in evaluation if r['split']=='test'
                              and r['mint']==episode['mint'] and r['policy']==p]) for p in ('trained','initial','cash')}
        paired.append(dict(mint=episode['mint'], trained_minus_initial_usd=float(by_policy['trained']-by_policy['initial']),
                           trained_minus_cash_usd=float(by_policy['trained'])))
    actual = [r for r in updates if r['status']=='updated']
    result = dict(status='completed', paper_only=True, manifest_sha256=digest(output/'manifest.json'),
        groups=group_index, training_trajectories=group_index*GROUP_SIZE, gradient_updates=len(actual),
        skipped_updates=len(updates)-len(actual), actor_parameters=sum(p.numel() for p in actor.parameters()),
        actor_weight_delta_l2=float(torch.cat([(p-q).detach().flatten() for p,q in zip(actor.parameters(),frozen.parameters())]).norm()),
        loss_first=actual[0]['loss'] if actual else None, loss_last=actual[-1]['loss'] if actual else None,
        gradient_norm_mean=float(np.mean([r['gradient_norm'] for r in actual])) if actual else 0.,
        encoder=encoder, evaluation=summary, paired_test_mints=paired, wall_seconds=time.monotonic()-started,
        promotion='blocked: mechanics pilot only; more independent dates/mints and execution validation required',
        uncertainty='No confidence interval: at most four test mints from one day; branches are correlated.')
    atomic_json(output/'result.json', result)
    commit()
    return result
