"""Small numerical actor/critic with on-policy PPO. This is not a language model."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .core import Costs, Environment, atomic_json, digest, features
from .telemetry import emit


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(64, 464), nn.Tanh(), nn.Linear(464, 464), nn.Tanh())
        self.actor = nn.Linear(464, 3)
        self.critic = nn.Linear(464, 1)

    def forward(self, obs):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def action(self, obs):
        return self.inspect(obs)["action"]

    @torch.no_grad()
    def inspect(self, obs):
        logits, value = self(torch.tensor(obs).unsqueeze(0))
        probabilities = logits.softmax(-1)[0]
        return {"action": int(logits.argmax(-1).item()), "action_probabilities": probabilities.tolist(),
                "action_targets": [0, .25, .5], "value_estimate": float(value.item()),
                "logits": logits[0].tolist(), "features": obs.tolist(),
                "gradient_update": False}


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != "64-market14-news50-v1":
        raise ValueError("Incompatible observation schema")
    p = Policy()
    p.load_state_dict(checkpoint["model"])
    p.eval()
    return p, checkpoint


def evaluate(policy, ticks, news, costs, start, end, baseline=None, stride=1):
    env = Environment(ticks, news, costs, start, end,stride=stride)
    obs, done = env.reset(), False
    rows, fills = [], 0
    while not done:
        decision_ts = ticks[env.i].ts
        observation = obs.copy()
        detail = {}
        if baseline == "cash":
            action = 0
        elif baseline == "equal_cap_buy_hold":
            # Allocate over the same per-order cap, then keep quantity without rebalancing.
            entry_steps = math.ceil(costs.capital * costs.max_exposure / costs.max_order)
            action = 2 if env.i < start + entry_steps*stride else None
        elif baseline == "trend":
            action = 2 if obs[8] > 0 else 0
        else:
            detail = policy.inspect(obs)
            action = detail["action"]
        if action is None:
            env.i = min(env.i+stride,env.end)
            eq = env.broker.equity(ticks[env.i])
            env.peak = max(env.peak, eq)
            info = {"equity": eq, "drawdown": 1 - eq / env.peak, "fill": {"status": "hold"}, "ts": ticks[env.i].ts}
            obs = features(ticks, env.i, env.broker, news)
            done = env.i >= env.end
        else:
            obs, reward, done, info = env.step(action)
            info["reward"] = reward
        info.update(decision_ts=decision_ts, action=action, detail=detail,
                    broker=env.broker.state(), observation=observation.tolist())
        fills += info["fill"]["status"] == "filled"
        rows.append(info)
    eq = np.array([costs.capital] + [r["equity"] for r in rows])
    return {"return_pct": (eq[-1] / costs.capital - 1) * 100, "max_drawdown_pct": max(r["drawdown"] for r in rows) * 100, "fees": float(env.broker.fees), "fills": fills, "steps": len(rows), "end_equity": float(eq[-1]), "ledger": rows}


class MultiEnvironment:
    """Separate chronological episodes; never splice one coin's price into another."""
    def __init__(self, episodes):
        self.episodes=episodes
        self.cursor=-1
        self.current=episodes[0]

    def __getattr__(self,name):
        return getattr(self.current,name)

    def reset(self):
        self.cursor=(self.cursor+1)%len(self.episodes)
        self.current=self.episodes[self.cursor]
        return self.current.reset()

    def step(self,action):
        return self.current.step(action)


def pooled_split(series):
    # Common wall-clock boundaries across all assets prevent cross-asset leakage.
    times=sorted({t.ts for seq in series.values() for t in seq if t.available})
    if len(times)<100 or sum(len(seq) for seq in series.values())<400:
        raise ValueError("Pooled training needs >=400 observations and >=100 distinct timestamps")
    return times[int(len(times)*.7)],times[int(len(times)*.85)]


def train(ticks, news, output, costs=Costs(), steps=8192, seed=7, dataset_sha="unknown", series=None, decision_stride=1):
    if (series is None and len(ticks) < 400) or steps < 128:
        raise ValueError("Need >=400 ticks and >=128 PPO transitions")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    split1, split2 = int(len(ticks) * .7), int(len(ticks) * .85)
    if series is None:
        env = Environment(ticks, news, costs, 63, split1 - 1,stride=decision_stride)
    else:
        cutoff1,cutoff2=pooled_split(series)
        episodes=[]
        for key,seq in sorted(series.items()):
            if any(t.product!=key for t in seq) or any(b.ts<=a.ts for a,b in zip(seq,seq[1:])):
                raise ValueError("Nonchronological or mixed pool episode")
            end=sum(t.ts<cutoff1 for t in seq)-1
            if end>63:
                episodes.append(Environment(seq,news,costs,63,end,stride=decision_stride))
        if not episodes:
            raise ValueError("No pre-cutoff pool has sufficient context")
        env=MultiEnvironment(episodes)
        dataset_sha=hashlib.sha256(json.dumps({k:[asdict(t) for t in v] for k,v in sorted(series.items())},sort_keys=True).encode()).hexdigest()
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rollout_path = output / "rollout.jsonl"
    rollout_path.write_text("")
    optimizer_updates = []
    emit("training_started", source=ticks[0].source, observations=len(ticks), transitions=steps, seed=seed)
    obs = env.reset()
    completed = 0
    started = time.perf_counter()
    updates = []
    while completed < steps:
        n = min(256, steps - completed)
        observations, actions, logs, rewards, dones, values = [], [], [], [], [], []
        rollout = []
        for _ in range(n):
            with torch.no_grad():
                logits, value = policy(torch.from_numpy(obs).unsqueeze(0))
                distribution = Categorical(logits=logits)
                action = distribution.sample()
            observations.append(obs.copy())
            actions.append(action.item())
            logs.append(distribution.log_prob(action).item())
            values.append(value.item())
            decision_ts = env.ticks[env.i].ts
            input_features = obs.tolist()
            obs, reward, done, info = env.step(action.item())
            rollout.append({"transition": completed + len(rollout) + 1, "decision_ts": decision_ts,
                            "product": env.ticks[env.i].product,
                            "action": action.item(), "probabilities": distribution.probs[0].tolist(),
                            "value": value.item(), "features": input_features, "reward": reward,
                            "done": done, "broker": env.broker.state(), **info})
            rewards.append(reward)
            dones.append(done)
            if done:
                obs = env.reset()
        with torch.no_grad():
            last_value = policy(torch.from_numpy(obs).unsqueeze(0))[1].item()
        advantages = np.zeros(n, dtype=np.float32)
        carry = 0
        for t in reversed(range(n)):
            future = last_value if t == n - 1 else values[t + 1]
            continuation = 0 if dones[t] else 1
            delta = rewards[t] + .99 * future * continuation - values[t]
            carry = delta + .99 * .95 * continuation * carry
            advantages[t] = carry
        returns = torch.tensor(advantages + np.array(values, dtype=np.float32))
        raw_advantages = advantages.copy()
        advantages = torch.tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        for row, advantage, normalized, target in zip(rollout, raw_advantages, advantages, returns):
            row.update(advantage=float(advantage), normalized_advantage=float(normalized), return_target=float(target))
        with rollout_path.open("a") as handle:
            handle.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in rollout))
        x, a, old_log = torch.tensor(np.array(observations)), torch.tensor(actions), torch.tensor(logs)
        # Four epochs over only this freshly collected rollout; no off-policy replay buffer.
        for epoch in range(4):
            for idx in torch.randperm(n).split(64):
                logits, value = policy(x[idx])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(a[idx]) - old_log[idx]).exp()
                policy_loss = -torch.minimum(ratio * advantages[idx], ratio.clamp(.8, 1.2) * advantages[idx]).mean()
                value_loss = (value - returns[idx]).square().mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + .5 * value_loss - .01 * entropy
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                optimizer.zero_grad()
                loss.backward()
                before = {name: p.detach().clone() for name, p in policy.named_parameters()}
                raw_gradients = {name: p.grad.detach().clone() for name, p in policy.named_parameters()}
                grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
                optimizer.step()
                layers = {}
                for name, parameter in policy.named_parameters():
                    gradient = raw_gradients[name]
                    change = parameter.detach() - before[name]
                    layers[name] = {"gradient_l2": float(gradient.norm()), "gradient_max_abs": float(gradient.abs().max()),
                                    "gradient_mean": float(gradient.mean()), "clipped_gradient_l2": float(parameter.grad.norm()),
                                    "update_l2": float(change.norm()), "parameter_l2": float(parameter.detach().norm())}
                optimizer_updates.append({"optimizer_step": len(optimizer_updates) + 1, "rollout_end": completed + n,
                    "epoch": epoch + 1, "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                    "approx_kl": float(((ratio - 1) - ratio.log()).mean().detach()),
                    "clip_fraction": float(((ratio - 1).abs() > .2).float().mean()),
                    "gradient_norm_before_clip": float(grad_norm),
                    "gradient_norm_after_clip": math.sqrt(sum(v["clipped_gradient_l2"]**2 for v in layers.values())),
                    "update_norm": math.sqrt(sum(v["update_l2"]**2 for v in layers.values())),
                    "learning_rate": optimizer.param_groups[0]["lr"], "layers": layers})
        completed += n
        update = {"steps": completed, "loss": float(loss.detach()), "mean_reward": float(np.mean(rewards)),
                  "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                  "entropy": float(entropy.detach()), "gradient_norm": float(grad_norm),
                  "optimizer_steps": len(optimizer_updates)}
        updates.append(update)
        emit("ppo_update", **update)
    atomic_json(output / "optimizer-updates.json", optimizer_updates)
    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "last_raw_gradients": raw_gradients,
                "last_parameter_updates": {name: p.detach() - before[name] for name, p in policy.named_parameters()},
                "torch_rng_state": torch.get_rng_state(), "seed": seed, "transitions": completed,
                "note": "Full tensors for FINAL optimizer step; per-layer statistics for EVERY optimizer step are in optimizer-updates.json."},
               output / "training-state.pt")
    available_news = news.db.execute("SELECT * FROM news ORDER BY id").fetchall()
    news_sha = hashlib.sha256(json.dumps(available_news, separators=(",", ":")).encode()).hexdigest()
    metadata = {"schema": "64-market14-news50-v1", "seed": seed, "steps": completed, "costs": asdict(costs), "dataset_sha256": dataset_sha, "source": ticks[0].source, "product": ticks[0].product, "train_until": ticks[split1 - 1].ts, "validation_until": ticks[split2 - 1].ts, "news_enabled": news.enabled, "news_snapshot_sha256": news_sha, "observations_with_news": sum(bool(news.features(t.ts)[48] > 0) for t in ticks), "parameters": sum(p.numel() for p in policy.parameters())}
    if series is not None:
        metadata.update(product="MULTI-DEX",train_until=cutoff1,validation_until=cutoff2,decision_stride=decision_stride,
                        assets=sorted(series),training_assets=len(episodes),
                        split_semantics="Strict global wall-clock cutoffs; independent pool episodes",
                        observations_with_news=sum(bool(news.features(t.ts)[48]>0) for seq in series.values() for t in seq))
    model_path = output / "policy.pt"
    torch.save({**metadata, "model": policy.state_dict()}, model_path.with_suffix(".partial"))
    model_path.with_suffix(".partial").replace(model_path)
    # The checkpoint is fixed before viewing validation/test. No selection on test results.
    policy.eval()
    result = {**metadata, "model_sha256": digest(model_path), "train_seconds": time.perf_counter() - started, "updates": updates, "splits": {}}
    for name, start, end in (("validation", split1, split2 - 1), ("test", split2, len(ticks) - 1)):
        result["splits"][name] = {}
        for baseline in (None, "cash", "equal_cap_buy_hold", "trend"):
            key = baseline or "ppo"
            if series is None:
                evaluation = evaluate(policy, ticks, news, costs, start, end, baseline,stride=decision_stride)
            else:
                audits={}
                low,high=(cutoff1,cutoff2) if name=="validation" else (cutoff2,float("inf"))
                for asset,seq in sorted(series.items()):
                    a=max(63,sum(t.ts<low for t in seq))
                    b=sum(t.ts<high for t in seq)-1
                    if a<b:
                        audits[asset]=evaluate(policy,seq,news,costs,a,b,baseline,stride=decision_stride)
                ledgers={asset:m.pop("ledger") for asset,m in audits.items()}
                # Equal starting capital per separate audit episode. Do not claim these
                # independent audits reproduce the four-sleeve deployed portfolio.
                evaluation={"return_pct":float(np.mean([m["return_pct"] for m in audits.values()])) if audits else 0,
                            "max_drawdown_pct":max((m["max_drawdown_pct"] for m in audits.values()),default=0),
                            "fees":sum(m["fees"] for m in audits.values()),"fills":sum(m["fills"] for m in audits.values()),
                            "assets_evaluated":len(audits),"assets_omitted":len(series)-len(audits),"per_asset":audits,
                            "aggregation":"Mean independent episode return, not live portfolio P&L","ledger":ledgers}
            atomic_json(output / f"{name}-{key}-ledger.json", evaluation.pop("ledger"))
            result["splits"][name][key] = evaluation
    result["hosting_break_even_monthly_pct"] = {str(cost): cost / costs.capital * 100 for cost in (20, 40, 100)}
    result["limitations"] = ["Research run, not a profitability certificate.", "Single chronological split and one seed; repeat with preregistered windows and seeds.", "Paper fills omit queue position and market impact; historical candle fills are proxies.", "News must have been collected at the time. Zero news in old history is intentional.", "Test is a one-shot audit. Repeatedly optimizing against it invalidates the holdout.", "Returns above are after modeled trading costs, before hosting. Hosting break-even scenarios are in metrics.json; no monthly return is extrapolated from this short sample."]
    atomic_json(output / "metrics.json", result)
    emit("training_completed", transitions=completed, optimizer_steps=len(optimizer_updates),
         seconds=result["train_seconds"], test=result["splits"]["test"]["ppo"], model_sha256=result["model_sha256"])
    lines = ["# Paper experiment", "", f"Source: **{ticks[0].source}**. {completed:,} PPO transitions, {metadata['parameters']:,} parameters. Seed {seed}.", "", "| Test policy | Net return | Max drawdown | Fills | Fees |", "|---|---:|---:|---:|---:|"]
    for key, m in result["splits"]["test"].items():
        lines.append(f"| {key} | {m['return_pct']:.3f}% | {m['max_drawdown_pct']:.3f}% | {m['fills']} | ${m['fees']:.2f} |")
    lines += ["", *result["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result
