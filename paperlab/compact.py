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

from .core import Costs, Environment, atomic_json, digest


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
        return int(self(torch.tensor(obs).unsqueeze(0))[0].argmax(-1).item())


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != "64-market14-news50-v1":
        raise ValueError("Incompatible observation schema")
    p = Policy()
    p.load_state_dict(checkpoint["model"])
    p.eval()
    return p, checkpoint


def evaluate(policy, ticks, news, costs, start, end, baseline=None):
    env = Environment(ticks, news, costs, start, end)
    obs, done = env.reset(), False
    rows, fills = [], 0
    while not done:
        if baseline == "cash":
            action = 0
        elif baseline == "equal_cap_buy_hold":
            # Allocate over the same per-order cap, then keep quantity without rebalancing.
            entry_steps = math.ceil(costs.capital * costs.max_exposure / costs.max_order)
            action = 2 if env.i < start + entry_steps else None
        elif baseline == "trend":
            action = 2 if obs[8] > 0 else 0
        else:
            action = policy.action(obs)
        if action is None:
            env.i += 1
            eq = env.broker.equity(ticks[env.i])
            env.peak = max(env.peak, eq)
            info = {"equity": eq, "drawdown": 1 - eq / env.peak, "fill": {"status": "hold"}, "ts": ticks[env.i].ts}
            done = env.i >= env.end
        else:
            obs, _, done, info = env.step(action)
        fills += info["fill"]["status"] == "filled"
        rows.append(info)
    eq = np.array([costs.capital] + [r["equity"] for r in rows])
    return {"return_pct": (eq[-1] / costs.capital - 1) * 100, "max_drawdown_pct": max(r["drawdown"] for r in rows) * 100, "fees": float(env.broker.fees), "fills": fills, "steps": len(rows), "end_equity": float(eq[-1]), "ledger": rows}


def train(ticks, news, output, costs=Costs(), steps=8192, seed=7, dataset_sha="unknown"):
    if len(ticks) < 400 or steps < 128:
        raise ValueError("Need >=400 ticks and >=128 PPO transitions")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    split1, split2 = int(len(ticks) * .7), int(len(ticks) * .85)
    env = Environment(ticks, news, costs, 63, split1 - 1)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    obs = env.reset()
    completed = 0
    started = time.perf_counter()
    updates = []
    while completed < steps:
        n = min(256, steps - completed)
        observations, actions, logs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(n):
            with torch.no_grad():
                logits, value = policy(torch.from_numpy(obs).unsqueeze(0))
                distribution = Categorical(logits=logits)
                action = distribution.sample()
            observations.append(obs.copy())
            actions.append(action.item())
            logs.append(distribution.log_prob(action).item())
            values.append(value.item())
            obs, reward, done, _ = env.step(action.item())
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
        advantages = torch.tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))
        x, a, old_log = torch.tensor(np.array(observations)), torch.tensor(actions), torch.tensor(logs)
        # Four epochs over only this freshly collected rollout; no off-policy replay buffer.
        for _ in range(4):
            for idx in torch.randperm(n).split(64):
                logits, value = policy(x[idx])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(a[idx]) - old_log[idx]).exp()
                loss = -torch.minimum(ratio * advantages[idx], ratio.clamp(.8, 1.2) * advantages[idx]).mean()
                loss = loss + .5 * (value - returns[idx]).square().mean() - .01 * dist.entropy().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), .5)
                optimizer.step()
        completed += n
        updates.append({"steps": completed, "loss": float(loss.detach()), "mean_reward": float(np.mean(rewards))})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    available_news = news.db.execute("SELECT * FROM news ORDER BY id").fetchall()
    news_sha = hashlib.sha256(json.dumps(available_news, separators=(",", ":")).encode()).hexdigest()
    metadata = {"schema": "64-market14-news50-v1", "seed": seed, "steps": completed, "costs": asdict(costs), "dataset_sha256": dataset_sha, "source": ticks[0].source, "product": ticks[0].product, "train_until": ticks[split1 - 1].ts, "validation_until": ticks[split2 - 1].ts, "news_enabled": news.enabled, "news_snapshot_sha256": news_sha, "observations_with_news": sum(bool(news.features(t.ts)[48] > 0) for t in ticks), "parameters": sum(p.numel() for p in policy.parameters())}
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
            evaluation = evaluate(policy, ticks, news, costs, start, end, baseline)
            atomic_json(output / f"{name}-{key}-ledger.json", evaluation.pop("ledger"))
            result["splits"][name][key] = evaluation
    result["hosting_break_even_monthly_pct"] = {str(cost): cost / costs.capital * 100 for cost in (20, 40, 100)}
    result["limitations"] = ["Research run, not a profitability certificate.", "Single chronological split and one seed; repeat with preregistered windows and seeds.", "Paper fills omit queue position and market impact; historical candle fills are proxies.", "News must have been collected at the time. Zero news in old history is intentional.", "Test is a one-shot audit. Repeatedly optimizing against it invalidates the holdout.", "Returns above are after modeled trading costs, before hosting. Hosting break-even scenarios are in metrics.json; no monthly return is extrapolated from this short sample."]
    atomic_json(output / "metrics.json", result)
    lines = ["# Paper experiment", "", f"Source: **{ticks[0].source}**. {completed:,} PPO transitions, {metadata['parameters']:,} parameters. Seed {seed}.", "", "| Test policy | Net return | Max drawdown | Fills | Fees |", "|---|---:|---:|---:|---:|"]
    for key, m in result["splits"]["test"].items():
        lines.append(f"| {key} | {m['return_pct']:.3f}% | {m['max_drawdown_pct']:.3f}% | {m['fills']} | ${m['fees']:.2f} |")
    lines += ["", *result["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result
