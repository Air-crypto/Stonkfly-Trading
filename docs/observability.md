# Five-minute experiment and learning diagnostics

The scheduled worker now collects one public BTC book observation every five minutes. Both independent paper policies can begin decisions with closed-candle price context plus the fresh observations already collected. Historical context provides the missing lookback only; it never adds ledger entries or training samples. If the initial historical fetch fails, the runner falls back to collecting 64 fresh observations. Compact PPO retrains every six hours after the archive reaches 400 observations (about 33 hours 20 minutes). More observations do not create more independent market regimes or establish an edge. RSS is checked once per 15-minute period; FinBERT loads only if a previously unseen headline needs encoding.

The faster experiment lives in `/state/fast-5m`. The original 15-minute SQLite accounts, news archive, and checkpoints remain at `/state` and are not relabeled as five-minute data. The full immutable connectome and Hugging Face download cache are shared. The existing budget reservation and provider controls remain in effect. Cold-start allowance is ten seconds with the same 2x rate cushion; the previously observed cold starts were below this allowance. All work remains CPU-only, with one writer and zero idle containers.

## Console events

Modal stdout now contains JSON events: `cycle_started`, `news_scan`, `market_observation`, `paper_decision`, `cycle_completed`, and `worker_completed`. Failures emit `cycle_failed`; the reserve guard emits `budget_stopped`. Each paper event includes bid/ask, action or target allocation, previous decision's fill, cash, inventory, cumulative fees, equity, and model diagnostics. A decision can HOLD or be rejected by the simulator; no trade is fabricated for display.

PPO emits `training_started`, one `ppo_update` per 256-transition rollout, and `training_completed`. These report the **last minibatch** loss components and gradient norm for each rollout, plus its mean reward. The complete optimizer-step sequence is in the artifact files, not just console samples.

The anonymous Hugging Face warning concerns public model download rate limits. It does not report a training or trading error. Model load progress previously dominated the console because business results were only persisted, not printed. No Hub token is necessary for this public pinned model.

## Durable artifacts

A single bounded `worker(diagnostics=True)` call creates `/state/fast-5m/diagnostics`. Invoke the already deployed function, so all calls share the same single-writer container limit. Do not launch a second ephemeral app against the state volume. A completed diagnostic call returns its saved result rather than running again.

- `compact/rollout.jsonl`: every sampled PPO transition, input features, probabilities, critic value, reward, advantage, return target, simulated fill, and account state. This is historical training data, not forward performance.
- `compact/optimizer-updates.json`: every optimizer minibatch's policy loss, value loss, total loss, entropy, approximate KL, clipped probability-ratio fraction, pre/post-clipping gradient norms, actual parameter update norm, learning rate, and per-layer statistics. At 8,192 transitions, four epochs and 64-sized minibatches there are 512 optimizer updates.
- `compact/training-state.pt`: final model and Adam optimizer state, final raw gradient tensors, final parameter-delta tensors, and Torch RNG state. Full tensor arrays are saved for the **final** update; all earlier updates retain scalar/per-layer summaries. Gradients from the old run cannot be reconstructed from its old loss-only log.
- `compact/replay.json`: 32 deterministic held-out inference steps. Inference has no optimizer gradient update. `test-ppo-ledger.json` covers the longer held-out test window; `metrics.json` includes fixed baselines and limitations.
- `fly/ledger.json`: every replay step's observed quote, previous decision's fill, new decision, equity change, reinforcement stimulus, decoder firing rates, spikes, and weight changes. The final outstanding decision is accounted for in `fly/metrics.json` at the next observation.
- `fly/steps/NNN.png`: the exact market/news image fed to the fly for that observation.
- `fly/steps/NNN.npz`: per-neuron spike totals and end-of-observation voltage, eligibility/modulation state, rate/memory traces, and every plastic edge's before/after weight and delta. This is an observation snapshot, not every native 0.1 ms integrator state.
- `fly/fly.npz`: the final native checkpoint, including graph weights and recurrent state.
- `completed.json`: source, time window, provenance, and summary of the paired replay. The historical assay has independent simulated accounts and never mutates the forward balances.

The fly uses a centered anti-Hebbian rule driven by neural rates and equity reinforcement. It has no differentiable scalar objective, backpropagation gradient, or PPO optimizer. Its `loss` and `backprop_gradient` fields are explicitly null. Connection deltas and trace dynamics are the relevant learning measurements; changing weights alone does not demonstrate useful learning.

Forward `latest.json`, `paper.db` ledgers, and six-hourly `candidates/<period>` training directories retain the same decision and optimizer diagnostics. Detailed raw per-neuron snapshot arrays are limited to the bounded diagnostic replay to keep normal collection small. Runtime checkpoint cleanup retains currently referenced state and approximately one day of old checkpoints.

New training or replays must use a separate run directory/protocol when changing source cadence. Never select a policy or hyperparameter because it performs best on the inspected held-out test. All returns use simulated execution costs; no real-order interface is connected.

## Cloud verification

On September 11, 2026, deployed code `b88d9341c2420e5505587ed46eb8f0a60ce5cf4c` completed 8,192 PPO transitions, all 512 optimizer updates, and the paired 32-step native replay. Every saved spike total, plastic-weight delta, and changed-edge count matched the raw native arrays. The final saved gradient tensors matched the logged norm; all gradient norms after clipping were at or below 0.5. The objective components, action probabilities, and transition chronology also passed their consistency checks. The first automatic five-minute cycle persisted a fresh observation and emitted its structured progress events. [Linux CI passed](https://github.com/Air-crypto/Stonkfly-Trading/actions/runs/34650927271); all 24 lab tests passed.

This verifies execution and diagnostics, not useful learning or profitability. Full performance records and account-level operational metadata are retained locally rather than included in this public engineering record.

## Historical price warm-up

The deployed worker enables `historical_warmup=True`. A separate immutable `warmup-context.json` contains 63 contiguous closed candle prices strictly before the first forward observation. The context has zero volume, matching the public book snapshots, and is never used as an execution quote. Its fetch time, source, cadence, and hash are recorded. Current headlines remain timestamp gated and are never attached retroactively to the candles.

Only enough context to reach the 64-observation feature window is prepended during inference. As fresh observations arrive, the historical prefix shrinks; at 64 fresh observations it disappears. Existing forward data, accounts, pending decisions, and checkpoints are preserved. The first new decision starts from the existing account state and can only execute at a subsequent observed quote. A context-enabled run cannot silently disable the mode, and changing/removing a context file while it is in use fails closed.

`historical_context_enabled` records the activation and provenance. Each decision logs `historical_context_rows` and `forward_observations`; `ticks` always counts real observations. The 400-observation retraining threshold and six-hour training cadence use only the forward archive. History accelerates the first paper decision; it does not create elapsed forward evidence or demonstrate profitability.
