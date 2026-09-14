# GSPO-inspired paper replay: measured pilot

Run: `group-replay-20260914-01`. Paper-only. **blocked: mechanics pilot only; more independent dates/mints and execution validation required**

Completed 32 groups × 12 branches = 384 training trajectories, 48 gradient updates. 8 groups had no reward variance and skipped training.
The 900-parameter actor used frozen pristine full-fly features. Native observations: 81; native weight delta: zero. Actor checkpoint L2 change: 0.662850.

Four earlier mints trained the actor; four later mints, excluded from the entire earlier archive, tested it. Each branch starts with its own $1,000 cash, targets at most 0.25% exposure, and lasts 60 seconds. These are isolated token episodes, not a combined portfolio or a forecast of monthly returns.

![Measured diagnostics](group-replay-01.png)

## Matched evaluation

PnL includes assumed entry fees, spread, slippage and liquidation costs. Missing terminal quotes value inventory at zero; open inventory is marked, not fabricated as a sale. Twelve seeds reduce action-sampling noise, not market uncertainty.

| Split | Policy | Mean PnL (USD) | Mean reward | Mean fills | Mean paid fees (USD) |
|---|---|---:|---:|---:|---:|
| train | trained | -0.486764 | -0.677303 | 1.77 | 0.040571 |
| train | initial | -0.736027 | -0.998306 | 3.29 | 0.081086 |
| train | cash | +0.000000 | +0.000000 | 0.00 | 0.000000 |
| train | buy_hold | -0.690004 | -1.016405 | 0.75 | 0.023438 |
| test | trained | -0.392294 | -0.585295 | 3.42 | 0.076702 |
| test | initial | -0.493709 | -0.678091 | 6.25 | 0.154828 |
| test | cash | +0.000000 | +0.000000 | 0.00 | 0.000000 |
| test | buy_hold | -0.397982 | -0.652552 | 1.00 | 0.031250 |

## Weight updates and verification

First / last objective loss: -0.00087836 / -0.00054675. Mean pre-clipping gradient norm: 0.197540. Group-normalized loss values do not establish convergence or profitable learning.
Independent Decimal audit passed: 592 trajectories, 7696 account rows, 1624 fills. The audit checks receipt timing, fees, cash, inventory, branch features, rewards, aggregate results and actual actor checkpoint changes.
Wall time: 243.1s. Conservative compute estimate: $0.0672. Shared worker ledger after settlement: $7.3128; this is not a provider invoice.

## All twelve alternatives from the first training group

Mint: `gKWYJZKVfsa73RpdUHGSXAhGCP5hXQ91VQrSdbspump`. Actions shown as seconds from episode start. HOLD steps are omitted for readability; every timestep and rejected/expired order remains in the raw ledger.

| Branch | Requested orders | Fills | Net PnL (USD) | Reward |
|---:|---|---:|---:|---:|
| 1 | 0s target_0.15_percent, 10s exit, 25s target_0.25_percent, 40s exit, 55s target_0.25_percent | 5 | -0.343533 | -0.430406 |
| 2 | 0s target_0.25_percent, 5s exit, 15s target_0.15_percent, 35s exit, 40s target_0.15_percent, 45s exit, 50s target_0.25_percent | 7 | -0.424004 | -0.532104 |
| 3 | 0s target_0.15_percent, 10s exit, 20s target_0.15_percent, 25s exit, 35s target_0.15_percent, 40s exit, 45s target_0.25_percent, 50s exit, 55s target_0.15_percent | 9 | -0.459130 | -0.574679 |
| 4 | 0s target_0.25_percent, 5s exit, 10s target_0.25_percent, 20s exit, 40s target_0.15_percent | 5 | -0.343671 | -0.431378 |
| 5 | 0s target_0.25_percent, 45s exit, 55s target_0.15_percent | 3 | -0.202308 | -0.254711 |
| 6 | 10s target_0.25_percent, 15s exit, 20s target_0.15_percent, 30s exit, 35s target_0.25_percent | 5 | -0.342483 | -0.431701 |
| 7 | 5s target_0.25_percent, 15s exit, 20s target_0.15_percent, 40s exit, 45s target_0.25_percent | 5 | -0.343174 | -0.431547 |
| 8 | 0s target_0.25_percent, 15s exit, 20s target_0.15_percent, 25s exit, 40s target_0.15_percent, 45s exit, 50s target_0.15_percent | 7 | -0.375639 | -0.471160 |
| 9 | 0s target_0.15_percent, 10s exit, 15s target_0.25_percent, 20s exit, 25s target_0.15_percent, 45s exit | 6 | -0.290185 | -0.364765 |
| 10 | 0s target_0.15_percent, 5s exit, 10s target_0.15_percent, 20s exit, 25s target_0.15_percent, 30s exit, 45s target_0.15_percent, 55s exit | 8 | -0.321454 | -0.403222 |
| 11 | 10s target_0.25_percent, 15s exit, 20s target_0.25_percent, 30s exit, 45s target_0.15_percent | 5 | -0.347262 | -0.435705 |
| 12 | 0s target_0.15_percent, 5s exit, 20s target_0.15_percent, 25s exit, 30s target_0.25_percent, 45s exit, 50s target_0.15_percent, 55s exit | 8 | -0.373047 | -0.467804 |

## Limits and next evidence gate

This pilot does not update native fly synapses, ingest news, change the live trader or schedule recurring GSPO jobs. A matched current-Q comparison is deferred because its encoder, reward conditioning and account/action settings differ. There is no market-only ablation yet, so any improvement cannot be attributed to the fly features. Reserve-based quotes are indicative; actual execution, latency, failed transactions and market impact remain unvalidated.
Before promotion: preregister a new forward collection across more independent dates and mints, retain failed and dead tokens, compare against cash, frozen/random, market-only and matched simple policies, then evaluate outcomes after realistic execution and hosting costs. Do not tune on or reuse these test results as a fresh holdout.

Raw local artifacts: `/Users/arihan/Documents/Codex/2026-09-10/can-you-look-through-x-and/work/Stonkfly-Trading/runs/group-replay-20260914-01`. Includes all rollout rows, masks, behavior probabilities, losses, gradients, sequence ratios, clipping, entropy/KL, feature provenance and model checkpoints.

Algorithm reference: [Qwen GSPO](https://qwenlm.github.io/blog/gspo/). This is a trading adaptation, not a claim of validated GSPO trading performance.
