# Training and reward-credit audit — 2026-09-14

The original training loop updated weights, but **did not train on the complete economic outcome it reported**. Missing observations, parked tokens, rejected intents, and the episode boundary could erase unresolved reward credit. This is a correctness defect, not merely disappointing trading performance. Historical checkpoints and ledgers remain unchanged; they must be labelled as trained under that earlier protocol.

## Reproduced discrepancy

The eleven archived independent $1,000 episodes in `runs/market-attribution-01` contain 1,028 Q updates. Their raw rewards, before clipping or discounting, sum to **+$2,071.492264**, while their reported marked PnL sums to **−$2,828.329599**. The discrepancy is **−$4,899.821864**. These are separate reset accounts, not a single compounded balance.

| Episode start UTC | Reported marked PnL | Raw rewards actually supplied to Q updates |
|---|---:|---:|
| 10:02 | −$603.32 | +$650.80 |
| 11:02 | −$209.23 | +$100.52 |
| 12:02 | −$187.07 | +$128.43 |
| 13:02 | −$44.02 | +$568.06 |

The old loop cleared `previous` and continuity state when a quote became unavailable, removed per-token contexts when parking inventory, and saved checkpoints without settling the last action. It stopped new inference during the final thirty seconds but could still execute orders and change marks during that interval. None of the eleven episodes recorded a terminal Q update. The dollar discrepancy is not an artifact of reward normalization: this comparison uses the raw `reward_usd` field before scaling.

Some old zero marks were themselves consequences of conflating entry eligibility with quote availability; see [the market audit](system-audit-market-20260914.md). The historical rewards and historical PnL should not be retroactively rewritten or relabeled as executable losses.

Reproduce the evidence without inference, networking, cloud work, or account changes:

```sh
../venv/bin/python scripts/audit_online_reward_credit.py \
  runs/market-attribution-01 reports/system-audit-training-20260914.json
```

The [JSON evidence](system-audit-training-20260914.json) records every episode, source-ledger SHA-256, update count, raw/scaled rewards, native feedback, and discrepancy. Inputs are ignored archived runtime artifacts and are not present in a clean public clone.

## Prospective corrections

`mint_credit_terminal_v2` retains each mint's outstanding Q transition and marked-contribution anchor through quote gaps, expiry, and inactive parking. When a fresh quote returns, the next update uses the net change since the last credited observation. It does not count a temporary zero mark followed by quote recovery as a profit. An attempted order rejected without inventory or reward can be explicitly skipped; rejection no longer discards nonzero inventory feedback.

At a completed episode boundary, every outstanding action receives its remaining outcome once, including fills after the last inference. The target contains the final reward with **zero bootstrap discount**; a fresh subsequent account cannot contribute artificial future value. Missing final quotes produce an explicitly tagged `terminal_missing_quote_stress` settlement. This represents the declared conservative accounting convention, not a verified sale or proof a token became worthless.

Terminal settlements are appended in a separate ledger row with no new order, fill, neural observation, or fabricated market receipt. Native and Q checkpoints are saved afterward. The account audit independently checks per-mint credit anchors, contribution changes, terminal targets, update counts, and the equality of summed raw rewards to the final episode PnL. A nonzero unexplained residual fails the cloud completion audit.

Fresh episodes also use `solana_observed_entry_exit_v2`: fresh indicative marks are separate from entry and exit permissions, while actual fills retain observed-liquidity limits. The final result reports full indicative inventory value separately from proceeds allowed by one size-limited exit. These semantics are versioned; old and new results are not directly comparable without replay under one protocol.

The loop records the feature snapshot's time, the SQLite rowid of the last event actually applied under the feed lock, and the exact current FX state. The event cursor excludes a message received before a snapshot but still being parsed. The FX state preserves an unavailable conversion after a failed refresh instead of allowing a replay to reuse a prior successful quote. These fields support causally matched evaluation at the actual source observation and decision times.

## What actually trains

The deployed hourly policy is an **802-parameter Q MLP**, `22 → 32 → 2`, with Tanh, Adam at 0.0003, Huber loss, gradient clipping at norm 1, and 10% random exploration. Its actions request FLAT or LONG exposure. Native fly outputs contribute features, but the fly's fixed BUY/SELL/HOLD decoder is not the executed policy. The separate twelve-branch pilot uses a different 900-parameter stochastic actor and frozen native features; it is not running inside this Q update loop.

The full fly graph is propagated, but its candidate plasticity changes only 7,835 selected KC-to-MBON connections. Those changes use the baseline-centered anti-Hebbian rule with engineered reward/aversive dopamine stimulation; they are not backpropagation, PPO, or GSPO. Checkpoint restoration validates graph/configuration identity and retains the learned native memory plus the Q optimizer and RNG. Resetting neural activity between tokens preserves learned efficacy memory.

Native credit remains limited to uninterrupted observations of the same mint. Switching mints or crossing an unobserved gap resets short-lived native state so another token cannot inherit its eligibility. The Q learner can now receive delayed and terminal outcomes, but no equivalent delayed native-eligibility replay has been implemented. Results explicitly report native feedback and its difference from Q dollar credit; the old eleven-window native feedback sum was −$404.57. Finite gradients or changing synapses therefore do not prove that the whole fly learned the complete trading objective.

## Tests and remaining methodological limits

The focused Solana integration suite passed **80 tests** at this audit stage. New regressions cover permanent quote disappearance, recovery with no artificial positive reward, the last delayed fill after inference stops, separate one-sided exit eligibility, terminal targets that do not inspect a next state, reward/account tampering, and receipt-before-accept event cursors. Tests use small fake neural adapters; the parent system audit records actual native-cloud smoke and forward-run verification separately.

Correct accounting is necessary but insufficient for effective learning:

- The optimized reward remains `clip(dollars / 25, −1, 1)`. In the old archive, 109 of 1,028 updates (10.6%) were clipped. Thus the optimized objective differs from raw profit, especially with $250 positions; raw reward conservation does not remove this modeling choice.
- Position/PnL features retain earlier $25 normalization and saturate for larger balances. This can hide distinctions among materially different inventory sizes. Scaling changes should be versioned and tested on matched future data.
- There is no target network, replay buffer, or convergence guarantee in this small online Q design. Huber loss and finite gradients measure optimization mechanics, not out-of-sample profitability.
- The native rule operates on 500 ms of simulated neural time per observation, with its own trace and memory constants. It has not been calibrated as a causal model of five-second financial decisions or hour-long gaps.
- Frozen evaluation changes native adaptation-related features, including weight-change diagnostics. Matched frozen comparisons are useful, but do not by themselves isolate native learning from the Q head or establish that the fly features add value. A market-only ablation and native/head checkpoint cross-combinations are still needed.
- Marked inventory is indicative, not guaranteed executable cash. Forward matched controls, execution calibration, more independent market windows, and uncertainty estimates remain the profitability gate.

No live-wallet trading is authorized or implied by this audit. Cloud deployments and verification status are documented in the main system audit; the monthly cloud limit remains $100.
