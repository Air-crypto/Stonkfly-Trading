# Stonkfly system audit — 2026-09-14

**Verdict: the infrastructure can collect, simulate, update weights and preserve experiments, but the historical training loop did not receive the complete economic outcomes it reported. Those records do not establish profitable learning.** This audit repairs concrete accounting, credit-assignment and evaluation defects prospectively, preserves the earlier evidence, and distinguishes verified mechanics from the remaining strategy-validation work.

The goal remains a cloud-hosted, paper-first Solana trading experiment that can demonstrate repeatable returns after execution and hosting costs. Authorization remains $1,000 per independent paper episode and a $100 total monthly cloud ceiling. No live trading, wallet integration or funding decision is part of this audit.

## What is built

| Component | Actual implementation and scope |
|---|---|
| Market input | Public confirmed Pump/PumpSwap events, newly observed launches and canonical migrations; SOL/USD conversion. Reconnect gaps have no historical backfill. This is not every Solana token. |
| Current cloud cadence | An hourly bounded fifteen-minute training/collection episode. Five seconds is the target observation cadence, not a guarantee of a new trade or optimizer update every five seconds. Approximately 25% collection duty leaves launches between windows unobserved. |
| Attention and accounts | Up to 128 watched tokens, eight active contexts, one native token observation per step in bursts. Each new episode starts with $1,000 and no inventory; native/head checkpoints and optimizer/RNG state continue. Episode losses remain archived; balances do not compound across resets. |
| Executed policy | An 802-parameter Q head, 22 → 32 → 2, selects FLAT/LONG exposure from market, inventory and native features. Adam, Huber loss, 10% exploration, dollar reward clipped after division by 25, and time-adjusted discounting. |
| Fly training | Full native graph propagation with candidate plasticity on 7,835 KC-to-MBON edges. The engineered dopamine/anti-Hebbian rule and Q optimizer are separate mechanisms. Native feedback only spans uninterrupted same-mint bursts. The fixed native BUY/SELL/HOLD decoder is not the executed policy. |
| Compact model | Earlier 247,780-parameter PPO MLP experiment; not an SLM or transformer and not the current Solana learner. Earlier compact quantization does not make the full native circuit quantized. |
| Twelve-branch pilot | A separate 900-parameter stochastic actor over frozen pristine fly features. Twelve action trajectories share each market tape; they do not create independent market samples or compound twelve accounts across groups. |
| News | Earlier news tooling exists, but news is disabled in the current Solana online feature path. |
| Observability | Decisions, fills, fees, losses, gradients, native activity, weight changes, checkpoints, account audits, control state, budgets and debug visualizations. These describe mechanics, not profitability. |
| Prospective evaluation | Retained native/Q checkpoints and untrained, cash and always-long controls, frozen on a subsequent common tape. The audited v2 protocol fixes timing and provenance issues described below. |

## Material findings and repairs

### 1. Losses were missing from the training signal — critical

Eleven archived episodes had **−$2,828.33 marked PnL**, while their 1,028 Q updates received **+$2,071.49 raw rewards**. The unexplained difference was **−$4,899.82**, before normalization or clipping. One episode reported −$603.32 but supplied +$650.80 to Q updates. Quote gaps, parked contexts, rejected intents and the final episode boundary could erase outstanding credit.

The new `mint_credit_terminal_v2` protocol retains per-mint credit across gaps, settles the remaining outcome once at completion, and uses zero bootstrap for terminal targets. An independent audit reconstructs cash, fills, per-mint contribution changes and total raw rewards. A completed fresh episode must reconcile raw Q reward to its declared final indicative PnL. Old checkpoints remain available and labelled by their original provenance; fixing the code cannot repair the training they already received.

See [training analysis and numerical evidence](system-audit-training-20260914.md).

### 2. Entry filters were also erasing observable holdings — high

Seven of sixteen zero-marked terminal holdings still had fresh reserve prices. Their remaining acquisition basis was $1,370.47. One-sided flow, low reserves or the inherited fixed $25 minimum-capacity check rejected entry eligibility; the old helper also rejected marking and exiting those holdings. Thus these historical zero marks do not establish equivalent realized trading losses.

The new `solana_observed_entry_exit_v2` separates observable price, entry permission and exit permission. Actual fills remain constrained by available size. Reporting separates the full indicative holding mark from proceeds permitted by a single capacity-limited exit. Independent intent checks now verify that each fill follows the recorded target, mint, direction, quantity and a later receipt.

See [market, execution and GSPO analysis](system-audit-market-20260914.md).

### 3. Frozen replay did not reproduce the source information set — high

The earlier replay used the ledger row's end time, after inference, instead of the source observation time; it could also include received-but-not-yet-applied events and reuse a successful FX conversion during a live FX failure. Its order issuance delay differed from the source. These differences undermine a matched comparison.

V2 records an atomic applied-event cursor, pre-inference observation time and actual FX availability. Replay uses those exact inputs and the later recorded issuance time. It follows the same active-token history, reports skipped opportunities, checks native and head weights remain unchanged, and seals the complete relevant Python/native/schema source set. Healthy no-inference windows may retain their parent's checkpoint, with explicit immutable provenance, rather than crashing registration.

The original never-executed v1 batch is archived with a supersession record. A new v2 cutoff must precede its future tape. No old tape is backfilled with invented timing metadata.

See [evaluation audit](system-audit-evaluation-20260914.md) and [versioned protocol](checkpoint-evaluation-protocol.md).

## What results mean now

Neither a positive episode nor a falling loss curve is evidence that learning beats market movement. The eleven historical windows differ in time, exposure and missing quotes, and their reward credit was incomplete. The GSPO pilot's held-out average loss improved from about $0.494 to $0.392, but cash remained better; four held-out mints from the same day are a small pilot.

The frozen checkpoint comparison estimates buy/sell behavior **conditional on the live learner's recorded token-attention opportunities**. It does not independently evaluate universe selection. Online training uses exploration and native adaptation-related features; frozen greedy inference removes exploration and makes weight-change features zero. This difference is explicit and needs a separate ablation. Choosing the best checkpoint on one tape requires a later untouched confirmation set.

Corrected raw reward conservation also does not prove the objective is ideal. Clipping affected 109 of 1,028 old updates; inventory features retain $25 scaling and can saturate with $250 positions. The head omits time remaining in the finite episode, and lacks a target network or replay buffer. Native delayed/terminal eligibility credit has not been implemented. These are versioned experiment changes to investigate, rather than silently mixing them into historical checkpoints.

## Verification and deployment

Verification is in progress. Final test counts, native-cloud smoke results, corrected live terminal reconciliation, retained archive hashes and the new prospective cutoff will be recorded here before this audit is closed.

The services use bounded non-retrying calls, shared writer leases, retained checkpoints and conservative budget reservations. The 8 GiB admission guard covers `/solana-live` archives, not the entire shared volume or evaluation artifacts; one admitted window can add data beyond that threshold. Worker allocation is $85, with $15 reserved for collector/overhead and another 25% admission margin. Non-preemptible compute costs three times the listed CPU/RAM rate; the ledger also applies a twofold safety factor. The ledger is an estimate, not the provider invoice, and not all build/coordinator/storage charges are individually attributed. The $30 Starter credit is not permission to exceed the user's $100 limit. [Modal pricing](https://modal.com/pricing)

## Checklist and roadmap

- [x] Inspect deployed control/call state, source, historical ledgers, reward paths, checkpoint registry, replay and costs.
- [x] Reproduce the historical reward discrepancy and classify zero marks without changing past outcomes.
- [x] Repair Q credit across gaps and episode termination; independently verify reward/account conservation.
- [x] Separate observation, entry and exit semantics; audit fill intent and capacity sensitivity.
- [x] Repair causal replay, FX failures, checkpoint reuse, source sealing and opportunity coverage.
- [ ] Finish native-cloud mechanics and one complete corrected live episode; resume the audited schedules.
- [ ] Complete the first prospective v2 checkpoint comparison, explicitly labelled as a small pilot.
- [ ] Add multiple nonoverlapping future windows and a later untouched confirmation set; report paired differences against cash and always-long, drawdown, turnover, fees, availability and concentration with uncertainty across windows.
- [ ] Add market-only, Q-only and native/head cross-combination controls before attributing value to the fly circuit.
- [ ] Version and test scale-aware features, episode time remaining, reward formulation and Q stabilization; preserve the current version as a control.
- [ ] Evaluate independent token selection and budgeted continuous collection/backfill. Collecting every launch is a separate coverage/cost target, not a property of the current hourly worker.
- [ ] Calibrate executable routing, impact, priority fees, failed transactions, token restrictions and exit capacity against actual read-only quotes.
- [ ] Only after repeated untouched performance exceeds trading and hosting costs, assess a separately authorized live deployment.

The immediate priority is trustworthy forward evidence. Increasing the number of optimizer steps before that evidence would amplify the effect of any remaining simulator or objective mismatch.
