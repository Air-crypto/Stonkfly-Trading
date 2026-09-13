# Goal checklist and roadmap

Reconciled on September 13, 2026. The objective is to make the fly inspectable,
use that inspection to diagnose poor learning, test changes against trading
outcomes, and repeat useful experiments. A verified visualization or weight
change completes an instrumentation task; it does not complete the trading
improvement task.

**The debugger is implemented. Effective learning and profitable trading remain
unproven.** Study 12's lower-rate candidate failed the fresh-data comparison.
The next change should address economically useful decisions, not pursue smaller
updates as an outcome. Additional neural diagnostics should resolve a specific
failed trade or learning hypothesis.

## Completed

- [x] Build an interactive fly debugger: spikes, membrane voltage, triggering
  inputs, decoder counts, connection weights, and stored `u/w` over time.
- [x] Distinguish source firing from weight changes, show paired recordings,
  find the first/next difference, and share selected observations and bins.
- [x] Support full-neuron lookup from retained arrays and bounded input tests:
  price patterns, timestamped news, neuron stimulation and connection restoration.
- [x] Run new native assays in Modal, with isolated outputs, owning call IDs,
  persistent claims, shared budget accounting and no automatic retry of an
  uncertain capture. Recordings can be inspected without new model compute.
- [x] Add README figures, portable recordings and reproduction commands, and
  commit/push the work to the requested repository.
- [x] Complete study 11: 16 conditions, 360 neural observations and 18,000 bins,
  with reconstructed accounts, inputs, feedback and plasticity. Every condition
  lost money; none passed development. No policy was promoted.
- [x] Reconcile its 400 account marks into price exposure, fees, spread,
  slippage, terminal exit allowance and missing-quote deductions. In test,
  reset's +$2.48 midpoint component was outweighed by $17.55 in costs;
  it filled 20 trades versus pristine's 10. See the
  [loss breakdown](fly-loss-attribution.md). This is a fixed-fill diagnosis,
  not a profitable alternative backtest.
- [x] Link all 360 decisions to their later execution slots and all 100 fills
  to their originating decisions. Inspect a single-gate BUY against the raw
  recording and paired browser view; distinguish BUY exposure targets from
  actual purchases or rebalance sales.
- [x] Add descriptive cash, one-entry holding and constant-target references.
  Both exposure references beat the fly variants in study 11 test but lost in
  development. This keeps the diagnosis focused on decision timing and exposure
  as well as costs. These [post-experiment references](market-baselines.md) are
  not new registered selection arms or evidence of generalization.
- [x] Explain concrete mechanisms: retained firing histories can update a
  currently quiet source connection; selective history resets can magnify
  updates; a lower learning rate can still create an extra gate spike and BUY.
- [x] Test the lower rate in a controlled neural assay: 15 observations and
  750 bins. Updates became smaller, but better trading was not established.
- [x] Export and independently check fresh RAY/STONK paper checkpoints, each
  with 214 existing training observations. No new training occurred during export.
- [x] Register and deploy study 12 before its future market window. It compares
  pristine frozen, trained frozen, online eta 0.001 and online eta 0.0001, with
  separate $1,000 accounts per arm and phase.
- [x] Finish all 16 study 12 captures and independent full-array audits: 372
  observations, 18,600 bins and 400 account marks. Lower eta failed selection;
  all arms lost in both phases. Publish the [results and loss diagnosis](fly-rate-result-12.md).
- [x] Trace its first changed action from raw counts into SELL and later BUY
  fills; verify the paired browser view and publish portable recordings.
- [x] Stop the completed temporary study app after verifying its idle worker.
  Its conservative compute ledger accounted for $0.90664 against a $3 allowance.

The [debugger guide](fly-debugger.md), [completed financial comparison](fly-online-comparison.md),
[learning-rate assay](fly-learning-rate.md) and [study 12 deployment record](../reports/fly-rate-market-execution-12.json)
provide the evidence and limitations. The current debugger JavaScript still
matches the [completed full/portable browser check](../reports/fly-learning-scale-browser-01.json).

## In progress

- [x] Implement a [Solana launch pilot](solana-live.md) with confirmed on-chain
  events, five-second observation targets, native fly plasticity and a trainable
  entry/exit readout. Preserve malformed/rejected/missing data and apply delayed,
  cost-inclusive paper execution. Live event checks and 49 combined Solana/account tests pass.
- [x] Verify an actual cloud training prefix: 22 native-array checks, 15 readout
  updates, 10 reconstructed paper fills and 5.001-second median step spacing.
  The [recorded check](../reports/solana-live-13-check.json) is not a completed
  holdout or proof of profitable learning.
- [x] Implement canonical PumpSwap migration tracking and completed-checkpoint
  continuation, preserving both account balances and native/readout state.
- [x] Verify the current continuation in the cloud: opening balances and the
  first native weights match its parent exactly, and restored readout predictions
  reproduce from the parent checkpoint. Up to three 15-minute
  windows are dispatched, with $25 maximum orders and 2.5% target exposure;
  inference still follows one selected token rather than every collected launch.
- [ ] Complete the cost-aware action/readout experiment using the audited first
  SELL–BUY divergence. Include timing and exposure: lower eta's development
  loss worsened mainly through costs, but its test loss worsened through the
  fixed-fill midpoint component despite almost unchanged friction.
- [ ] Test its mechanics on retained development examples, then preregister
  a fresh comparison against unchanged controls and cash. Existing study 11/12
  results are now diagnostic data, not a new untouched test set.
- [ ] Require improvement after costs on new data before broadening or promoting
  the candidate. A rule that merely stays in cash has not established profitable learning.

The [results workflow](fly-rate-market.md#download-audit-and-inspect-the-completed-comparison)
downloaded and audited every study 12 condition without constructing a local
brain. The [completion record](../reports/fly-rate-completion-12.json) preserves
the archive hashes, stopped study app and separate retained paper-lab deployment.

## Important gaps

- [ ] Establish that learning improves trading on unseen data. Study 11 did
  not establish this, and a smaller update in the neural assay is insufficient.
- [ ] Validate the decision rule and reward timing. The older studies use a
  fixed BUY/SELL/HOLD readout; the Solana pilot adds a trainable head. Native
  plasticity still updates its configured 7,835 connections, not all
  approximately 25.6 million graph connections.
- [ ] Demonstrate generalization across more pools and separate dates, with
  compact PPO and simple cash/holding baselines evaluated under the same rules.
- [ ] Measure whether news improves decisions beyond price-only inputs. The
  timestamped news integration exists; its incremental trading value is unproven.
- [ ] Separate tradability/coverage failures from model losses. Some inventory
  is stress-marked at zero when quotes fail eligibility checks. Such a mark is
  not a realized sale or proof of a market crash.
- [ ] Validate fills against executable market prices before any live-readiness
  claim. Current fills use adverse indicative-price simulation.
- [ ] Demonstrate sustained net performance after fees, spread, slippage and
  hosting. No monthly profitability has been established.
- [ ] Measure total provider billing, including storage/builds, against the
  $100 first-month ceiling and $20–40 steady target. Compute ledgers and stop
  thresholds are implemented; those ledgers are not provider invoices.

Discovery is also bounded: the collector samples a changing universe with a
240-pool tracking capacity and received new-launch events. Study 12 isolates
two fixed pools. Neither is coverage of every current or newly created Solana
memecoin; the indexer can also return pools that are not memecoins.

## Roadmap and decision gates

1. **Completed gate: study 12.** All 16 conditions passed their audits and
   coverage threshold. Development selection preceded every test simulation;
   no candidate was selected. Preserve its sources, missing observations and
   failed hypothesis. Do not turn the best test arm into a selected winner.

2. **Choose the next change from the failed trades.** Distinguish trading costs
   and excessive turnover, poor directional decisions, reward timing, and data
   gaps. Compare notional turnover and exposure, not fill count alone: the
   constant-target reference had more test fills than reset but less notional
   turnover and a better outcome. If coverage is inadequate, fix collection
   first. If extra gated trades
   drive losses, test a cost-aware abstention/readout rule. If updates consistently
   hurt an otherwise useful readout, test reward alignment or a constrained
   learning rule. Register one change and its acceptance rule before collecting
   new evaluation data. These are conditional designs, not deployed changes.

3. **Broaden validation.** Use more eligible pools across separate time windows,
   preserve disappeared/rejected pools, and reserve chronological holdouts.
   Compare trained fly, frozen fly, compact PPO and simple baselines with equal
   capital, inputs and costs. Include news/no-news controls. Require consistent
   improvement across windows before moving a candidate into the main paper
   trader. Do not choose a model using its final holdout.

4. **Run sustained forward paper trading.** Track decisions, delayed fills,
   drawdown, turnover, quote failures, checkpoint changes and cloud costs over
   multiple weeks. Check executable-price assumptions and live operational
   failure modes. A short favorable replay does not establish this milestone.

5. **Optimize a useful policy for the hosting budget.** Measure runtime, memory
   and total billing first. Test quantization or other optimizations against
   both decisions and account outcomes: changed precision can change spike gates.
   Keep cloud operation independent of the laptop. Real-money operation remains
   a separate future decision and is not authorized or enabled by this roadmap.

The repeat loop is: identify a concrete failure, inspect its recording, specify
one change, register a fresh comparison, audit outcomes, then retain or reject
the hypothesis. A failed hypothesis remains a result. The project does not
require labeling an unsuccessful strategy as improved.

### Recurring cross-launch paper RL

The single-token continuation completed 45 minutes with only twenty native
observations; the last thirty minutes had none as PAPER lost usable liquidity or
activity. The next implementation is a [recurring cross-launch service](solana-online.md):
shared account, eight active tokens, preserved inventory, shared fly/Q learning,
checkpoint continuation and budget-paced Modal dispatch. Unit and entrypoint-logic
checks cover multi-token accounting, reward isolation, rotation, continuation and
single-flight scheduling. Deployment and observed learning must be reported
separately; these tests are not trading-performance evidence.

Deployment verified on September 13, 2026: `fly-paper-solana-online` is active,
with a fifteen-minute Modal coordinator schedule and twelve-hour training-window
cadence. Window `solana-online-20260913-183646` resumed the prior account. Its
84-row published prefix contained 39 native observations across three tokens,
30 new Q updates and 15 simulated fills. Independent cash/inventory/fee
reconstruction passed, and the first plastic weights exactly matched the parent
checkpoint's last recorded weights. These are mechanics and learning-activity
checks, not evidence of profitable learning. Sixty tests passed. The subsequent
worker version parks unavailable/dust holdings and records full portfolio marks
for independent valuation checks; the initial window's older ledger does not
contain those marks. Continuous discovery between windows remains unimplemented.
