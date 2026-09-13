# Goal checklist and roadmap

Reconciled on September 13, 2026. The objective is to make the fly inspectable,
use that inspection to diagnose poor learning, test changes against trading
outcomes, and repeat useful experiments. A verified visualization or weight
change completes an instrumentation task; it does not complete the trading
improvement task.

**The debugger is implemented. Effective learning and profitable trading remain
unproven.** The next decision gate is an improvement in account outcomes on
fresh data. Additional neural diagnostics should resolve a specific failed
trade or learning hypothesis.

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

The [debugger guide](fly-debugger.md), [completed financial comparison](fly-online-comparison.md),
[learning-rate assay](fly-learning-rate.md) and [study 12 deployment record](../reports/fly-rate-market-execution-12.json)
provide the evidence and limitations. The current debugger JavaScript still
matches the [completed full/portable browser check](../reports/fly-learning-scale-browser-01.json).

## In progress

- [ ] Complete study 12's **06:00–08:00 UTC development** and **08:00–10:00 UTC
  test** collection on September 13 (1–5 a.m. Chicago time overall). The cloud
  witness was verified at 05:43:24 UTC. Collection is followed by chronological
  simulation, not by live orders during that window.
- [ ] Capture all 16 complete pool/arm/phase conditions, then independently
  audit their prices, news, trades, feedback, learning bins and displayed curves.
- [ ] Publish the results, paired debugger views and cost/turnover comparison;
  stop the temporary study app after verified completion.

The [results workflow](fly-rate-market.md#download-audit-and-inspect-the-completed-comparison)
is implemented and tested. It reads existing calls, downloads hash-matching
recordings and builds audited views without constructing a local brain. Its
implementation checks are not study 12 performance results. Completion time
depends on coverage and cloud calls after 10:00 UTC; it is not promised at the
collection endpoint.

## Important gaps

- [ ] Establish that learning improves trading on unseen data. Study 11 did
  not establish this, and a smaller update in the neural assay is insufficient.
- [ ] Validate the decision rule and reward timing. The fly currently uses a
  fixed BUY/SELL/HOLD readout; only its configured 7,835 plastic connections
  learn, not all approximately 25.6 million graph connections.
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

1. **Finish the registered comparison.** Preserve its current sources and
   cohort. Reconstruct all 16 results, retain missing observations, and report
   every arm. Development selection is fixed before any test simulation. Low
   coverage makes the result inconclusive; it does not permit replacing a pool
   retrospectively. A favorable two-pool result only earns replication.

2. **Choose the next change from the failed trades.** Distinguish trading costs
   and excessive turnover, poor directional decisions, reward timing, and data
   gaps. If coverage is inadequate, fix collection first. If extra gated trades
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
