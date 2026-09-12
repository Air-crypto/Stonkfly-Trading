# Longer online learning comparison

The study 11 runner and offline auditor are implemented and tested, and the
once-only dispatcher is connected after the normal paper cycle. The
[registration](../reports/fly-market-study-11-preregistration.json), recorded
September 12, 2026 at 20:17:29 UTC, fixes development at **20:35–22:35 UTC** and
test at **22:35–00:35 UTC** (ending September 13). In Chicago these are
3:35–5:35 p.m. and 5:35–7:35 p.m. on September 12. The
[cloud arming check](../reports/fly-online-cloud-arming-11.json) verified that
the 20:25 worker committed the exact registration and execution-source hashes
at **20:25:47 UTC**, before development began. Its owning call
`fc-01M2BMPXVS3ZSQY2YVD4TN5RM1` completed with `paper_online_collecting`.
This proves prospective registration in the deployed worker; no study chunk
had run at that check, and no performance result is claimed.
The [validation record](../reports/fly-online-validation-01.json) retains the
native artifact/source hashes and the completed test scope.
The dispatcher retains owning Modal call/input IDs and persists development
selection before any test condition runs. Normal paper policies are unchanged.

The previous [trace-reset experiment](fly-debugger.md#learning-trace-reset-result)
showed that clearing only KC/DAN rate traces between images changes later online
updates and an action, while frozen spike counts remain unchanged. It did not
measure a profitable improvement. This follow-up extends the comparison from
three images to two hours of development and a separate two hours of test data.

## Fixed comparison

| Condition | Starting connection memory | Updates during each phase | Between observations |
| --- | --- | --- | --- |
| `pristine_frozen` | Native initial memory | Frozen | Carry activity |
| `trained_frozen` | Pinned paper-trained memory | Frozen | Carry activity |
| `trained_online_carry` | Same pinned paper memory | Online equity feedback | Carry activity |
| `trained_online_reset_rates` | Same pinned paper memory | Online equity feedback | Clear only `rate_kc` and `rate_dan` |

The definitions are in `paperlab/fly_online_protocol.py`. The timestamped
registration pins the completed mechanism report/audit, study 10 parent
plan/result, actual imported memory, and the future evaluation windows.

Each phase has **24 five-minute decision slots plus a terminal execution/mark**.
The two previously pinned pools, ALL and baton, retain their actual audited
paper memories. Each condition starts with two $250 accounts and $500 idle cash,
for $1,000 aggregate paper capital. Each pool, condition and phase starts with
fresh neural activity and its declared connection memory. Development accounts
and learned updates never become test initial state. This research comparison
is separate from the normal four-allocation paper portfolios.

All conditions receive the same sealed price history and causally available
news vectors. Each online condition computes its own feedback after executing
the preceding decision: equity change above $0.01 triggers reward, below
-$0.01 triggers aversive stimulation, otherwise no pulse. The existing 200 ms
pulse at current 20 and 500 ms neural observation remain unchanged. Online
plasticity remains enabled without a pulse. Missing quotes skip observation;
the first observation following an unpriced gap receives zero feedback.

The sole candidate is `trained_online_reset_rates`. Development selection
requires at least 18 of 24 observations **in each pool**, identical coverage
across conditions, and equity strictly higher than every control and cash plus
the declared hosting allocation. That allocation is $40/month prorated to two
hours on a 30-day month, or $0.111111 per $1,000 comparison. Fees, artificial
spread and slippage remain the existing adverse DEX scenario. Every test
condition is evaluated regardless of selection; no policy is promoted
automatically. Two pools and four hours cannot establish monthly profitability.

## Complete recordings and audit

There are 16 complete chunks: two pools × four conditions × two phases.
A chunk captures an entire two-hour phase in one invocation; it does not resume
partially saved neural dynamics. The runner refuses to overwrite an existing
output. Test chunks require all eight development summaries before loading
the graph. Before the first market window starts, the scheduled worker must
commit an arming receipt that pins the registration, runner/auditor modules,
native sources and deployed worker source. Missing that deadline stops this
registration; the worker does not silently move its windows.

Each scheduled call commits its normal paper cycle first. Research is deferred
if that cycle took over 90 seconds or insufficient call time remains. Sealing
and each chunk have durable claims before work begins; failed and uncertain
claims are never retried or skipped. Source or registration changes before
study completion stop research. Selection is committed with its own receipt
after all eight development chunks and before any test chunk. All sixteen
capture results still require independent audits before publication.

The existing single worker, 600-second reservation and $25 monthly worker cap
remain unchanged, as does the separate $15 collector cap. No GPU is added.
After the four-hour market window, sixteen worker slots take roughly another
80 minutes if every slot can run a chunk; missing data, long paper cycles or
provider interruptions can delay or stop execution. Full recordings remain on
the existing volume. Modal currently lists volume storage at $0.09/GiB-month
with 1 TiB/month included, separately from compute; account-wide consumption
determines any actual charge ([pricing, checked September 12, 2026](https://modal.com/pricing)).

For each eligible observation, retain all 50 native bins, all neuron counts and
voltages, every plastic weight and `u/w` update, before/after boundary arrays,
complete end dynamics, input pixels, and the paper ledger. The independent
auditor reconstructs the original visual model's fixed R8-to-aMe12 sign
corrections from locked graph data and annotations before checking whole-graph
weights. It then reconstructs the rate rule for every plastic connection and
bin, validates the fixed decoder, and replays fills and equity feedback.

Full float32 weights must match exactly. Reconstructed rate and `u/w` arrays
use relative tolerance 1e-11 and absolute tolerance 1e-12. Only the derived
float32 update norm receives the existing four-ULP reduction allowance.
End-state continuity checks cover reset targets, untouched state and clocks.
News and raw price audits retain all decision slots, disappeared pools and
late/unavailable observations.

The synthetic native validation recorded 21 eligible observations and three
missing decision slots, covering **1,050 bins × 7,835 plastic connections**.
The runner took approximately 90 seconds locally. This is a runtime measurement
for this fixture, not a cloud timing promise. Raw artifacts occupied about
766 MB; full experiment storage/transfer therefore needs to be accounted for
before scheduling. No recorder output is silently reduced to three images.

The browser check covers all 21 observations, 84 selected bins, all 24 decision
slots and the terminal mark, online/reset labels, frozen-control labels, a
recording larger than the former 15 MB import ceiling, and mobile overflow.
The import limit is now 64 MiB. Every model-submission request is blocked during
this check. Audited synthetic inputs receive an explicit implementation-check
badge; the screenshot below contains no real market performance claim.

![Long recording checked with synthetic prices](assets/fly-online-validation-01.png)

## Run the implementation checks

From the repository root, with the full graph prepared using the
[existing setup instructions](fly-debugger.md):

```sh
FLY_TRACE_DATA="$PWD/data/fly" uv run --extra dev pytest -q \
  tests/test_fly_online.py::test_native_long_online_chunk_reconstructs_feedback_boundaries_and_every_bin \
  --basetemp=runs/online-native-check-02

# Re-audit retained recordings; this loads arrays but does not simulate a brain.
uv run python -m paperlab.fly_online_audit \
  --root runs/online-native-check-02/test_native_long_online_chunk_0/online \
  --fly-data data/fly --out runs/online-native-check-02/offline-audit

# Read-only viewer. It does not need the graph and cannot submit model work.
uv run python -m paperlab.debugger serve --out runs/online-native-check-02 --port 8767
```

In that viewer, select `offline-audit` or import its `view.json`. Pytest owns and
recreates its `--basetemp` directory; use a new directory when preserving an
earlier fixture. The audit CLI also refuses to overwrite existing output.

With Playwright/Chromium installed and that server running:

```sh
FLY_VIEW_URL=http://127.0.0.1:8767 \
FLY_ONLINE_VIEW=runs/online-native-check-02/offline-audit/view.json \
node scripts/check-fly-online-view.cjs
```

The browser check also reads the shipped frozen control at
`examples/fly-debugger/market10-pool0-trained_frozen/view.json`. It replaces only
browser read responses with fixtures, and aborts `/api/run`.

`python -m paperlab.fly_online_study --help` describes the complete-chunk CLI.
Use it only with a sealed study 11 plan and the matching memory/news archives.
The CLI does not create a registration or schedule a cloud job. Deployment uses
the already authorized app and all five feature flags:

```sh
PAPERLAB_SCHEDULE=1 PAPERLAB_FLY=1 PAPERLAB_UNIVERSE=1 \
PAPERLAB_PAPER_STUDY=1 PAPERLAB_ONLINE_STUDY=1 uv run --extra cloud modal deploy cloud.py
```

This requires the two verified private memory exports already described in
the checkpoint study setup. Keep the registration and pinned execution sources
unchanged until completion. The dispatcher writes receipts and snapshots below
`/state/registered-paper-11`; actual chunk artifacts live in
`chunks/<phase>-pool<index>-<arm>/artifacts`. An absent or timed-out observation
of a call does not authorize another submission. Fresh data, prospective cloud
receipts and complete comparison audits are required before reporting a result.

## Observe the scheduled experiment

The observer reads the existing volume and reattaches saved owning calls. It
does not invoke a worker, resume partial neural state, or submit model compute.
From a checkout matching the pinned execution sources, with Modal authenticated:

```sh
uv run --extra cloud python -m paperlab.fly_online_cloud \
  --registration reports/fly-market-study-11-preregistration.json \
  --out runs/online-cloud-11
```

Before the endpoint, this reports the registered collection phase. Afterward,
it verifies completed chunk summaries and the development-selection receipt.
An in-flight owning call is reported as pending, including when a completed
chunk is waiting for its outer worker to return. Repeat the same command to
observe it again. The last successful observation is saved in `progress.json`
with an observation timestamp; it is not a continuously refreshed status page.

Once chunks become available, download and independently audit their full
recordings using the prepared graph:

```sh
uv run --extra cloud python -m paperlab.fly_online_cloud \
  --registration reports/fly-market-study-11-preregistration.json \
  --out runs/online-cloud-11 --audit --fly-data data/fly

uv run python -m paperlab.debugger serve \
  --out runs/online-cloud-11/views --port 8767
```

Allow approximately 12–15 GB locally for the full recordings, based on the
native validation fixture; actual size depends on eligible observations. The
observer reuses verified downloads and links the full neuron recordings into
each view without duplicating their data. Repeat the audit command as more
chunks complete. These local inspection commands can exit while cloud
collection and scheduled evaluation continue with the laptop closed.

The raw price archive is audited before any neural recording is downloaded.
Each view is published only after its complete chunk audit passes. The final
`report.json` requires all sixteen independent audits, matching native builds
and fresh starting states, and development selection preceding every test
chunk. Partial audited views are useful for diagnostics; they are not a
completed comparison. The report retains every condition's development/test
equity and coverage and never promotes a policy automatically.

In the viewer, select `development-pool0-trained_online_reset_rates` and compare
it with `development-pool0-trained_online_carry`. Follow the learning-drive
plot, connection `u/w`, gate spikes and subsequent fills across observations;
repeat for pool 1 and the separate test phase. The reset clears KC/DAN **firing
rate traces**, not a learning-rate hyperparameter or stored connection memory.
Keep the pinned execution sources unchanged until capture and audits finish.
