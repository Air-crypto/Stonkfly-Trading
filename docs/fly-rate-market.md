# Fresh market comparison of learning rates

The [lower-rate neural assay](fly-learning-rate.md) reduced later weight movement
but introduced an extra BUY. It did not establish better trading. The next
comparison must measure fresh account outcomes and turnover after execution
costs, with a development decision saved before the test phase.

The [study 12 registration](../reports/fly-rate-market-registration-12.json) now
fixes development at **06:00–08:00 UTC** and test at **08:00–10:00 UTC** on
September 13, 2026. The cloud witness was saved at **05:43:24 UTC**, before
development. Its exact registration bytes, all 67 execution source hashes and
owning call/input were independently read back and verified. The
[deployment record](../reports/fly-rate-market-execution-12.json) retains this
evidence. The temporary app and main paper app are both deployed.
The [market preflight record](../reports/fly-rate-market-preflight-12.json)
contains the 91 passing checks, execution source hashes and observed shared
budget. No study 12 market observations have been simulated yet.

The first metadata call settled to a $0.00303 compute estimate; the shared
worker ledger stood at $3.34372. It only armed the future experiment. Its $3
study allowance, $25 shared worker limit and provider-billing distinction
remain in force.

## Checkpoint preparation

**Checkpoint capture and array audit are complete.** The
[execution record](../reports/fly-rate-checkpoint-execution-12.json)
preserves the owning call, executed sources, selected cohort, independent audit
and stopped temporary app. The [preflight record](../reports/fly-rate-checkpoint-preflight-12.json)
retains the original 25 tests and the observed coverage of the old cohort.

| Selected pool | Recent eligible slots | Existing training observations | Positive / negative rewards |
| --- | ---: | ---: | ---: |
| RAY | 24 / 24 | 214 | 85 / 116 |
| STONK | 23 / 24 | 214 | 92 / 111 |

The snapshot was finalized at 05:15:00 UTC on September 13, 2026, after the
captured checkpoints' last slot ended. These are existing observations, not
new training during export. The call constructed one cloud reference, performed
zero observations or orders, and settled to a $0.04828 compute estimate. The
provider invoice remains unverified. The main paper app remains deployed.

All ten artifacts downloaded with matching hashes: 97.7 MB, no read retries.
Local verification reproduced cohort selection and reward exposure and compared
`u/w` to the saved checkpoints. The [independent array audit](../reports/fly-rate-checkpoint-array-audit-12.json)
then reconciled all 7,835 exported plastic weights and 25,575,103 nonplastic
weights per checkpoint against the locked graph and original visual adapter.
No local brain was constructed.

The [portable checkpoint archive](../reports/fly-rate-checkpoint-12.zip) includes
both exported memories, their audit, selection and capture records, completed
cloud result, local verification and independent array audit. It omits the raw
universe, ledger, full checkpoints and graph required to repeat the full audit.

```sh
python3 -m zipfile -e reports/fly-rate-checkpoint-12.zip runs/rate-checkpoint-12-export
```

In the checked snapshot, ALL had only 6 eligible observations in the last 24
decision slots and its newest quote was stale. Baton had 24. Reusing this pair
would risk an uninformative comparison. The old pair, its recordings and failed
study 11 selection remain unchanged.

The new capture selects two distinct Solana contracts from the currently
assigned fly paper sleeves. Each needs a current eligible quote, 64 historical
eligible observations, at least 18 eligible observations in the last 24 completed
five-minute slots, and a versioned checkpoint with complete positive and negative
reward exposure. Candidates are ranked by current five-minute volume and then
pool key. Equity and returns do not enter the ranking. This is a small Solana
pool sample; the discovery feed does not establish that every pool is a memecoin.

The worker freezes the ledger, universe snapshot and both checkpoints under the
shared worker lock. It waits until the captured checkpoint's final slot has
ended, then constructs one native reference and exports the existing plastic
weights and `u/w`. It performs no observations, gradient steps or paper orders.
Normal paper work may skip an overlapping invocation while this lock is held.
The collector uses its separate lock and continues independently.

The export uses an unscheduled Modal worker with two CPUs, 8 GiB, a 600-second
timeout and no retries. Its $0.1608936 maximum reservation uses 3× pricing for
non-preemptible compute and the existing 2× margin. It shares the $25 worker
ledger and 75% stop threshold. The reservation is an estimate, not a provider
invoice. The temporary app was stopped after its original call completed and the
captured files were verified.

The commands below describe the original capture workflow. In the working
directory with the saved completed receipt, `observe` reuses that result.
The persistent claim rejects another capture, even if the app is redeployed.

```sh
uv run --extra dev --extra cloud python -m pytest -q tests/test_fly_rate_checkpoint.py

PAPERLAB_FLY=1 PAPERLAB_UNIVERSE=1 \
  uv run --extra cloud modal deploy rate_checkpoint_cloud.py --env main
uv run --extra cloud python -m paperlab.fly_rate_checkpoint_cloud prepare \
  --out runs/rate-checkpoint-12/cloud
uv run --extra cloud python -m paperlab.fly_rate_checkpoint_cloud submit \
  --out runs/rate-checkpoint-12/cloud
uv run --extra cloud python -m paperlab.fly_rate_checkpoint_cloud observe \
  --out runs/rate-checkpoint-12/cloud
```

`submit` refuses an active shared worker or an existing capture attempt. An
observation timeout means repeat **observe** for the saved call; never resubmit.
The observer verifies ownership and budget before bounded, hash-checked reads.
It replays cohort selection and exposure from the downloaded snapshots and
compares exported `u/w` to the full checkpoints without constructing a local
brain. The native graph/configuration and complete weight export are checked in
the cloud. A built-in SDK timeout initially escaped the observer; both timeout
types now return `pending`. The original call remained live and was never
resubmitted. Its original image/source hashes and the subsequent observer-only
change are recorded separately. Synthetic tests are not trading evidence.

Repeat the additional array audit with the retained complete download and graph:

```sh
uv run python -m paperlab.fly_rate_checkpoint_audit \
  --root runs/rate-checkpoint-12/cloud --fly-data ../fly-data \
  --out runs/rate-checkpoint-12-array-audit-new.json
```

## Registered comparison rules

`paperlab/fly_rate_protocol.py` now defines four arms: pristine frozen, trained
frozen, trained online at eta 0.001, and trained online at eta 0.0001. All retain
both rate histories. Each arm starts with separate $1,000 paper capital per
phase: $250 per pool and $500 idle. Two-hour development and two-hour test phases
have 24 five-minute decision slots each. Existing execution fees, simulated
spread and slippage remain fixed.

The only candidate is the lower online rate. Development must have at least
18 eligible slots per pool and identical coverage across arms. Its equity must
strictly beat all three controls and cash plus $0.11111 of prorated hosting
cost. Selection is saved before any test condition; test data cannot reselect
the candidate. Every test arm is reported, including fees, turnover and missing
marks. Insufficient test coverage is inconclusive.

The runner, sealer, ledger replay and independent auditors are implemented in
separate study 12 modules. Its 91 passing checks cover changed rates, altered
cohorts, late registration, input and account corruption, fresh-process news
seeding, source drift, budget limits and interrupted captures. A saved selection
receipt must follow every development completion and precede every test claim.
The original studies and their recorded results remain unchanged.

The registration pins the completed capture and its source hashes, all
comparison arms, future windows, costs and selection rules. Newly
exported memory is still memory trained at eta 0.001; using eta 0.0001 afterward
tests a lower online learning rate, not retraining the starting memory at that
rate. No policy is promoted by exporting a checkpoint.

## Cloud collection and capture

The main app collects live quotes and news during the four-hour window. Once
the endpoint is present in its snapshot, a dedicated temporary worker seals
the inputs and replays them chronologically. This is a prospective replay of
fresh data; the experiment does not submit trades during collection. The
existing main paper trader continues separately.

The study worker wakes every five minutes, at minutes 2, 7, 12 and so on. It
uses the shared worker lease, so an overlapping main call or study call skips
instead of writing concurrently. Each complete pool/arm/phase runs once in a
new non-preemptible container with two CPUs and 8 GiB. The 600-second call
contains one seed-zero subprocess with at most 480 seconds for neural capture.
Sixteen complete conditions and independent audits follow collection; results
are not expected exactly at 10:00 UTC.

Every claim is committed before capture. An interrupted, failed or uncertain
claim halts further work and is never automatically retried. Both source
hashes and the registration must still match the cloud witness. All metadata
and capture calls share a $3 compute allowance within the existing $25 worker
ledger, 75% stop threshold and 2× margin. Non-preemptible CPU/RAM pricing is
accounted at 3×, with a $0.1608936 maximum reservation per call. These are
compute estimates; provider billing also includes other resources.

The initial image build failed before any study call: Modal serialized the
entrypoint into `/root`, while the registered files were uploaded to
`/opt/paperlab`. An explicit cloud project path fixed the import. Seven affected
runtime tests passed again, and the subsequent real cloud build verified the
entrypoint, pinned reference files and execution manifest. No registration had
been witnessed before this fix, and no neural capture was retried.

Inspect existing cloud state without creating a call:

```sh
uv run --extra cloud python -m paperlab.fly_rate_cloud status \
  --out runs/rate-market-status-12.json
```

Reproduce the checks without constructing or running a native brain locally:

```sh
PYTHONHASHSEED=0 uv run --extra dev --extra cloud python -m pytest -q \
  tests/test_fly_rate_protocol.py tests/test_fly_rate_pipeline.py \
  tests/test_fly_learning_scale_credit.py tests/test_fly_rate_schedule.py \
  tests/test_fly_rate_runtime.py tests/test_fly_rate_cloud.py
```

The original preparation and deployment commands were:

```sh
uv run --extra cloud python -m paperlab.fly_rate_cloud prepare \
  --start 1789279200 --out runs/rate-market-preflight-12/prepared.json
PAPERLAB_FLY=1 PAPERLAB_UNIVERSE=1 \
  uv run --extra cloud modal deploy rate_market_cloud.py --env main
```

`prepare` refuses an existing registration/reference or a start less than ten
minutes in the future. The committed registration is historical evidence, not
a command to start a new comparison. No result from this comparison will
automatically promote the lower rate to the main trader.

## Download, audit and inspect the completed comparison

`paperlab.fly_rate_results` is a read-only results workflow. It checks the exact
registration and source manifest, reattaches the saved owning calls, and
requires a matching completed call result before transferring a recording.
It reconstructs raw prices before downloading the larger neural arrays. Reads
have bounded deadlines; interrupted reads never become new model submissions.
Repeated downloads reuse verified files and reject changed evidence.

To wait for the existing cloud work, run the optional local observer:

```sh
uv run --extra cloud python -m paperlab.fly_rate_watch \
  --registration reports/fly-rate-market-registration-12.json \
  --out runs/rate-results-12
```

It checks every five minutes for up to six hours and emits changed milestones.
It stops when all recordings are captured, when a saved condition needs
inspection, or when the observer reaches its own deadline. Three consecutive
read timeouts also stop the observer; they do not establish a cloud failure.
It rejects duplicate watchers on the same output and never submits, restarts,
downloads large recordings or changes cloud work. `watch.json` records its last
check; use the actual process status to determine whether it is still running.
This observer runs on the laptop, while collection and model execution remain
in Modal. Closing the observer does not stop those cloud jobs. Thirteen tests
cover its waiting, timeout, ownership and attention boundaries.

When it reports `observer_ready`, download the verified recordings:

```sh
uv run --extra cloud python -m paperlab.fly_rate_results observe \
  --registration reports/fly-rate-market-registration-12.json \
  --out runs/rate-results-12 --download
```

Before capture completes, this reports the current phase and completed/downloaded
counts. The phase follows the registered clock; actual quote coverage is checked
from the sealed data. It does not wait for hours or start the cloud worker. Once all 16 calls,
artifacts and the cloud aggregate are present, it reports `downloaded_pending_audit`.

The following audit requires the retained graph data. It uses array reconstruction
only, with a fresh seed-zero process for timestamped news, and never constructs
or propagates a native brain locally:

```sh
PYTHONHASHSEED=0 uv run python -m paperlab.fly_rate_results audit \
  --registration reports/fly-rate-market-registration-12.json \
  --root runs/rate-results-12 --fly-data ../fly-data --out runs/rate-audit-12
```

The audit refuses partial comparisons or an existing output directory. It
rechecks receipts, call results, artifact hashes, selection timing, accounts,
news and all recorded learning bins, then checks each debugger projection.
It writes the final report only after every condition passes. Synthetic fixtures
stay labeled as validation, and missing test coverage stays inconclusive.

Open the resulting views and full-neuron recordings:

```sh
uv run --extra cloud python -m paperlab.debugger serve --backend modal \
  --out runs/rate-audit-12/views --port 8772
```

Visit <http://127.0.0.1:8772>. Viewing the recordings requires no new model run;
the Modal backend keeps any separately requested input assay off the laptop.
The complete-recording links use local hard links to the verified download.
This workflow prepares result inspection; it does not claim the pending study
has already produced audited trading results.

The [results preflight record](../reports/fly-rate-results-preflight-12.json)
preserves the 42 passing orchestration/transfer checks and the real read-only
arming-call verification. The 33 reader/audit checks passed again after switching
call observation to Modal's asynchronous API; the nine shared transfer checks
were unchanged. Mocked native-audit wiring remains labeled as a fixture.
