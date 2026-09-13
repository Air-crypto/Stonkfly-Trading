# Fresh market comparison of learning rates

The [lower-rate neural assay](fly-learning-rate.md) reduced later weight movement
but introduced an extra BUY. It did not establish better trading. The next
comparison must measure fresh account outcomes and turnover after execution
costs, with a development decision saved before the test phase.

The [study 12 registration](../reports/fly-rate-market-registration-12.json) now
fixes development at **06:00–08:00 UTC** and test at **08:00–10:00 UTC** on
September 13, 2026. Cloud deployment and a witness before 06:00 are still pending.
The [market preflight record](../reports/fly-rate-market-preflight-12.json)
contains the 91 passing checks, execution source hashes and observed shared
budget. No study 12 market observations have been simulated yet.

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
