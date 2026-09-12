# Separate KC and dopamine trace resets

**Status on September 12, 2026: specified and checked against saved boundary
arrays; native trajectories have not been run.** The current study 11 deployment
and its pinned sources are unchanged. No selective-reset action or return is
claimed here.

The [first-drive reconstruction](fly-debugger.md#why-a-quiet-source-connection-can-still-update)
found that earlier KC activity accounts for the first differing weight updates
while the connected KC neurons are silent. Replacing KC history in one recorded
bin reproduces the both-traces-reset weights; replacing DAN history does not.
That calculation holds firing fixed. It cannot tell us what separate resets do
to later firing, memory, or decisions.

The [new protocol](../reports/fly-selective-trace-protocol-01.json) tests that
question with the complete native graph: 166,700 neurons, 25,582,938 edges, and
7,835 plastic connections. It uses the same three historical market images and
initial trained memory as the completed credit-reset assay.

| Boundary before images 2 and 3 | KC trace | DAN trace | Role |
|---|---|---|---|
| Carry | Retained | Retained | Reproduce prior control |
| Reset both | Zeroed | Zeroed | Reproduce prior intervention |
| Reset KC only | Zeroed | Retained | New intervention |
| Reset DAN only | Retained | Zeroed | New intervention |

Each row is tested with learning plus recorded pulses, learning without injected
pulses, and frozen memory plus recorded pulses: **12 conditions, 36 observations,
and 1,800 recorded ten-millisecond bins**. Those are planned counts. All six prior
conditions must reproduce their complete count, sampled-voltage, weight, memory,
and rate recordings before either new intervention runs. Every first image starts
from identical dynamics. The frozen selective conditions must retain every
recorded count and voltage sample as well as unchanged connection memory.

The runner saves all boundary arrays and full-network recordings. Its independent
auditor reconstructs the plasticity rule and fixed decoder, checks continuity
between observations, and compares each new condition against both controls.
The output is designed for the existing paired debugger, including memory-drive
curves and first-difference navigation. End-to-end native execution, its audit,
and browser verification with the new recordings are still pending.

## What has been checked

The [preflight record](../reports/fly-selective-trace-preflight-01.json) verifies
all 84 parent artifacts, then applies all four operations to copies of six
recorded boundary states. All 24 checks preserve the clock, connection memory,
and full weight array; only the specified rate fields change. All 21 native
dynamic fields are checked. This is saved-array manipulation, with **zero new
neural observations and zero cloud submissions**.

The [validation record](../reports/fly-selective-trace-validation-01.json) also
covers tampered arrays with recomputed hashes, changes to untouched fields or
nonplastic weights, missing state, dtype changes, first-image behavior, and
refusal to execute before the actual study 11 has completed and been audited.
These checks do not prove that the new intervention improves learning.

## Check what the debugger draws

The [projection audit](../reports/fly-view-projection-01.json) checks the six
existing carry/both-reset views against verified graph data and their full
recordings: 18 observations and 900 bins. Every selected neuron, displayed
connection, and plotted series passed. This includes nonplastic connection
endpoints and baseline weights, every edge between selected neurons, population
and whole-brain spike curves, trace averages, changed-edge counts, and weight
distance from both the observation start and pristine memory. It also checks
per-neuron spike totals across the three observations.

The new selective auditor now requires these checks. Its earlier selected-memory
checks covered spikes and plastic edges but did not cover every aggregate curve
or nonplastic edge. This was a verification gap; the stronger checks found no
incorrect values in the six existing views. Float32 norm reductions receive a
four-ULP rounding allowance; underlying displayed samples must match exactly.

The [regression record](../reports/fly-view-projection-validation-01.json) includes
wrong-curve and wrong-connection fixtures that previously passed the narrower
memory-enrichment helper, plus tests for altered images and retimed spikes whose
whole-observation count totals remain equal. These are deliberate test mutations,
not observed model failures. The graph check loads data without constructing a
native simulator or changing the active cloud deployment.

```sh
uv run --extra dev python -m paperlab.fly_view_projection \
  --audit reports/fly-credit-reset-audit-01.json \
  --artifacts runs/credit-reset-01/cloud/artifacts \
  --fly-data data/fly \
  --out runs/view-projection-new

# Optional retained-data regression; requires the prepared graph and recordings.
FLY_CREDIT_RESET_RECORDINGS="$PWD/runs/credit-reset-01/cloud" \
FLY_TRACE_DATA="$PWD/data/fly" \
  uv run --extra dev python -m pytest -q tests/test_fly_view_projection.py
```

## Reproduce the preflight

This requires the previously downloaded credit-reset cloud artifacts and original
payload from the [parent assay workflow](fly-debugger.md). The artifacts are
private local recordings; a fresh clone does not contain them. This command does
not construct the native brain or submit a Modal call. Choose a new output path
each time; earlier evidence is never overwritten.

```sh
uv run --extra dev python -m paperlab.fly_selective_trace \
  --parent-payload runs/credit-reset-01/payload.json \
  --parent-audit reports/fly-credit-reset-audit-01.json \
  --reference-recordings runs/credit-reset-01/cloud/artifacts \
  --out runs/selective-trace-preflight-new

# Set this only when the retained recordings are present; otherwise that one
# data-dependent test skips explicitly. No test below propagates the graph.
FLY_CREDIT_RESET_RECORDINGS="$PWD/runs/credit-reset-01/cloud" \
  uv run --extra dev python -m pytest -q tests/test_fly_selective_trace.py
```

## Carry the completed study evidence

The local observer produces the study 11 audits, while the next native assay will
run in Modal. `paperlab.fly_study_evidence_bundle` provides that handoff. It copies
exactly 51 JSON files: the report, sealed plan and price audit, plus all 16 chunk
summaries, independent audits and completed-call receipts. The original bytes
and their SHA-256 hashes are preserved. Full recordings and the market database
stay with the original audit evidence; this package does not replace their audit.

Packing validates a private copy through the existing study completion gate.
Unpacking requires the bundle hash recorded at packing, validates that same gate
before creating the destination, and checks the installed evidence again. The
gate requires the actual study 11 registration, all completed audits and an ended
window; synthetic fixtures cannot pass. A completed negative result can pass
this handoff, since it authorizes a later mechanism test, not a policy promotion.

**No real completion bundle exists yet.** Study 11 is still collecting. These
commands become usable only after the observer has produced its final audited
report. They perform local file operations and do not upload, deploy or submit a
Modal job:

```sh
uv run --extra dev python -m paperlab.fly_study_evidence_bundle pack \
  --study runs/online-cloud-11 \
  --out runs/study-11-completed-evidence.json

# Retain the bundle_sha256 printed above as the expected transfer identity.
uv run --extra dev python -m paperlab.fly_study_evidence_bundle unpack \
  --bundle runs/study-11-completed-evidence.json \
  --sha256 '<bundle_sha256 printed by pack>' \
  --out runs/study-11-restored-evidence
```

Both commands refuse existing outputs. The file allowlist rejects extra paths,
missing files and duplicate JSON keys; source files and directories cannot be
symlinks. Limits are 32 MiB per decoded file, 128 MiB total decoded evidence and
192 MiB for the bundle. If final evidence exceeds a limit, transport stops; it
does not omit or truncate any audit.

The transport regression uses explicitly synthetic study fixtures with only the
release decision mocked. It still checks all 16 chunk ledgers and audit/receipt
relationships, compares all 51 restored files byte for byte, and rejects altered
hashes, rehashed incomplete reports, path injection and existing destinations.
Separate tests retain the production gate and confirm that synthetic evidence
cannot be packed or unpacked. No native model is constructed.

```sh
uv run --extra dev python -m pytest -q tests/test_fly_study_evidence_bundle.py
```

## Prepared cloud bridge

`paperlab.fly_selective_trace_cloud` now prepares the completed-study bundle,
submits through the existing worker, resumes its saved call, downloads the full
recordings, runs the independent auditor and writes all twelve comparison views.
**This bridge is not connected to the deployed worker yet.** Its activation must
wait for study 11 capture and independent audits; none of that study's 56 pinned
execution files or the cloud deployment changed while preparing this bridge.

The local receipt distinguishes preparation from paid submission. An interrupted
bundle upload can resume using the same content hash. Before submitting, the
client checks the worker lease, running inputs and backlog, and the assay's
existing remote claim. It records `submitting` before the RPC; a lost response
at that point cannot cause a second submission. Later invocations with a saved
call ID only observe that call. A 50-second observation timeout leaves the call
pending, rather than restarting it.

The future worker adapter validates the same bundle and request source hashes,
checks the original 84 parent artifacts, and commits a persistent claim before
constructing the native model. That claim is shared across output run IDs and
remains after a failure or timeout. All 175 output files, including twelve sets
of three full recordings, are hashed before transport. Downloads resume at the
file level and check both existing and newly downloaded bytes. Expanded views
are installed only after the independent recording audit passes. These checks
do not replace the audit or prove improved trading performance.

The transport tests use a labeled synthetic study fixture, with only its release
decision mocked while the actual study evidence consistency checks still run.
Neural capture and transport responses are simulated in bridge tests; separate
tests retain the production release gate and reject incomplete and synthetic
studies before native execution or submission. No test starts a cloud call.
The [validation record](../reports/fly-selective-cloud-bridge-validation-01.json)
records 83 passing checks across the bridge and related modules, with one optional
retained-recording check skipped; native execution and browser checks of its new
recordings remain pending.

```sh
uv run --extra dev python -m pytest -q \
  tests/test_fly_selective_trace_cloud.py \
  tests/test_fly_selective_trace.py tests/test_fly_study_evidence_bundle.py
```

## Next execution step

After the actual study 11 finishes and passes its independent audits, connect
the `selective_plan` request in `cloud_debug.validate_request` to the new bridge's
validator and route execution to `run_request`. Supply the actual Modal call ID,
input ID and volume commit callback from within the existing worker lease and
budget reservation. Package `fly-first-drive-01.json` and the unchanged study 11
registration under `/opt/paperlab/reports/`, where the native runner's validation
expects them. Test this route and deployment before submission. Retain the $25
worker and $15 collector monthly reservation caps and the 420-second inner bound.
These activation edits are deliberately pending; the current deployed worker
does not accept `selective_plan`.

Once that activation is verified, use a new output directory for the single
submission and the same directory for every subsequent observation:

```sh
uv run python -m paperlab.fly_selective_trace_cloud \
  --payload runs/selective-trace-preflight-01/payload.json \
  --reference-recordings runs/credit-reset-01/cloud/artifacts \
  --completed-study runs/online-cloud-11 \
  --fly-data data/fly --out runs/selective-trace-01/cloud
```

The command packs and verifies the study evidence itself; no manual bundle
upload is needed. Successful downloads remain in `artifacts/`, the independent
audit in `audit.json`, and each named condition receives `view.json`, `report.json`
and `remote.json` for the debugger. A partial native run remains incomplete and
must not silently restart.

This remains a historical mechanism test. It does not recompute account feedback,
fills or profit, and a changed BUY/HOLD is not evidence of a better trade. Any
promising mechanism needs a subsequent cost-aware prospective comparison.
