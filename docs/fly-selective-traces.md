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

## Next execution step

The native runner currently has no CLI or cloud-dispatch route. Its internal
entrypoint requires the completed, audited, non-synthetic study 11 evidence. After
that study finishes, integrate this assay with the existing budgeted worker and
one-submission receipt, run it once, and independently audit every condition before
opening the new paired views. Retain the $25 worker and $15 collector monthly
reservation caps and the 420-second inner bound; a partial run is incomplete and
must not silently restart.

This remains a historical mechanism test. It does not recompute account feedback,
fills or profit, and a changed BUY/HOLD is not evidence of a better trade. Any
promising mechanism needs a subsequent cost-aware prospective comparison.
