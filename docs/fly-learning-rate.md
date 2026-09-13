# Retain both traces and reduce update strength

**The single native call is running as of 04:52 UTC, September 13, 2026.**
The [execution record](../reports/fly-learning-scale-execution-01.json) verifies
its durable claim and active worker. Capture completion and the independent
audit remain pending. The [registered protocol](../reports/fly-learning-scale-protocol-01.json)
tests `eta = 0.0001` against the original `0.001`, retaining both KC and DAN
rate histories. The [memory reconstruction](fly-selective-traces.md#why-the-single-resets-move-weights-more)
motivates this choice: removing just one history leaves a large one-sided
earlier-image contribution. Reducing eta scales both terms together.

The full recurrent network can change its firing at the lower rate, so its
observed updates need not shrink by exactly tenfold. This will be measured from
new native trajectories, rather than inferred by rescaling the previous curves.
The first image may already differ; only the initial memory and dynamics are
required to match. Smaller updates do not by themselves establish better learning.

| Order | Learning | Injected pulses | Eta | Role |
| --- | --- | --- | ---: | --- |
| 1 | Online | Recorded | 0.001 | Original carry control |
| 2 | Online | None | 0.001 | Original carry control |
| 3 | Frozen | Recorded | 0.001, inactive | Original frozen control |
| 4 | Online | Recorded | 0.0001 | Lower-rate candidate |
| 5 | Online | None | 0.0001 | Lower-rate candidate |

Every condition uses the original trained memory, complete 166,700-neuron graph,
and same three historical images. Both rate histories carry between images;
there are no activity resets, new feedback calculations or accounts. The first
three conditions must reproduce every recorded count, sampled voltage, rate,
weight and `u/w` value before either candidate runs. Fifteen observations and
750 ten-millisecond bins are planned.

The independent auditor reconstructs the rule at each arm's declared eta,
checks boundary continuity and all full-graph weights, verifies the fixed
decoder and displayed curves, then compares both candidates with their matched
online controls. All original execution modules remain unchanged.

The [preflight record](../reports/fly-learning-scale-preflight-01.json) records
44 tests for eta propagation, frozen memory, wrong-rate rejection, protected
submission, bounded downloads and cloud budget limits. Nine retained original
control observations also reproduce through the new pure-rule auditor. These
are local validation results, with zero new neural propagation or cloud calls.

## One bounded cloud call

`scale_cloud.py` defines `fly-paper-scale-01`, with no schedule, one 2-CPU/8-GiB
non-preemptible container and a 360-second timeout. Capture has a 240-second
inner deadline. The worker shares the existing lease and $25 ledger, including
its 75% stop threshold. At 3× provider pricing and the existing 2× margin, the
maximum call reservation is **$0.0975912**. This is a conservative compute
estimate, not a provider invoice or a separate spending allowance.

Deployment and capture require the completed study 11 gate and previous audited
evidence. Inputs are installed in the image or transferred as the same exact
completed-study bundle. The worker persists a claim before constructing the
native brain; a failed or uncertain call cannot silently start again.

```sh
PAPERLAB_FLY=1 PAPERLAB_UNIVERSE=1 \
  uv run --extra cloud modal deploy scale_cloud.py --env main

# Requires the retained parent payload and full recordings.
# Reuse this output for every later observation of the same call.
uv run python -m paperlab.fly_learning_scale_cloud \
  --payload runs/learning-scale-preflight-01/payload.json \
  --reference-recordings runs/credit-reset-01/cloud/artifacts \
  --completed-study runs/online-complete-audit-11 \
  --fly-data data/fly --out runs/learning-scale-01/cloud
```

The historical assay does not select a policy or report profit. Its inputs
cannot serve as fresh held-out performance data. Any candidate must subsequently
face a prospective comparison that includes fees, spread, slippage and hosting.
