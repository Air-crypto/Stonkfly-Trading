# Frozen-checkpoint scheduling repair — September 14, 2026

The new `fly-paper-checkpoint-dispatch` app checks every minute and calls the
existing frozen evaluation worker. It prioritizes the newest checkpoint eligible
at the already sealed cutoff, then remaining controls, then other sealed
checkpoints newest first. It never ranks checkpoints using their results. For
`evaluation-1789402218`, the first priority is `solana-online-20260914-152719`.
Later-trained weights cannot be added to this older test cohort.

Previously, controls ran first, one policy was dispatched per fifteen-minute tick,
and training completion could remain unacknowledged until another training tick.
Together with the required training gap, this delivered about one evaluation per
hour even when the preceding evaluation took only minutes.

The added dispatcher reconciles a finished evaluation and dispatches its successor
in the same invocation. It asks the original training coordinator to acknowledge
completed calls promptly. If training is due, that coordinator still gets priority;
its accounting, cadence, archive and budget logic remain authoritative. Otherwise
an evaluation can start only with at least 1,250 seconds before the next training
eligibility, covering the unchanged 1,200-second worker timeout plus headroom.
The new scheduler does not predict shorter runtimes or risk overrunning training.
Normal completion-to-next-dispatch latency is up to one minute plus platform delay.

## What remains sealed

Every source hash in the existing evaluation plan was compared with the checkout:
all matched, with no differences. The new scheduler files live outside the sealed
worker source set and are never imported by that worker. The old evaluator app was
not redeployed. Plans, test tapes, policy membership, existing results and model
weights are not rewritten. A separate `dispatch-policy-v1.json` records the plan
hash and scheduling priority without altering the plan's policy order.

Both coordinators share the same atomic coordinator lease. Heavy evaluations still
share the existing exclusive worker lease. No attempts are overwritten. Missing
call IDs, uncertain dispatch results, operational failures, audit pauses, archive
pauses and budget stops remain fail-closed. The original coordinator still seals
future cohorts, selects future tapes and aggregates completed comparisons.

## Cost

The lightweight dispatcher uses at most 0.125 CPU and 512 MiB, with a sixty-second
call timeout. Its separate serialized `checkpoint-eval/dispatch-budget.json`
reserves worst-case call time and settles measured duration with the existing
conservative CPU/RAM rates and startup allowance. It has a $5 monthly allocation
within the existing $15 collector/overhead allowance, with the existing 25% safety
margin. Shared workers retain their $85 allocation. The total authorization stays
$100; these ledgers are estimates, not the provider invoice. If the dispatch
allowance is exhausted, acceleration stops and the original coordinator remains.

## Verification

Focused scheduler, frozen-evaluation and training tests passed: 70 tests. These
cover priority without seal changes, immediate successor dispatch, fresh training
acknowledgement, training precedence, full timeout headroom, active leases, paused
services, uncertain spawn outcomes, missing results, and separate budget handling.
The full seeded suite passed **1,169 tests with 26 skips** in 189.80 seconds.
Two expected warnings concern intentionally local Modal worker tests.

The deployed entrypoint successfully wrote its immutable priority manifest and
respected training priority. It asked the original training coordinator to launch
the already-due window `solana-online-20260914-183754`; the trained evaluation is
next after that window completes and the full timeout gap is available. The
manifest prioritizes `solana-online-20260914-152719`. Cloud verification again
confirmed every sealed worker source hash matches. See the
[durable control and manifest snapshot](checkpoint-dispatch-cloud-20260914.json).
This snapshot verifies scheduling and preservation, not a completed trained score.
