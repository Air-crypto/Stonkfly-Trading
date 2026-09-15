# Prospective evaluation follow-up

On September 15, the user requested a later unseen evaluation of the newer
checkpoints. The previous comparison (`evaluation-1789430835`, 23 policies)
remains archived. Its reduced losses are a hypothesis to test again, not a
profitability claim.

## Deployed change

The existing daily coordinator already supports chronological future tapes.
Added an explicit `request_prospective` operator entrypoint to advance one daily
cutoff under the evaluation coordinator and shared worker leases. It checks the
expected completed batch, preserves existing cohorts, records the old schedule,
and refuses paused/error/budget states. Duplicate requests cannot replace a
sealed cohort. The evaluation algorithm and its sealed source fingerprint are
unchanged.

New plans explicitly preselect the latest completed checkpoint as the primary
comparison, independent of returns. All retained eligible checkpoints and the
three controls remain included. The first completed episode starting strictly
after the cutoff supplies the shared tape. Later training checkpoints are not
added to that cohort. Future daily cohorts run serially after prior completion,
subject to training gaps and the same budget.

## Cloud evidence

- Deployment: `fly-paper-checkpoint-eval`, `checkpoint_runtime_cloud.py`.
- Request call: `fc-01M2KCVF70V4RQJPB60YGASKNH`, completed successfully.
- New batch: `evaluation-1789504902`.
- Cutoff: **2026-09-15 20:41:42.926 UTC** (15:41:42 Chicago).
- Membership: **37 retained checkpoints plus 3 controls**, 40 policies.
- Primary: `solana-online-20260915-194714`, training ended 20:02:34 UTC.
- Verified at 20:42:45 UTC: **waiting for unseen market window**, tape unset.
- Next training scheduled for 20:47:14 UTC; actual start/completion must still
  be observed before assigning its tape. No future result is claimed here.
- Next daily cutoff eligibility: September 16 00:00 UTC; the active cohort
  completes first.
- Preparation estimate: **$0.00979**; shared worker ledger **$20.87387**.
  These are estimates, not provider billing. The $100 authorization is unchanged.

The companion JSON preserves the plan, checkpoint and source hashes, preparation
record and hash of the previous 23-policy comparison. Assertions verified all
checkpoint training ends precede the cutoff and the deployed sealed source
matches the local evaluator.

## Validation and interpretation

Targeted regression tests: **66 passed**. Tests cover cutoff overlap, later
checkpoint exclusion, unchanged prior artifacts, duplicate requests, protected
pauses, stale request rejection and the actual cloud entrypoint's lease order.

The initial full-suite invocation omitted the required `PYTHONHASHSEED=0` and
hit the historical replay fixtures' deterministic-process guard. It was stopped
and rerun with the setting documented in README and CI.
The corrected full run passed: **1,215 passed, 26 skipped**, with four expected
warnings from mocked local Modal entrypoint tests.

Scoring remains paper-only, frozen and cost-aware, at common recorded attention
opportunities. Each policy gets a fresh $1,000 account. It does not independently
test universe selection or establish executable live returns. Repeated later
windows must outperform cash and untrained controls after costs before claiming
an improvement that generalizes; continued training alone does not imply profits.
