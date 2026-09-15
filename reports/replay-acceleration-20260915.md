# Replay and native cadence experiment — September 15, 2026 UTC

Implemented two isolated candidates: an experience-replay Q head with a frozen target network, and a runner accepting a 2.5-second observation interval. The recurring hourly baseline and sealed checkpoint evaluator remain unchanged. This experiment does not establish better trading.

## Protocol

- Same 802-parameter Q head and initial checkpoint in both replay arms; 1 or 4 optimizer steps per newly credited transition, batch size 32, capacity 10,000, target copy every 100 optimizer steps. Four updates reuse experience; they do not create four independent market observations.
- Save/restore includes optimizer state, replay buffer, target parameters, both RNGs and separate transition/optimizer counters. Recorded economic rewards are credited once, including terminal settlement.
- Five chronological training episodes precede one development episode. All registered test tapes and episodes crossing the active evaluation cutoff are excluded. Inputs and initial checkpoints are hash-pinned before training. Development computes TD error against the same frozen initial target and performs no updates.
- This logged-transition comparison uses features from the original online fly states. It does not replay candidate trading policies or measure candidate P&L. Both arms have replay and target networks; the 1× arm is not the previous no-replay baseline. Target synchronization happens four times as often per new transition in the 4× arm.
- Cloud-only native benchmark: three independent 150-second windows, 24 synthetic token streams, identical restored full-fly/head weights, fresh $1,000 paper accounts, and arms (1×, 5s), (4×, 5s), (4×, 2.5s). Synthetic returns are accounting diagnostics only.
- The faster cadence samples histories more frequently, reducing the 12-observation warmup from about 55 to 27.5 seconds; it also changes the time span represented by lagged features. This is a cadence experiment, not a pure parallel-compute comparison. One shared fly still serves tokens sequentially.

## Execution safeguards

The unscheduled `fly-paper-acceleration-study` app uses the shared worker lease and budget ledger, checks training pauses, and starts only with a 700-second gap before the next training window. Its hard timeout is 650 seconds. Reservation covers the 3× non-preemptible CPU/memory rate, with the existing conservative budget margin; failed runs retain their reservation. No native fly is constructed locally. No baseline accounts, checkpoint files, protected tapes, or evaluator sources are replaced.

The first pilot retained successful replay results but failed at the synthetic-feed clock interface before native observations. The failed artifact remains immutable. A second invocation returned `writer_busy` without starting a study. The third was cancelled after the real clock exposed a snapshot/heartbeat freshness mismatch: zero observable quotes, so its outputs are invalid for throughput. Both failed/cancelled reservations remain charged. The fourth uses one atomic snapshot/heartbeat timestamp and refuses to accept a result without native observations and optimizer steps. The corrected feed now passes the entire runner with a fake fly locally, including fills and independent reward reconciliation.

## Validation

Full suite before the feed-interface fix: **1,179 passed, 26 skipped**. Focused tests after the fix and the explicit cloud-only guard: **13 passed**. Tests cover exact optimizer counts, reward conservation, frozen/masked targets, deterministic checkpoint continuation, bounded replay, test-tape exclusion, unchanged sealed source fingerprints, and the actual synthetic-feed runner interface. Native throughput results are recorded separately after the cloud benchmark completes.

## Reproduction

Deploy `acceleration_study_cloud.py` with the existing Modal environment, then spawn the deployed `worker` with a new immutable ID such as `acceleration-YYYYMMDD-NN`. It can return a pause, busy, or insufficient-gap status without running. Do not erase a shared lease or bypass the budget to force a run. Artifacts are stored under `/state/acceleration-studies/<id>/`; each replay arm retains per-update loss/gradient metrics and its checkpoint, and each native arm retains decisions, fills, neural diagnostics and final checkpoints.
