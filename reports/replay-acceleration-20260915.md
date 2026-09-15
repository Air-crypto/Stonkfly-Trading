# Replay and native cadence experiment — September 15, 2026 UTC

Implemented two isolated candidates: an experience-replay Q head with a frozen target network, and a runner accepting a 2.5-second observation interval. The recurring hourly baseline and sealed checkpoint evaluator remain unchanged. This experiment does not establish better trading.

## Protocol

- Same 802-parameter Q head and initial checkpoint in both replay arms; 1 or 4 optimizer steps per newly credited transition, batch size 32, capacity 10,000, target copy every 100 optimizer steps. Four updates reuse experience; they do not create four independent market observations.
- Save/restore includes optimizer state, replay buffer, target parameters, both RNGs and separate transition/optimizer counters. Recorded economic rewards are credited once, including terminal settlement.
- Five chronological training episodes precede one development episode. All registered test tapes and episodes crossing the active evaluation cutoff are excluded. Inputs and initial checkpoints are hash-pinned before training. Development computes TD error against the same frozen initial target and performs no updates.
- This logged-transition comparison uses features from the original online fly states. It does not replay candidate trading policies or measure candidate P&L. Both arms have replay and target networks; the 1× arm is not the previous no-replay baseline. Target synchronization happens four times as often per new transition in the 4× arm.
- Cloud-only native benchmark: three independent 120-second windows, 24 synthetic token streams, identical restored full-fly/head weights, fresh $1,000 paper accounts, and arms (1×, 5s), (4×, 5s), (4×, 2.5s). Synthetic returns are accounting diagnostics only.
- The faster cadence samples histories more frequently, reducing the 12-observation warmup from about 55 to 27.5 seconds; it also changes the time span represented by lagged features. This is a cadence experiment, not a pure parallel-compute comparison. One shared fly still serves tokens sequentially.

## Execution safeguards

The unscheduled `fly-paper-acceleration-study` app uses the shared worker lease and budget ledger, checks training pauses, and starts only with a 590-second gap before the next training window. Its hard timeout is 540 seconds. Reservation covers the 3× non-preemptible CPU/memory rate, with the existing conservative budget margin; failed runs retain their reservation. No native fly is constructed locally. No baseline accounts, checkpoint files, protected tapes, or evaluator sources are replaced.

The first pilot retained successful replay results but failed at the synthetic-feed clock interface before native observations. The failed artifact remains immutable. A second invocation returned `writer_busy` without starting a study. The third was cancelled after the real clock exposed a snapshot/heartbeat freshness mismatch: zero observable quotes, so its outputs are invalid for throughput. Both failed/cancelled reservations remain charged. A fourth invocation was also skipped by the lease guard without a reservation. The fifth uses one atomic snapshot/heartbeat timestamp and refuses to accept a result without native observations and optimizer steps. The corrected feed now passes the entire runner with a fake fly locally, including fills and independent reward reconciliation.

## Validation

Full suite before the feed-interface fix: **1,179 passed, 26 skipped**. Focused tests after the fix and the explicit cloud-only guard: **13 passed**. Tests cover exact optimizer counts, reward conservation, frozen/masked targets, deterministic checkpoint continuation, bounded replay, test-tape exclusion, unchanged sealed source fingerprints, and the actual synthetic-feed runner interface. A clean archive of implementation commit `a1e6281` passed **1,182 tests, 26 skipped**. The timestamp correction also passed the 13 focused tests. The final runtime commit `72ff5cd` also passed **1,182 tests, 26 skipped** from a clean Git archive. All four deployed study source hashes match the committed files. The cloud benchmark completed successfully, with finite loss/gradient metrics, nonzero recorded native weight changes on every observation, and exact reward reconciliation in all three arms.

## Measured results

Successful run: `acceleration-20260915-05`, Modal call `fc-01M2H8S3J8V7782WT949E38DKJ`. Full values, source/input hashes, gradients and weight deltas are in [the evidence JSON](replay-acceleration-20260915.json).

| Replay arm | Training transitions | New optimizer steps | Training time | Later-development TD loss |
|---|---:|---:|---:|---:|
| 1× | 499 | 499 | 0.553s | 0.00978862 |
| 4× | 499 | 1,996 | 1.771s | 0.01026118 |

The 4× arm's final training minibatch loss was 0.003576, versus 0.011994 for 1×; these are different sampled minibatches. Its error on the 108 development transitions was **4.83% higher** against the common frozen target. This single small development split does not support promoting 4× as a better policy. Both arms replayed the same historical rewards; replay counts are not new independent data, and losses are not dollar returns. These experimental optimizer counts do not advance the recurring baseline checkpoint.

| Native arm (120s requested) | Fly observations | Distinct tokens inferred | Credited transitions | Optimizer steps | Fills | Ending synthetic equity |
|---|---:|---:|---:|---:|---:|---:|
| 1× / 5s | 7 | 3 | 7 | 7 | 0 | $1000.00 |
| 4× / 5s | 7 | 3 | 7 | 28 | 0 | $1000.00 |
| 4× / 2.5s | 24 | 19 | 24 | 96 | 10 | $983.74 |

All native arms used the full fly on Modal. Median observed spacing was 5.001s, 5.000s and **2.501s**; median native compute took 1.920s, 1.892s and 1.257s, respectively. The faster arm had one step overrun. Maximum pre-clipping head gradient norms were 0.2725, 0.1990 and 0.4206, all finite. Mean optimizer losses were 0.004819, 0.003595 and 0.008968; the faster arm generated a different transition sequence, so these are not matched learning-quality scores.

First native observation occurred about 56s into the five-second arms and 29s into the faster arm. The 24-versus-7 count therefore combines a faster cadence with earlier warmup; median spacing supports roughly twice the observation frequency, not a 3.4× steady-state hardware improvement. Final account settlement adds Q updates after inference stops. In the faster arm, 19 of 24 credits occurred at terminal settlement; higher token turnover can reduce the uninterrupted same-token feedback available to native plasticity. The Q head still receives all audited economic credit. In this faster arm it received −$16.2557 while native feedback received −$4.8767; this remaining credit gap is another reason not to equate more native weight changes with better economic learning.

Synthetic reserves were constant. The faster arm made ten fills and ended $16.26 below cash from simulated trading frictions; the five-second arms did not trade. This verifies accounting under changed cadence, not a profitable market strategy. Neither candidate was promoted into the recurring hourly trainer. A separate pre-registered, later-market policy comparison remains necessary before making that choice.

The completed run's conservative compute estimate was **$0.109**; retained failed/cancelled pilot reservations add **$0.520**, for **$0.629** across these studies. The shared worker ledger was **$14.85** after completion. This is an internal estimate, not the provider bill; other services remain separately budgeted. The user's $100 monthly authorization and existing pause thresholds are unchanged. At export, recurring training remained enabled and hourly, with its next window scheduled for 01:02:18 UTC.

## Reproduction

Deploy `acceleration_study_cloud.py` with the existing Modal environment, then spawn the deployed `worker` with a new immutable ID such as `acceleration-YYYYMMDD-NN`. It can return a pause, busy, or insufficient-gap status without running. Do not erase a shared lease or bypass the budget to force a run. Artifacts are stored under `/state/acceleration-studies/<id>/`; each replay arm retains per-update loss/gradient metrics and its checkpoint, and each native arm retains decisions, fills, neural diagnostics and final checkpoints.
