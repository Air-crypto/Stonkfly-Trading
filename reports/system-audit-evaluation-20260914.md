# Checkpoint evaluation audit — 2026-09-14

The retained-checkpoint design is a useful starting point for a prospective comparison, but the first implementation needed corrections before its results could be interpreted as evidence of deployment behavior. The first sealed v1 batch had not run any policy at the audit pause. It must remain archived and be superseded by a new v2 plan with a fresh cutoff; changing the old plan or backfilling new timing fields into old tapes would invalidate that claim.

## Findings and corrections

| Finding | Effect | Correction |
|---|---|---|
| Replay consumed events through the ledger row's end, after live inference, and issued its order immediately. | The replay saw a different market snapshot and did not retain the measured inference delay. | Replay consumes only the recorded `observation_at` and preserves the source decision's later `issued` timestamp for every policy. |
| Network receipt timestamps precede parsing and application to the live feed. | A receipt before a snapshot can still be unapplied at that snapshot. Wall-clock filtering alone does not recreate the information actually observed. | Live rows record the last atomically applied SQLite event cursor; replay also caps its prefix by that cursor, in database application order. |
| FX failures set the live quote conversion to zero, while the FX archive records only successful fetches. | Replay could reuse an earlier fresh FX value during a period when the live trader rejected quotes. | Each new row records the FX availability state; replay respects failures and verifies successful states against the contemporaneous FX log. |
| Replay accumulated histories for every quoted token, irrespective of recorded admission and retirement. | The same selected mint could have different input histories from the common live opportunity. | Replay follows recorded active membership, preserves parked histories as v2 training does, and clears retired execution intents. |
| The shared quote helper treated entry eligibility as position observability and exit eligibility. | An entry restriction could block risk reduction or write an observable holding down to zero. | The sealed v2 quote protocol separates observation, entry and exit. The actual later-receipt trade direction selects its guard. Capacity-limited next-fill sensitivity is reported alongside the full indicative mark. |
| Registry assumed every completed window saved new native/head files. | A healthy window with no native observations reuses its parent's files and could halt future registration with a missing-file exception. | Registry retains the reused checkpoint paths and hashes, plus the immutable continuation-state record describing that reuse. Missing files after actual training still fail closed. |
| Source hashes covered a short list of Python files. | Changes to native kernels, settings, schemas or imported helpers could alter later policy calls without tripping the seal. | The source fingerprint covers experiment Python, native Python/C++/headers, schemas/lock JSON and image recipes. Added source modules are detected too. |
| Skipped opportunities and frozen-policy inputs were not recorded in detail. | A low-coverage or behaviorally different replay could appear to be a complete comparison. | Results expose offered/issued/used opportunity counts and reasons for exclusions; decision rows retain Q values, input features and frozen-state diagnostics. |

Terminal learning-settlement rows are handled as metadata. They reuse the last observed market mark and do not create another receipt, decision or fill. Native transient state resets at switches and gaps; learned synaptic weights remain frozen. Exact native and Q weight equality is checked independently of the per-step diagnostic claims.

## What the experiment can answer

For checkpoints trained before the newly sealed cutoff, all policies see the same subsequent market tape and recorded token-attention opportunities. They begin with independent $1,000 accounts and the same sizing, fees, latency, availability and execution rules. The untrained reference uses pristine native weights and a seed-13 random head; cash and always-long references have their documented distinct semantics. No evaluation calls the optimizer or native plasticity.

This estimates **conditional buy/sell behavior at those recorded opportunities**. The opportunity set itself came from the live learner's active-universe and scheduling process. It is not an independent universe-selection evaluation. Receipt-derived prices remain indicative reserve observations, not executable exchange quotes.

The comparison also changes the policy from online learning/exploration to frozen greedy deployment. In particular, the input encoding includes a native weight-change diagnostic that is necessarily zero in frozen mode. This is an explicit feature-distribution limitation, not a claim of identical online and frozen inputs. A market-only ablation and matched Q-only/native-only controls remain necessary before crediting any improvement specifically to the fly circuit.

One 15-minute future tape is a pilot comparison, not a reliable estimate of profitability. Multiple checkpoints share the same tape and are not independent market samples. Selecting the best of many checkpoints on it requires a later untouched confirmation window. Additional nonoverlapping windows, coverage reporting, costs, concentration, capacity sensitivity and uncertainty across windows are needed before a deployment decision.

## Verification and operational boundaries

At implementation time, the focused evaluator/quote suite passed **22 tests**, including **9 evaluator test cases**. The evaluator cases cover immutable-file mutation, reused checkpoints, future-window eligibility, native/schema/new-module fingerprint changes, delayed fills, receipt-prefix timing, FX failures, terminal settlement behavior, accounting reconstruction, frozen arrays, invalid chronology and invalid FX order. Local tests use lightweight native fixtures; they do not establish that the full native cloud entrypoint runs successfully.

A separate unscheduled synthetic cloud smoke is used to verify pristine and retained native checkpoint restoration, six real native forward passes per policy, frozen weights and ledger reconstruction. It is explicitly a mechanics check and must not be reported as market performance. Its observed result and the final deployed source revision are recorded by the coordinating audit after completion.

The evaluator keeps the existing shared worker lease, live-training priority, bounded non-retrying calls and the unchanged $100 monthly authorization. Large all-checkpoint cohorts may require multiple days; the daily cutoff is an eligibility schedule, not a promise that a complete comparison finishes daily. An idle or completed scheduler, finite gradients and a successful audit do not establish profitable learning.
