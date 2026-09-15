# Training archive and evaluation preparation recovery — September 15, 2026

Two independent scheduler failures stopped progress. The original cloud log at 2026-09-14 23:47:17 Chicago records `storage_cap_requires_review`: preserved Solana run data had reached **9,083,388,893 bytes**, exceeding the old 8 GiB cap. The next disabled scheduler tick overwrote that specific status with `disabled`, hiding the reason. There was no budget exhaustion and no new failed training episode.

The evaluation coordinator call `fc-01M2HBBMX9AVEX53YKFYS64YYH` independently hit its 120-second timeout. Its coordinator lease survived termination and blocked both the original coordinator and the minute dispatcher. No policy in the new 23-policy cohort had been scored, and the plan still had no sealed tape. The retained call result establishes the timeout; the exact internal operation at termination was not available in its fetched logs.

## Repair

- Versioned scheduling module `training_service_v2.py` preserves the original learning runtime and the sealed evaluator source hashes. It retains the original pause reason and timestamp on later disabled ticks.
- Raise the bounded Solana training archive allowance from 8 to **64 GiB**. Retain all original checkpoints, event archives, decisions, losses and sealed plans. No evidence was deleted. At the published $0.09/GiB-month rate, the full 64 GiB training allowance would be $5.76/month before any included allowance; this is a storage estimate, not the complete provider bill. See [Modal pricing](https://modal.com/pricing). Existing compute guards and the $100 total authorization remain unchanged.
- Recover only the verified idle storage pause, preserving the exact parent and original control in an immutable recovery record. Other error, budget, pending-call and audit pauses remain blocking. Reserve a bounded recovery gap for evaluation preparation, then return to hourly training.
- Deploy `checkpoint_runtime_cloud.py` to the existing evaluator app. Its policy worker delegates to the **unchanged** sealed `_evaluate` implementation. Preparation is separated from scoring and receives 2 CPUs, 8 GiB, a 600-second timeout and a shared-budget reservation. It records durable phase progress, leaves a sealed cohort unchanged, and avoids rehashing ready batches on every tick.
- Recovery checks the exact old owner through Modal and accepts only a confirmed `FunctionTimeoutError`. It records the old owner, rechecks ownership, and only then releases that lease. A merely slow, running or uncertain owner cannot be reclaimed.

## Validation

The targeted scheduler, account-continuation, dispatcher and recovery suite passed **78 tests**. This includes the real cloud coordinator entrypoint with mocked storage, preserved accounting/parent continuity, persistent storage-pause reasons, rejection of unrelated pauses, fail-closed preparation, exact lease behavior, and unchanged sealed source fingerprints. Full-suite and cloud completion evidence will be recorded below after execution.

## Current deployment entrypoints

```sh
PAPERLAB_ALL_PUMP=1 PAPERLAB_ONLINE_SCHEDULE=1 PAPERLAB_ONLINE_MODE=hourly modal deploy solana_online_cloud.py
PAPERLAB_ALL_PUMP=1 modal deploy checkpoint_runtime_cloud.py
modal deploy checkpoint_eval_dispatch_cloud.py
```

Keep `checkpoint_eval_cloud.py` unchanged: it is retained as the sealed scoring implementation, not the current preparation deployment entrypoint. The accelerated 2.5-second/replay candidate remains an isolated experiment; this repair does not promote it or establish profitable learning.
