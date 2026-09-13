# Preparing a fresh market comparison of learning rates

The [lower-rate neural assay](fly-learning-rate.md) reduced later weight movement
but introduced an extra BUY. It did not establish better trading. The next
comparison must measure fresh account outcomes and turnover after execution
costs, with a development decision saved before the test phase.

## Checkpoint preparation

**Preparation only: no new market window is registered yet.** The checkpoint
capture implementation passed 25 local tests with native construction and
propagation mocked. Cloud execution and verification are separate steps. The
[preflight record](../reports/fly-rate-checkpoint-preflight-12.json) pins this
code and the observed coverage of the old cohort.

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
invoice. The temporary app should be stopped after its original call completes
and the captured files are verified.

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
the cloud. Synthetic tests are not trading evidence.

The eventual market registration will pin the completed capture and its source
hashes, all comparison arms, future windows, costs and selection rules. Newly
exported memory is still memory trained at eta 0.001; using eta 0.0001 afterward
tests a lower online learning rate, not retraining the starting memory at that
rate. No policy is promoted by exporting a checkpoint.
