# Recurring Solana paper RL

The `fly-paper-solana-online` Modal application resumes the saved fly and Q head,
keeps one $1,000-origin paper portfolio, and admits new observed Pump launches.
It migrates the account from `solana-live-14-next-next` without replacing its
$971.24 cash or outstanding PAPER inventory. The prior single-token records
remain immutable. No wallet, signer, or real order execution is involved.

## Learning and coverage

The collector subscribes to confirmed Pump.fun and PumpSwap logs while a training
window runs. It checks received markets every five seconds and admits up to eight
eligible active tokens (plus at most 128 watched active/parked mints), earliest eligible creation receipt first. Other received launches
remain in the bounded discovery archive. Coverage is not all Solana, not all launchpads,
and not continuous between training windows. There is no backfill of those gaps.

The shared full fly processes one token per five-second target, rotating in
three-observation bursts. Each token first needs twelve distinct sampled prices.
Per-token sampling and fills can proceed while another token gets inference;
one token's cash flows cannot create another token's reward. The Q head learns
from that token's net cash flows plus changes in its liquidation value. Invalid
quotes break learning credit; there is no fabricated fill or invented rug-loss label.
Learning from sustained loss of liquidity remains an evaluation limitation.

On token switches, transient neural activity and eligibility traces reset while
plastic weights and native memory remain. The native reward is suppressed on the
switch; subsequent contiguous observations can supply equity reinforcement.
The shared Q head can learn a mint-specific transition across other tokens, up
to a 60-second gap, with elapsed-time discounting. Training logs report both
native plasticity and Q loss, gradient, TD error, exploration, and weight changes.
This experimental online Q learner does not have a target network or replay buffer
and has no claim of convergence or profitable trading.

Held inventory is never erased or replaced with fresh cash. Unavailable holdings
after two minutes and sub-dollar dust move to a parked, still-tracked list; they
retain cash flows and risk allocation and can reenter when usable. Flat tokens leave
the active queue after two minutes of unusable data or twenty minutes of token age;
retired tokens are not readmitted. This prevents picking the same apparently
successful token repeatedly. Acquisition cost, including fees, is capped at $100
across all holdings, with a $25 order cap and the existing account loss stop.
Unavailable holdings retain their quantities and risk allocation, with zero
stress liquidation value. The inherited position's initial risk reserve is
conservatively approximated by legacy net cash spent; it is not a reconstructed
FIFO cost basis. Exact swap execution and sellability are still unverified.

## Background execution and budget

The lightweight coordinator is scheduled in Modal every fifteen minutes. It
starts a detached, at-most-fifteen-minute training window when due and records
its immutable run ID and call ID before it can dispatch another. Default cadence
is twelve hours between window starts, approximately two windows per day.
The current operator-selected `hourly` mode targets twenty-four windows daily,
with the next due start handled by the fifteen-minute coordinator check.
The worker resumes checkpoints, optimizer/RNG state and all account records.
The read-only `python -m paperlab.solana_online_audit <downloaded-run-directory>`
reconstructs recorded fills and available portfolio marks independently.
Collection pauses between windows; the laptop can be off throughout.

The coordinator confirms the previous Modal call terminated before treating a
published completion file as resumable. An ambiguous dispatch, crashed worker,
invalid result, paper account loss stop, or storage cap disables further dispatch pending review. Busy
shared writers are retried at the next scheduled check. There is no blind crash
retry. Budget exhaustion defers the next attempt until the following UTC month.

The user authorized a $100 total monthly ceiling on September 13, 2026 and chose
to pause at the budget limit. The online worker now has an $85 allocation in the
existing shared worker ledger and stops new reservations at 75% ($63.75).
Previous spend remains counted; $15 is allocated to collector/other overhead,
and the retained safety margin leaves additional room for scheduler/storage.
Other experiments retain their existing lower limits. Each twenty-minute hard
timeout reserves about $0.319; successful
fifteen-minute windows have recently settled near $0.242 under the conservative
2x margin and 3x nonpreemptible rate estimate. Hourly windows would cost about
$175 per thirty days if uninterrupted, so this cadence intentionally pauses
before month-end when its reservation threshold is exhausted, resuming next UTC month.
Other workers sharing this ledger also consume this allowance. This is an
estimate, not the provider invoice.
The low-resource coordinator has an approximately $0.99/month compute bound at
its full scheduled timeout, including a 2x pricing margin. Existing collector and
provider budget settings are unchanged; neither free credits nor future profits
are assumed to fund more compute.

The service stops admitting windows once its preserved Solana archives exceed
8 GiB; a window can add more before the next check. Full native arrays are sampled
every twelve neural observations, while all scalar learning diagnostics, decisions,
executions and final checkpoints are retained. Archiving/retention management is
required before this storage limit can be reached indefinitely.

The documented scheduler behavior follows [Modal scheduled functions](https://modal.com/docs/guide/cron).
Resource estimates use [Modal pricing](https://modal.com/pricing).

## Operation

```sh
# Default cadence is budget paced; deployment updates do not erase control state.
PAPERLAB_FLY=1 PAPERLAB_UNIVERSE=1 PAPERLAB_SCHEDULE=0 \
  PAPERLAB_ONLINE_SCHEDULE=1 PAPERLAB_ONLINE_MODE=paced \
  uv run --extra cloud modal deploy solana_online_cloud.py

# Read-only status: /state/solana-online/control.json identifies the current run.
uv run --extra cloud python -m paperlab.solana_watch solana-online-YYYYMMDD-HHMMSS

# Explicit cadence change under the coordinator lease; preserve the current window.
# Never retries an uncertain dispatch or overrides a loss/failure/budget pause.
uv run --extra cloud python -c "import modal; print(modal.Function.from_name('fly-paper-solana-online', 'set_cadence').remote('hourly'))"
```

To stop dispatch, set `enabled` false in the durable control JSON under the
coordinator lease, then remove `PAPERLAB_ONLINE_SCHEDULE=1` and redeploy to remove
scheduler wakeups. Do not clear a live writer lease or reuse a run ID. A cadence
change uses the `set_cadence` function to update the durable control's `mode`
(`paced`, `six_hour`, `hourly` or `consecutive`) and append a change record; an
environment change alone cannot overwrite a running service's stored policy.

## Remaining evaluation

This changes the training universe and mechanics. It does not validate returns.
The next evidence gate is multiple tokens/dates with cost-aware portfolio audit,
then matched market-only, frozen-fly and compact controls plus a chronological
holdout. News features remain disabled. Continuous cheap discovery, replay/target
networks, and compact or distilled inference are follow-on work; they are not
silently represented as features of this service.
