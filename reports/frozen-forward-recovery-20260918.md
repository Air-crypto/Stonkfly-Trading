# Frozen forward account recovery

The September 17 forward worker stopped during session four. A repeated input
re-entered its existing session directory and hit the immutable-session guard.
The old shutdown helper also created an incompatible asyncio event loop around
Modal's client, causing paused coordinator calls to hang until their timeout.

Recovery preserves the interrupted session as a partial segment. It does not
claim that the scheduled session completed or that the overnight gap was traded.
The checkpoint and original September 20, 15:58 Chicago endpoint stay unchanged.

## Retained account

All 121 complete decision rows reconcile against the opening cash, inventory,
fees and prior order intents. The metadata reconstruction consumed 99,589 raw
events only through the last recorded observation cursor. No partial trailing
record was discarded and the latest snapshot matches the last complete row.

The trained account retains 14 cumulative buys, 11 open positions, cash of
$89.63749925593688 and $11.239043219062507 cumulative trading fees. The always-long
account retains four buys, its holdings, $0.0010000000000135 cash and
$12.345666666666666 cumulative fees. Neither account receives fresh capital.

Missing quote marks remain missing; the old marked equity is not a realized loss
or a current price. The next session obtains fresh live quotes and warms up its
price history again. Pending intents expire across the gap. Prior source files
and the failed session's ledger remain unmodified, with separate recovery files.

## Runtime changes

- A repeated cloud input with a recoverable interruption audits the retained
  segment and returns its recovered account rather than repeating trades.
- The coordinator records interrupted segments separately from complete sessions,
  retaining the full compute reservation and dispatching the next distinct session.
- Non-interruption failures, ledger mismatches, changed checkpoint hashes,
  missing held-token metadata and uncertain cloud-call status still stop recovery.
- Explicit operator recovery matches the exact paused call and reviewed ledger
  hash, archives the previous control/account, and is idempotent.
- The shutdown helper now uses Modal's synchronized `experimental.stop_app`.
  The actual paused cloud deployment successfully stopped itself during the probe.
- Paused coordinator calls are accounted for; recovery conservatively reserves
  $0.2646675568040753 for the previously unmetered wakeups and recovery/probe work
  within the existing $17.50 prepaid envelope. No new allowance was added.

The interrupted process has no final full in-memory weight comparison. Recovery
verifies immutable checkpoint-file hashes, recorded zero native weight changes,
fixed readout update counters and learning-disabled configuration. It reports
that narrower verification explicitly rather than calling it a normal frozen
completion.

## Validation

104 relevant tests passed, including 25 forward-run/recovery tests. Tests cover
account carryover, incomplete records, future metadata rejection, unchanged
source evidence, repeated recovery, partial-segment promotion, cancellation,
uncertain RPC errors and the synchronized shutdown interface. The full actual
partial ledger was independently rehearsed locally without propagating the brain.
Sealed replay source files remain unchanged.

The [cloud recovery evidence](frozen-forward-recovery-20260918.json) records the
operator receipt, next opening-account comparison and resumed runtime status.

## Successful resumed session

Session five completed after 182.85 seconds. It processed 14 new neural
observations (409 cumulative), recorded one additional trained-account buy, and
passed both full cash-flow audits. The final native and readout weight comparison
passed with zero new backpropagation updates. All 15 cumulative trained fills and
12 positions carry forward, with $0.00008963749924949124 cash remaining after the
new purchase and $12.345677905709886 cumulative trading fees.

Its closing marked equity was $464.27, but eight of twelve held tokens lacked
fresh quotes and were conservatively marked at zero. This snapshot is not a
complete realizable liquidation value or evidence of profitable learning.
The scheduled coordinator accepted the result and dispatched session six without
operator intervention, retaining the original deadline and allowance.

Session six's opening state exactly matches session five's continuation. Its
900-second live window started with both learning modes disabled and cash reset
disabled. The handoff gap was 32.67 seconds. The prepaid ledger stood at $0.589682
spent or reserved of $17.50 after dispatch; this is the dedicated experiment
ledger, not the full monthly Modal bill.
