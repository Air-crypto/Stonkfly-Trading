# Three-day frozen checkpoint forward test

This separate paper-only experiment selects `solana-online-20260916-005626`
because it had the highest marked return in one earlier frozen comparison.
That selection was made after seeing the earlier result. Only observations
received after this new experiment starts count as its prospective evidence.
It is not a claim that the checkpoint is the best policy in general.

The new Modal app is `fly-paper-frozen-forward`. It starts a new $1,000 paper
account once, then carries cash, holdings, acquisition basis, fees, and per-token
cash flows across all subsequent sessions. The old training account is not
imported. Fly synaptic weights and readout weights are copied and hashed;
learning is disabled and exploration is zero. The saved readout has 4,853
historical updates; this experiment performs zero new backprop updates.

The model chooses FLAT (request an exit) or LONG on received Pump/PumpSwap
markets. Existing execution constraints remain: $250 acquisition cost per token,
shared cash only, and each fill bounded by one percent of observed reserves.
Simulated fees are 125 basis points and slippage is 100 basis points per side,
plus the recorded bid/ask spread. No wallet or real orders are involved.

An always-long account begins with its own $1,000 and follows the trained
model's decision opportunities with the same execution rules. Cash is the
$1,000 reference. This compares action choices on matched opportunities; it does
not compare independent token-selection strategies. News remains disabled.

## Runtime and continuity

The coordinator checks every minute. The first session lasts three minutes for
startup verification; later sessions last up to 15 minutes, with a 20-minute
hard function timeout. Both accounts persist. Neural transient state resets at
session boundaries while learned weights remain fixed. Feature price histories
carry forward. Pending unfilled orders expire; they are never filled across an
unobserved gap. Session handoff, initialization and feed reconnect gaps are
recorded, so this is not uninterrupted or complete coverage of every token.

The worker uses one CPU core and 8 GiB memory with standard preemptible execution.
It targets observations every five seconds, subject to native computation and
fresh data. A repeated input after interruption audits the retained ledger and closes that
partial segment without replaying its trades. The coordinator then continues in a
new session. Other worker errors or unverifiable state pause for review. Recovery
retains cash, holdings, fees and fills; price history warms up again after the gap. The coordinator
stops this app on completion, budget exhaustion, or a failure requiring review,
using the synchronized `modal.experimental.stop_app` interface. Its actual cloud
shutdown path was verified during the September 18 recovery.

## Budget

The user prioritized this test over further hourly training. A $17.50 internal
compute envelope is debited once from the existing shared worker ledger before
launch. This prevents the training/evaluation workers from spending that same
allowance. Their earlier budget guards remain in place, so they can pause sooner.
This debit is a reservation, not a provider charge. The overall authorization
remains $100; the original $63.75 shared-worker cutoff and overhead buffers remain.

The forward runner uses a separate durable budget inside that prepaid envelope,
reserves a full timeout before each worker, and retains the reservation on failure.
Pricing estimates use a 2x margin over standard CPU/memory rates. The experiment
ends after 72 wall-clock hours or sooner if its compute or archive guard trips.
Raw closed event databases are compressed losslessly. Retained storage is bounded
to 64 GiB, allowing room for the next collector archive before dispatch.

The obsolete `fly-paper-lab` app was stopped on September 17 to free two of the
Starter plan's five schedule slots. The current online trainer and frozen replay
evaluator were not redeployed or stopped for this experiment.

## Evidence

The dedicated `fly-paper-frozen-forward` Modal volume contains:

- `manifest.json`: selected checkpoint hashes, start/end, policy and allowance.
- `control.json`: current call, cumulative conservative spend, pause or completion.
- `state.json`: account continuation from the last verified completed session.
- `sessions/session-*/opening.json`: exact opening accounts and feature history.
- `sessions/session-*/decisions.jsonl`: decisions, predictions, fills, quotes and accounts.
- `sessions/session-*/latest.json`: newest within-session observation and metrics.
- `sessions/session-*/completed.json`: frozen-weight and reconstructed-fill checks.
- `sessions/session-*/continuation.json`: carried balances and feature history.
- `sessions/session-*/events.db.gz`: lossless raw event archive after completion.
- `sessions/session-*/recovery.json` and `recovered-state.json`: audited partial
  segment evidence and continuation, separate from normal completion.
- `operator-recovery-session-*.json`: explicit recovery identity, retained balances,
  original deadline and budget, and conservative overhead reconciliation.

Report marked P&L, realized P&L, unrealized P&L, fees, cash, open positions and
unavailable quotes separately. The capacity-stress scenario permits one constrained
exit fill per position and values the remainder at zero. It is not an actual
realized loss. Report cloud costs separately from trading P&L. Changes in marked
inventory alone do not establish an executable, profitable trading edge.

The [September 18 recovery](../reports/frozen-forward-recovery-20260918.md)
retains session four's 14 cumulative buys and original endpoint. Interrupted
segments have no final in-memory weight comparison; checkpoint-file hashes and
recorded frozen diagnostics are verified and this limitation is reported.
