# Recurring Solana paper RL

The `fly-paper-solana-online` Modal application resumes the saved fly and Q head
and admits new observed Pump launches. Following the user's September 14 request,
`fresh_training_episode` mode starts each hourly training window with $1,000 cash,
zero fees and no holdings, while carrying model weights, optimizer/RNG state and
lifetime update counters. Transient neural activity and per-token action credit
reset at the episode boundary. No wallet, signer or real orders are involved.

The original continuous account is archived at `solana-online-20260914-003217`:
it ended with $902.60 cash and unavailable inventory. Its ledgers, holdings, fees
and losses are preserved. Independent training bankrolls are not deposits into
that account and must not be reported as continuous portfolio performance.

Each fresh episode makes the entire $1,000 available, with no $902 cash floor or
$100 aggregate allocation cap. A token can acquire up to $250 including fees.
Each simulated fill is additionally limited to one percent of observed real SOL
reserves in USD; unavailable liquidity prevents execution. LONG may build a
position up to its allowance; EXIT sells subject to the same liquidity limit.
Cash cannot go negative, and only a new window receives new paper capital.
The per-episode loss allowance is the full bankroll; the cloud budget is separate.

Every completed training episode retains its ending inventory and cost-adjusted
stress valuation. `/solana-online/episode-results/<run-id>.json` and control
`episode_totals` preserve episode PnL and fees, including depleted episodes.
These are training diagnostics, not a live-profitability estimate. Fresh-capital
training continues next hour after account exhaustion; infrastructure failures,
ambiguous dispatch, budget and storage guards still pause it. The separate GSPO
replay pilot remains a completed, unpromoted experiment.

## Learning and coverage

The collector subscribes to confirmed Pump.fun and PumpSwap logs while a training
window runs. With `PAPERLAB_ALL_PUMP=1`, it checks received markets every five
seconds and queues every observable token, including existing curves and arbitrary
PumpSwap pools. The former eight-active, 128-watched and 512-discovered limits,
age limits, two-sided-flow and mayhem exclusions are disabled. Unknown pool events
are archived while verified account metadata is fetched; they are not retroactively
injected into decisions. Prices use fresh SOL/USD, measured USDC/USD, or fresh
observed conversion paths. Unknown quote assets remain visibly unpriced.
Coverage is not all Solana, not all launchpads, and not continuous between training
windows. There is no backfill of gaps. See the [coverage report](../reports/all-pump-universe-20260914.md).

The shared full fly processes one token per five-second target, rotating in
three-observation bursts. Each token first needs twelve distinct sampled prices.
Per-token sampling and fills can proceed while another token gets inference;
one token's cash flows cannot create another token's reward. The Q head learns
from that token's net cash flows plus changes in its indicative liquidation value.
The September 14 audit found that the previous protocol dropped quote-gap and terminal
losses. In `mint_credit_terminal_v2`, outstanding Q credit survives gaps and parked
inventory. Every episode ends with non-bootstrapped terminal settlements; raw rewards
must reconcile to account PnL, or the account audit fails. A missing terminal observation
uses the explicitly tagged conservative zero mark, not an invented sale or confirmed rug.
Historical checkpoints and outcomes remain unchanged and are labeled as using the
previous incomplete-credit protocol.

On token switches, transient neural activity and eligibility traces reset while
plastic weights and native memory remain. The native reward is suppressed on the
switch; subsequent contiguous observations can supply equity reinforcement.
The shared Q head preserves mint-specific credit across other tokens and quote gaps,
with elapsed-time discounting. Only uninterrupted same-mint feedback reaches native
plasticity; the difference from fully settled Q rewards is explicitly reported. Training logs report both
native plasticity and Q loss, gradient, TD error, exploration, and weight changes.
This experimental online Q learner does not have a target network or replay buffer
and has no claim of convergence or profitable trading.

Within each episode, held inventory is never erased or replaced with fresh cash. Unavailable holdings
after two minutes and sub-dollar dust move to a parked, still-tracked list; they
retain cash flows and quantities and can reenter when usable. Flat tokens leave
the active queue after two minutes of unusable data. In all-observed mode they can
return when fresh quotes recover; there is no permanent age-based retirement.
The legacy bounded mode additionally retires flat tokens at twenty minutes.

The archived continuous mode's September 13 allocation repair (`quarantined_inventory_cash_floor_v1`) replaced
the blanket $100 open-cost reservation that blocked trading behind unquotable holdings.
After two minutes without a usable quote on a healthy feed, held inventory is
quarantined from the tradable acquisition-cost allowance. A feed outage alone
cannot trigger quarantine. Quantities, basis, cash flows, cash, and all past fees
remain unchanged, with zero stress liquidation value until a valid quote returns.
Recovery immediately restores that inventory's acquisition-cost reservation.
Quarantine is not a sale, a debt release, a cash refill, or a confirmed rug label.

In that continuous mode, new entries are sized at execution to at most $2.50 acquisition cost per mint,
including fees, and clipped to both the $100 tradable cost allowance and cash above
$902 (the original $900 account loss stop plus a $2 buffer). The legacy inventory
remains in the account; its acquisition cost is not included in that $100 allowance
while quarantined. Aggregate historical open basis can therefore exceed $100.
Existing holdings can still be sold in $25-notional chunks, and LONG holds an
existing lot. No buy can replenish or evade the preserved account loss budget.
Infeasible flat-to-long actions are masked from sampling and bootstrap targets;
rejections are recorded as no-fill outcomes. Zero-reward rejected actions without
inventory can be skipped, while prior inventory credit is preserved.
The inherited position's initial risk reserve is
conservatively approximated by legacy net cash spent; it is not a reconstructed
FIFO cost basis. Exact swap execution and sellability are still unverified.

## Quote and evaluation correctness after the September 14 audit

`solana_observed_entry_exit_v2` separates a fresh observed reserve-derived price from
entry eligibility and exit capacity. A one-sided market or an entry-risk filter no
longer implies that a held asset is worth zero. Fill direction determines the guard,
including LONG targets that rebalance by selling. Simulated exits remain limited to
1% of actual SOL reserves. Reports show both full indicative inventory value and
capacity-limited next-fill proceeds; neither proves exchange executability.

The checkpoint evaluator compares retained frozen fly/Q pairs, an untrained pair,
cash and always-long controls at identical recorded attention opportunities and
latency. Cutoffs precede evaluation tapes; hashes cover checkpoints, tapes, schemas,
source and native code. This evaluates conditional buy/sell behavior, not independent
universe selection. It also tests a deployment-mode distribution with native learning
disabled; plasticity-derived input features differ from online training.

See [the system audit](../reports/system-audit-20260914.md) and
[the evaluation protocol](../reports/checkpoint-evaluation-protocol.md).

## Background execution and budget

The lightweight coordinator is scheduled in Modal every fifteen minutes, at
minutes 02, 17, 32 and 47. This offsets dispatch from the legacy worker's five-minute
schedule; the shared atomic writer lock still prevents overlapping writes. It
starts a detached, at-most-fifteen-minute training window when due and records
its immutable run ID and call ID before it can dispatch another. Default cadence
is twelve hours between window starts, approximately two windows per day.
The current operator-selected `hourly` mode targets twenty-four windows daily,
with due starts handled by the fifteen-minute coordinator checks or a training
refresh from the every-minute evaluation dispatcher. Startup, worker contention
and budget guards can delay starts. See the [verified Modal schedule](modal-schedule.md)
for the active recurring jobs and legacy wakeups.
The worker resumes checkpoints and optimizer/RNG state. Account records carry
only in continuous mode; fresh training episodes archive their parent and start
a separate paper account. Opening records hash both parent checkpoints and the
parent completion/state to make this boundary auditable.
The read-only `python -m paperlab.solana_online_audit <downloaded-run-directory>`
reconstructs recorded fills and available portfolio marks independently.
Collection pauses between windows; the laptop can be off throughout.

The coordinator confirms the previous Modal call terminated before treating a
published completion file as resumable. An ambiguous dispatch, crashed worker,
invalid result or storage cap disables further dispatch pending review. In
continuous mode, a paper account loss stop or exhausted entry capacity also
disables it. Fresh training mode retains the episode outcome and starts the next
independent account on the existing cadence. Busy
shared writers are retried at the next scheduled check. There is no blind crash
retry. Budget exhaustion defers the next attempt until the following UTC month.

Every completed new window independently audits its delayed fills, cash, inventory,
basis, fees, available valuation marks, and entry risk allowances. It records
`training_health`, nonzero reward updates, dropped rejected-order credit, and the
audit result alongside loss/gradient diagnostics. `learning_from_paper_execution`
means there were fills and nonzero reward updates, not that performance improved.
Zero-reward updates are labeled explicitly. The coordinator propagates these
diagnostics into its durable control record.

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
64 GiB; a window can add more before the next check. This bounded allowance
replaced the original 8 GiB cap after its September 15 stop. Pause reasons remain
visible across subsequent scheduler checks. Full native arrays are sampled
every twelve neural observations, while all scalar learning diagnostics, decisions,
executions and final checkpoints are retained. Archiving/retention management is
required before this storage limit can be reached indefinitely.

The documented scheduler behavior follows [Modal scheduled functions](https://modal.com/docs/guide/cron).
Resource estimates use [Modal pricing](https://modal.com/pricing).

## Operation

```sh
# Current all-observed hourly deployment; updates do not erase control state.
PAPERLAB_FLY=1 PAPERLAB_UNIVERSE=1 PAPERLAB_SCHEDULE=0 \
  PAPERLAB_ALL_PUMP=1 PAPERLAB_ONLINE_SCHEDULE=1 PAPERLAB_ONLINE_MODE=hourly \
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

Frozen checkpoint evaluation runs prospectively on Modal. Each daily cohort
retains all completed pre-cutoff checkpoints and designates the newest one as
the primary comparison before seeing results. The first completed market
episode that **starts after** that cutoff becomes its immutable test tape;
an episode overlapping the cutoff cannot qualify. Checkpoints trained on that
tape cannot join its cohort. Cash, untrained-fly and always-long controls share
the recorded attention opportunities, execution timing and cost model, each
starting with $1,000. Both the native weights and readout remain frozen during
scoring. New daily cohorts wait for the previous one to finish, and evaluation
uses training gaps and the existing shared budget rather than raising the cap.

An operator may advance one daily cutoff with the runtime's `request_prospective`
function, passing the verified `last_completed_batch` from evaluation control.
The request is serialized with preparation and the native worker, refuses
paused/error/budget states, preserves active cohorts and records the previous
schedule. Repeating it cannot replace a sealed cohort. Deployment:

```sh
PAPERLAB_ALL_PUMP=1 uv run --extra cloud modal deploy checkpoint_runtime_cloud.py
```

Plans, test-tape hashes, per-policy results and `comparison.json` are retained
under `/state/checkpoint-eval/<batch>/`. Smaller losses on one window do not
establish profitability. Check the preselected latest policy against cash and
untrained controls over several later dates, including fees, drawdowns and
liquidity-stress marks. Do not select the best historical checkpoint by test P&L
and then report that same test as independent validation. These comparisons
remain conditional on recorded opportunities, not full-universe execution tests.

This changes the training universe and mechanics. It does not validate returns.
The next evidence gate is multiple tokens/dates with cost-aware portfolio audit,
then matched market-only, frozen-fly and compact controls plus a chronological
holdout. News features remain disabled. Continuous cheap discovery, replay/target
networks, and compact or distilled inference are follow-on work; they are not
silently represented as features of this service.
