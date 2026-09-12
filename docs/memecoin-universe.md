# Dynamic memecoin research universe

Enable with `PAPERLAB_UNIVERSE=1`. This replaces the worker's BTC workload with a
new `meme-pools-v1` experiment. The BTC accounts, checkpoints, news, and recorded
losses remain under `fast-5m`; they are not reset or counted as memecoin results.
No wallet, transaction submission, paid trading stream, or real-order API is used.

## Coverage and collection

The small `universe_collector` function subscribes to PumpPortal's **free token
creation and migration events**, using one WebSocket. It does not subscribe to
metered token/account trade events. A bounded 240-second collection window starts
every five minutes, leaving time for publication within the 300-second function
timeout; connection setup, restart gaps, and provider errors are recorded.
This is near-continuous discovery with observable gaps, not guaranteed complete
event delivery or a backfilled blockchain index.

GeckoTerminal supplies all-network new-pool pages, rotating trending pages, and
rotating searches for established meme tickers. Ticker search only discovers
candidates: identity always uses network, contract, and pool addresses. Included
tokens are speculative candidates, not a verified taxonomy of memecoins. Newly
created Pump.fun tokens are recorded immediately on receipt, but they need an
indexed pool, adequate market data, and eligibility before becoming tradable in
the simulation. Creation events alone cannot establish liquidity or price.

At most **240 pools** are selected for roughly one-minute refresh. Assigned pools
take priority; eligible incumbents and a rotating discovery tranche fill the rest.
HTTP work is capped at eight requests per round, at least 7.5 seconds apart.
Assigned-pool batches precede two rotating discovery requests and other refreshes.
The current [API reference](https://api.geckoterminal.com/docs/index.html) states
approximately ten requests per minute, varying with traffic (checked September 12,
2026); older 30/minute guidance is not used. `provider-rate.json` persists pacing
and exponential 429 cooldowns across windows. `Retry-After` can extend the pause.
After a 429, the round ends; recovery starts with assigned pools rather than
the remaining discovery requests. A window that cannot fit a request records
that fact without issuing it.
Overflow and provider throttling remain visible. Capacity is a ceiling, not a promise
that 240 markets will refresh every minute. A bounded working set avoids unlimited
memory growth. The durable registry retains rejected and disappearing pools and
all received launch events. First-seen timestamps never move backwards. A pool
that falls out of collection is reported as missing, not silently counted as a
successful surviving investment.

**This is not every memecoin on every chain.** Free indexer pagination, ranking,
launch-provider scope, disconnects, and the compute budget constrain coverage.
The archive records only actual observations; repeated requests in the same
minute do not manufacture additional examples. An indexer may itself return
cached/stale underlying market data despite a fresh HTTP receipt.

## Two paper portfolios

Each strategy has **$1,000 total**, divided into four persistent $250 sleeves.
Maximum target exposure is 50% per sleeve, with $25 maximum orders. Quantities,
fees, cash, and halt state survive restarts and token changes. A sleeve may select
a replacement after one hour only when flat with no pending decision. Selection
uses observed five-minute volume, never subsequent return. Duplicate contracts
across different pools cannot occupy multiple sleeves of one portfolio.

The compact policy is shared across its four sleeves. The fly loads the full
graph once per worker call and restores an independent native state for each
sleeve's token. The first two native steps should show 500 ms then 1,000 ms for
each context, rather than accidentally continuing another token's neural state.
Switching to a different token starts that token's fly context from the saved
initial state. There is no claim of cross-token transfer learning in this fly
version; PPO does pool training examples across assets.

Both strategies receive up to 64+ observed minute-price history points and the
causally available global RSS/FinBERT features. There is no fabricated historical
news or claim that the feed supplies token-specific fundamentals. Fly plasticity
starts once its assigned pool has 64 eligible observations. Compact initially
waits in cash, then trains when four series each have at least 100 eligible
observations. These are data gates, not a fixed wall-clock completion promise.

Pooled PPO uses separate episodes per pool, common wall-clock train/validation/test
cutoffs, and five-observation action intervals to approximate the five-minute
forward decision horizon. Price gaps and irregular polling still make those
horizons approximate. Prices from different tokens are never concatenated into
a fictitious chart. Missing/ineligible market states reject orders and value
unpriced inventory at zero for a conservative stress mark. Full archived series
include their unavailable observations after entry into the training cohort;
short-lived tokens without enough context remain outside that cohort and must
not be represented as tested successes. Candidate updates run on a fixed six-hour
cadence after eligibility; checkpoint promotion is never selected by test profit.
Per-asset audit returns are separate episodes, not the actual portfolio return.

## Simulation limits and guards

Admission requires observed liquidity at least $25,000, five-minute volume at least
$1,000, at least three buys and three sells, pool age at least 15 minutes, a positive
price, and a receipt no older than 180 seconds. Major/stable assets identified by
provider metadata are excluded. These are market-data filters, **not an audit of
token safety, mint/freeze authorities, ownership, taxes, or sellability**.

Indexer USD prices are **not executable bid/ask quotes**. The declared cost scenario
adds a 100-bps artificial spread, 100 bps per-side slippage, and 125 bps per-side
fees. A $25 order is at most 0.1% of the minimum admitted reported liquidity.
Those assumptions do not reconstruct exact AMM impact, token taxes, gas, priority
fees, failed transactions, MEV, or withdrawal/rug behavior. This is an indicative
price research simulation; do not use its apparent P&L to authorize real trading.

Pending orders require a subsequent observed market receipt and expire after
420 seconds. Missing pools retain token quantities but contribute zero inventory
value to the stress mark. They are not sold at an old price. Restored liquidity
does not produce an artificial fly reward across an unpriced gap. A provider
outage is therefore distinguishable from an actual realized sale loss.

## Cloud and diagnostics

```sh
PAPERLAB_SCHEDULE=1 PAPERLAB_UNIVERSE=1 PAPERLAB_FLY=1 .venv/bin/modal deploy cloud.py
# Bounded native isolation check, strictly separate synthetic archive:
.venv/bin/python -c 'import modal; print(modal.Function.from_name("fly-paper-lab", "worker").spawn(diagnostics=True).object_id)'
# Disable BOTH schedules, preserving their state:
PAPERLAB_SCHEDULE=0 PAPERLAB_UNIVERSE=1 PAPERLAB_FLY=1 .venv/bin/modal deploy cloud.py
```

The collector owns `fly-paper-lab-universe` (`universe.db`, `universe-snapshot.db`,
`latest.json`, `last-window.json`, `budget.json`). It publishes a closed SQLite
backup through atomic replacement after committed updates. The trader reads
`universe-snapshot.db`, not the changing writer database: a read-only connection
cannot recover a hot journal captured while that database is being modified.
Volume publication uses the async API so it does not block launch reception.
The trader only reads this volume. The trader
owns `fly-paper-lab-state` (`meme-pools-v1/paper.db`, `latest.json`, `watch.json`,
native checkpoints, and candidate PPO artifacts). The collector only reads the
trader's watch file. Each function has exactly one writer; no shared writable
SQLite database or shared budget file exists between them.

Resource ceilings are 0.125 CPU/256 MiB for collection and 2 CPU/8 GiB for the
full-fly worker. Separate monthly reservation caps sum to the existing $40 planning
ceiling: $15 collector and $25 worker. Existing worker spend stays in its original
budget ledger. Reservations double CPU/RAM rates, include startup time, and stop
substantive work at 75% of their cap. This can stop research early; it does not
guarantee full-month uptime. Provider billing limits remain authoritative, and
no provider spending limit is raised by this change.

An atomic Modal Dict ownership guard also prevents overlapping deployment
versions from opening a second writer. Normal completion or handled exceptions
release ownership; a hard kill can leave `writer_busy`. Verify the previous
input has actually ended before removing that writer's key in
`fly-paper-lab-writers`. Do not reset the budget or delete market/account data.
When migrating from a version without the guard, stop its scheduled app first,
then deploy the guarded version; record that collection gap.

`multi_paper_decision` logs include pool identity, model output, pending order,
fill or rejection, quantities, fees, stress equity, and learning diagnostics.
`universe_collected` logs include coverage counts, attempted/completed requests,
API errors, request overflow, cooldown state, and per-assigned-pool quote ages and
rejection reasons. Fresh-but-inactive pools are distinguished from stale data.
The launch `health` table records reconnect gaps. Compact rollout and optimizer
artifacts retain pool identity, probabilities, losses, gradients, and updates.
The older static BTC visualization does not automatically become a live memecoin
dashboard. Inspect these new records for this experiment's progress.

## Provider references

- [PumpPortal free launch/migration versus metered trade streams](https://pumpportal.fun/data-api/real-time/)
- [GeckoTerminal current public API reference and approximate rate limit](https://api.geckoterminal.com/docs/index.html)
- [GeckoTerminal new pools, trending, and multi-pool endpoints](https://apiguide.geckoterminal.com/changelogs)
- [Modal resource limits and usage billing](https://modal.com/docs/guide/resources)
