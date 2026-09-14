# Expanded Pump/PumpSwap paper universe — September 14, 2026

The deployed hourly trainer now uses `all_observed_pump_v1` and
`solana_all_observed_quotes_v3`. All received Pump/PumpSwap markets can enter the
paper decision queue. No allowlist, age, popularity, two-sided flow, mayhem,
eight-active, 128-watched or 512-discovered-token restriction applies in this mode.
It discovers existing curves from trades and existing/arbitrary PumpSwap pools
through verified pool-account metadata. It preserves raw unresolved pool events.

This is **all observed markets**, not a claim that every historical or newly
created token is captured or receives inference. Public RPC delivery has gaps;
collection runs during fifteen-minute hourly training windows. There is no
historical backfill or between-window collection in this service. The full fly
still provides one shared inference slot per five-second target, using three-step
bursts and twelve distinct price samples for warmup. Logs expose observed,
observable, ready-for-native and never-served-active counts separately.

## Cloud coverage measurement

The approximately 80-second unscheduled, read-only probe
`universe-probe-20260914-01` completed successfully:

| Measurement | Observed |
| --- | ---: |
| Distinct discovered tokens | 1,336 |
| Verified existing pool lookups | 1,033 |
| Archived events | 25,512 |
| Decode errors / lookup failures / token evictions | 0 / 0 / 0 |
| Still-unresolved pools at stop | 2 |
| Raw trades received before pool resolution | 1,568 |

Those 1,568 raw events are retained, not retroactively inserted into model history.
After metadata arrives, a later fresh trade is needed for pricing. At the last
periodic quote snapshot, 170 of 1,272 discovered tokens had an entry-capable quote;
314 had stale trades, 680 lacked a subsequent trade, 55 lacked minimum fill
capacity, and 53 had an unpriced quote currency. Final discovery continued to
1,336 after that snapshot. These counts are a short sample, not a market census.
See [raw probe measurements](all-pump-probe-20260914.json).

## Pricing and resource boundaries

SOL and USDC markets use contemporaneous measured USD rates; USDC is not assumed
to equal one dollar. Other quote currencies can be converted through fresh
observed paths to those anchors. Unconnected or stale quote currencies stay
visible but unpriced. Unknown creation time uses an explicit feature sentinel;
first observation is not presented to the model as a known launch time.

The paper simulator still needs valid, fresh reserves and at least $1 of permitted
fill capacity. It retains one percent of actual reserves per fill, $250 acquisition
cost per token, finite cash, fees and slippage. No fake fill is made for missing
liquidity, prices or migration data. Reserve marks remain indicative rather than
verified executable exchange quotes; fee/transfer behavior is not fully modeled.
Allowing more markets does not resolve that evaluation limitation.

All raw received events are archived until the explicit 2 GiB per-window resource
guard stops collection; this replaces the old event-count cap for this mode.
The existing overall archive and monthly budget guards remain. The $100 total
monthly cloud authorization is unchanged, with $85 allocated to shared workers
and $15 reserved for collector/overhead plus the existing admission margin.
The probe settled about $0.0098 and the native verification about $0.0164 in the
compute ledger. Its cumulative shared-worker estimate was $11.1181 afterward;
these are estimates, not the Modal invoice.

## Verification and current runs

- Full local suite: **1,147 passed, 26 skipped**, 206 seconds. The two warnings
  concern intentionally local Modal worker tests. Focused expanded-path tests:
  92 passed, including 600-token admission, raw unresolved-event retention,
  quote conversion, old/mayhem markets, account credit, and exact frozen replay.
- Cloud native synthetic verification `audit-smoke-20260914-03` completed:
  six opportunities each for the untrained fly and latest checkpoint, no skipped
  opportunities, unchanged native/Q weights, and four audited trained-policy
  fills. This verifies mechanics only. See [native results](all-pump-native-smoke-20260914.json).
- Both trainer and evaluator were deployed with `PAPERLAB_ALL_PUMP=1` and resumed
  after verification. The old unused evaluation plan was archived unchanged.
- New prospective plan `evaluation-1789402218` seals 13 checkpoints plus
  untrained, cash and always-long controls, v3 quotes and source hashes. It is
  waiting for a future broad-universe tape. See [sealed plan](all-pump-evaluation-plan-20260914.json).
- The latest completed real episode remains `solana-online-20260914-152719`:
  **126 native observations, 106 Q updates, 46 fills, $880.40 final indicative
  equity, −$119.60 PnL** from $1,000. Its reward reconciliation passed. This was
  the prior narrow universe; no real broad-universe episode has completed yet.
- Training is hourly and awaiting the next due window. Eligibility is
  16:27:19 UTC / 11:27:19 CDT; the next normal coordinator tick is
  **16:32 UTC / 11:32 a.m. CDT**. The existing monitor now checks broad coverage
  and v3 replay metadata as well as training health, budget and failures.

## Remaining evidence

Verify the first completed expanded episode's account/reward audit, actual
inference coverage, event volume and lookup health. Then score the sealed
checkpoint cohort on the same unseen opportunities. No profitability or improved
learning is established by broader discovery. Full continuous/historical coverage
would require an independently tested data/backfill design and resources within
the authorized budget; it cannot be inferred from this short successful probe.

The account decoder pins the official [PumpSwap IDL](https://raw.githubusercontent.com/pump-fun/pump-public-docs/main/idl/pump_amm.json)
with its source hash. Public endpoints are documented by
[Solana](https://solana.com/docs/references/clusters).
