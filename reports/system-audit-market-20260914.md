# Market, execution, and GSPO audit — 2026-09-14

The recorded cash arithmetic is consistent, but the historical quote policy conflates **a rejected new entry** with **an unobservable or worthless holding**. This materially distorts the meaning of the reported losses. The historical figures remain unchanged. The new quote protocol separates those concepts prospectively; it still uses indicative reserve prices and cannot establish executable profitability.

## Scope and reproducibility

This is a local, read-only audit of archived market data and paper ledgers, plus focused code changes. It performs no native inference, cloud training, account reset, or live order.

The exact historical cohort is the eleven archived independently reset $1,000 episodes under `runs/market-attribution-01/`:

`solana-online-20260914-021337`, `031716`, `041721`, `053214`, `064712`, `074716`, `084716`, `100213`, `110216`, `120216`, and `130216` (the full `solana-online-20260914-` prefix applies to every suffix).

Reproduce the classification and fill-intent checks from this checkout:

```sh
PYTHONPATH=. ../venv/bin/python scripts/audit_market_observations.py \
  runs/market-attribution-01 reports/system-audit-market-20260914.json
../venv/bin/python -m pytest tests/test_solana_quotes.py tests/test_solana_execution_audit.py -q
```

The [JSON evidence](system-audit-market-20260914.json) contains each unavailable holding, original rejection reason, last receipt age, separate prospective valuation sensitivities, and SHA-256 hashes of every source event database, FX log, and decision ledger. The archived runtime inputs are ignored artifacts; copying only the public repository does not recreate them.

## What the historical records establish

All **565 fills** across the eleven episodes satisfy independent checks for the issued action/target, mint, subsequent receipt, requested direction, and quantity. Existing cash-flow attribution also reconciles cash, fees, token contributions, and total PnL. The unchanged total is **−$2,828.329599**, across eleven separate accounts rather than one compounded portfolio.

There are sixteen holdings with zero terminal recovery marks under the old protocol:

| Historical rejection | Holdings | Remaining acquisition basis |
|---|---:|---:|
| Stale trade | 9 | $1,677.08 |
| One-sided recent flow | 4 | $624.28 |
| Thin actual SOL reserves | 1 | $250.00 |
| Fixed $25 order would exceed 1% of reserves | 2 | $496.19 |

Seven of these holdings still had fresh reserve observations. Their entry filters failed; market observations were not actually absent. The original `tick_for` uses the same eligibility result for entries, exits, and portfolio marks. It also requires $2,500 of actual reserves because its legacy nominal order is $25, although fresh episodes already resize actual fills to the available 1% reserve allowance.

The 10:02 episode illustrates the effect: two holdings with **$496.19** of remaining basis received fresh trades **0.20–0.21 seconds** before the final row, but the fixed-order liquidity filter rejected their quotes. Another **$246.85** of missing basis in that episode had a genuinely stale trade. The seven entry-filter rejections are not proof that those holdings could have been fully sold, nor should their reported zero marks be relabeled realized losses.

The accounting audit previously accepted a forged recorded FLAT decision followed by the unchanged BUY fill: it checked that a decision timestamp existed, without checking the target's meaning. The new independent `verify_fill_intent` helper rejects this mutation. It also permits a LONG target to rebalance by selling when the later price makes its desired quantity smaller; action labels alone do not determine fill direction.

## Prospective quote protocol

`solana_observed_entry_exit_v2` in [solana_quotes.py](../paperlab/solana_quotes.py) preserves the historical quote function and exposes three distinct results:

- **Observed indicative price:** valid protocol, reserves, SOL/USD conversion, fresh receipt, and connected feed. An entry's two-sided-flow requirement does not erase this observation.
- **Entry eligibility:** existing launch, two-sided-flow, and minimum-reserve rules, followed by dynamic order capacity. A fixed $25 notional is no longer required merely to observe a price or make a smaller trade.
- **Exit eligibility:** a valid observed price and sufficient capacity for a minimum-sized simulated exit. One-sided flow or failure of the entry's five-SOL reserve floor alone does not prevent risk reduction.

Actual fills remain limited to **1% of actual SOL reserves**, with a $1 minimum notional and the existing execution maximum. The quote object reports both the full indicative liquidation mark and the net proceeds allowed by one capacity-limited fill. The latter's remainder-at-zero stress sensitivity is separate from the full mark. Neither figure is an executable quote or a promise that several sequential exits will obtain the same price.

The focused tests cover thirteen observation/entry/exit cases and nine action-fidelity cases: **22 passed**. They include a one-sided market, dynamic capacity below the old $2,500 threshold, partial exit from a thin pool, zero capacity, stale/future receipts, unavailable FX, protocol provenance, altered decision/mint/timing/quantity, and legitimate LONG rebalancing. Eligibility is chosen from the later receipt's requested change in inventory: a positive LONG target that requests a sale receives exit eligibility. Integration tests and deployment status are recorded in the main system audit.

## GSPO pilot verdict

The independent artifact audit of `runs/group-replay-20260914-01` passes: **592 trajectories, 7,696 rows, 1,624 fills, 32 training groups, and 48 gradient updates**. Source cohorts are chronological and mint-disjoint; the test cohort excludes mints present in the training archive. Orders fill at later receipts, missing quotes remain unavailable, and no terminal sale is fabricated.

This is a bounded mechanics experiment, not the currently deployed hourly Q learner. Its **900-parameter stochastic actor** trains above a frozen pristine fly encoder. Four training mints are reused for eight passes; twelve branches share each market tape and independently start at $1,000. They add alternative actions, not independent market observations or twelve continuously compounded accounts. Four held-out mints from the same day are insufficient for a reliable statistical learning claim.

Held-out average loss fell from approximately **$0.494 to $0.392**, with lower fees and turnover. Cash remained better. No native synapses learn in this pilot; no market-only ablation demonstrates that fly features add value. The GSPO audit validates recorded accounting, cohort separation, diagnostics, and checkpoint differences; it is not an exchange execution audit or a proof that every saved optimizer step was independently replayed.

## Remaining limits

- Confirmed public Pump/PumpSwap logs have reconnect gaps and no historical backfill or coverage guarantee. The bounded registry follows newly observed Pump launches and their canonical migrations, not all existing or newly formed Solana tokens.
- The online worker collects during its bounded fifteen-minute window. An hourly launch cadence does not itself provide continuous collection of every launch between windows.
- Reserve-derived mid prices, a synthetic spread, fixed slippage, and fixed fees are a simulator. Transaction failure, priority fees, token extensions/transfer restrictions, executable routing, reserve changes caused by our own order, and final settlement are not comprehensively modeled.
- Whole-position marks may exceed a single permitted exit's capacity. Missing observations, entry-filter rejection, partial liquidity, and genuine economic loss must remain separately reported.
- The eleven-episode market comparison remains descriptive. Differences in market windows, exposure, quote availability, and incomplete reward credit prevent attributing its trend to learning. Future checkpoints require identical unseen tapes and controls under one sealed evaluation protocol.

The next profitability gate is repeated forward matched evaluation plus execution calibration. Passing accounting tests or changing weights is not that gate.
