# Attribution of the first three fresh paper episodes

The reported **+$83.0873** is the sum of three independent $1,000 accounts, not a compounded portfolio return. It includes **+$35.6118 from simulated sales** and **+$47.4755 from unsold inventory marks**. Paid fees of $82.7325 are already included; cloud costs are excluded.

| Episode (UTC run ID suffix) | Net marked PnL |
|---|---:|
| 20260914-021337 | +$30.8416 |
| 20260914-031716 | -$112.7578 |
| 20260914-041721 | +$165.0035 |

The dominant winner was `CeSFzoAqSMXMgodeLzrTMe3V5MdV5Nhinehedxohpump`: **+$434.6070**, comprising $84.4121 realized in simulated sales and $350.1949 in unsold appreciation. Every other token contribution combined was **-$351.5197**. This outcome is concentrated in one winner; it is not evidence of a repeatable learned advantage.

The largest negative contribution was `42faVxmVo85HFzsuMEJg2spDZV8Qk6haDWM7pCnLpump`: **-$255.4087**. Its remaining acquisition basis was approximately $250 and its final quote was unavailable, so the system marked remaining inventory at zero. This is a conservative missing-data mark, not proof that the token actually became worthless or that a sale occurred. `dRQvsszWkyBbgzWyjVmj8xX9fWew5aBJA5i9T4Wpump` also had no final quote.

## Method and evidence

`scripts/attribute_paper_episodes.py` independently reconstructs each token's cash flow from recorded simulated fills, checking per-token cash flows, total cash, paid fees, and the sum of marked contributions. Realized PnL uses average acquisition cost including entry fees and sale proceeds after exit fees. Unsold PnL is marked liquidation value minus remaining acquisition basis. Available marks use recorded bid less 1% slippage and 1.25% exit fees; missing quotes use zero. These are indicative paper valuations, not verified executable proceeds.

The full per-token values and source ledger paths are in `fresh-episodes-pnl-attribution-01.json`. Downloaded ledgers are ignored runtime artifacts. Reproduce with:

```sh
python scripts/attribute_paper_episodes.py \
  runs/episode-continuation-verification/first-decisions.jsonl \
  runs/episode-continuation-verification/solana-online-20260914-031716/decisions.jsonl \
  runs/episode-continuation-verification/third-decisions.jsonl \
  --output reports/fresh-episodes-pnl-attribution-01.json
```

## Distinct from the twelve-branch replay

These hourly episodes used native fly/Q training. The separate GSPO-inspired pilot trained a 900-parameter actor over frozen fly features, with 12 independent branches per 60-second group and a historical $2.50 order cap. Branch accounts reset; sample IDs do not define persistent cross-group portfolios. Its held-out mean loss decreased from $0.4937 to $0.3923, while cash returned zero. Four test mints on one day and fewer trades/fees do not establish profitable learning or a fly-specific advantage. See `group-replay-01.md`.
