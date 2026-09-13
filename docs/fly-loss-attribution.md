# Study 11 loss breakdown

All eight aggregate outcomes lost money. The new breakdown explains those
results using the original fills: seven had unfavorable midpoint price exposure,
and every account paid execution costs. Test reset had a small favorable
midpoint component, which its costs outweighed.

![Study 11 loss decomposition](assets/fly-loss-attribution-11.png)

Each row starts with $1,000: two $250 pool sleeves and $500 idle cash. Dollar
amounts below are summed across the sleeves. Costs include simulated paid fees,
spread and slippage at fills, and the estimated cost of liquidating remaining
inventory. Hosting is excluded.

| Phase / variant | Midpoint component | Total costs | Recorded net P&L | Fills |
| --- | ---: | ---: | ---: | ---: |
| Development / pristine frozen | −$43.24 | $10.03 | −$53.27 | 9 |
| Development / trained frozen | −$43.75 | $13.39 | −$57.14 | 12 |
| Development / online carry | −$26.80 | $8.75 | −$35.55 | 7 |
| Development / online reset | −$16.22 | $11.75 | −$27.97 | 12 |
| Test / pristine frozen | −$1.72 | $11.94 | −$13.66 | 10 |
| Test / trained frozen | −$9.15 | $14.49 | −$23.64 | 14 |
| Test / online carry | −$14.30 | $13.57 | −$27.87 | 16 |
| Test / online reset | +$2.48 | $17.55 | −$15.07 | 20 |

Test reset's $17.55 consists of $5.63 paid fees, $2.24 spread at fills, $4.48
slippage at fills and $5.19 terminal exit allowance. Its midpoint component was
$4.20 better than pristine, but costs were $5.61 greater. It therefore finished
$1.41 worse. Online carry suffered both unfavorable exposure and costs.

All terminal quotes were usable. Zero-valued missing inventory was therefore
not a direct component of these final losses. Earlier quote gaps still affected
marks and potentially decisions and learning; this analysis does not remove them.

## Meaning and limits

The midpoint component holds actual filled quantities fixed, accounts for every
purchase/sale at its observed midpoint, and values final inventory at midpoint.
Subtracting paid fees, fill spread, fill slippage and exit allowance reconstructs
the recorded result. At unavailable marks, stale reference inventory is explicitly
deducted in full; it is not treated as liquidatable.

This is an accounting identity, **not a zero-cost strategy backtest**. Changing
costs would change sizing, risk limits, feedback and later actions. The midpoint
component does not estimate the return such a different policy would earn.
Prices and costs remain the original indicative-price simulation assumptions,
not verified executable DEX fills. The test phase is now diagnostic historical
data; it cannot serve as an untouched holdout for a change motivated by it.

The evidence suggests a cost-aware abstention/readout experiment if study 12
also shows extra trades consuming any directional benefit. Study 12 remains
unchanged. A subsequent rule needs a prospective registration, fresh development
and test windows, equal-cost controls, and a net-outcome acceptance gate. No
variant has been selected or promoted from this postmortem.

## Verification and reproduction

The [attribution report](../reports/fly-loss-attribution-11.json) retains every
account mark and component. The original 16 receipts, audits, outcomes and
selection were verified, and all 16 ledgers replayed. All **400 marks** reconciled;
the maximum numerical residual was **$3.44 × 10⁻¹⁴**. Eleven tests cover
hand-calculated open positions and partial sales, zero fees/slippage, missing and
stale quote recovery, and corrupted fill/account evidence. No neural observations
or cloud submissions were needed.

Install the project and plotting extra using the README instructions, then:

```sh
python3 -m zipfile -e reports/fly-online-evidence-11.zip runs/study-11-evidence
uv run --extra plots python -m paperlab.fly_loss_attribution \
  --root runs/study-11-evidence --out runs/study-11-losses --plot
uv run --extra dev python -m pytest tests/test_fly_loss_attribution.py -q
```

Reuse an already extracted, unchanged evidence directory when available. Choose
a new output directory to preserve prior results. The command verifies evidence
before publishing and needs no graph download, credentials or model compute.
The [figure record](../reports/fly-loss-attribution-11-figure.json) identifies the
published report, analysis source and PNG/SVG hashes.
