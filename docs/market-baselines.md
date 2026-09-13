# Simple market-exposure references

On study 11's test window, both simple exposure references finished above their
starting capital while every fly variant lost money. Development was different:
both exposure references lost, and constant exposure lost more than every fly
variant. This strengthens the need to test decision timing and exposure as well
as transaction costs. It does not establish a successful replacement strategy.

| Reference | Development equity | Test equity | Test fills | Test fees |
| --- | ---: | ---: | ---: | ---: |
| Cash | $1,000.00 | $1,000.00 | 0 | $0.00 |
| One capped entry, then hold | $984.94 | $1,002.33 | 2 | $0.63 |
| Constant exposure target | $930.02 | $1,007.61 | 32 | $4.17 |
| Pristine frozen fly | $946.73 | $986.34 | 10 | $3.08 |
| Trained frozen fly | $942.86 | $976.36 | 14 | $4.34 |
| Online carry fly | $964.45 | $972.13 | 16 | $3.95 |
| Online reset fly | $972.03 | $984.93 | 20 | $5.63 |

These baselines were added **after study 11 was completed and after study 12
collection began**. They are descriptive references, not preregistered selection
arms. They do not alter either study's selection rule or promote a model.

Each phase starts fresh with $1,000: two $250 sleeves and $500 idle cash. All
policies use the same recorded price series, quote eligibility, five-minute
decision grid, subsequent-quote execution, $25 maximum order, 50% maximum
sleeve exposure, loss stop, spread/slippage and fees. Final inventory uses the
same adverse liquidation mark; unavailable inventory is stress-marked at zero.
No baseline reads future prices or headlines. Hosting is excluded throughout.

- **Cash:** never places an order.
- **One entry, then hold:** requests the maximum sleeve exposure at new usable
  quotes until its first purchase fills, then holds that fixed quantity. With
  the original $25 order cap, this study invests only $25 per pool before fees.
  It is a low-exposure reference, not a fully invested buy-and-hold portfolio.
- **Constant target:** requests 50% sleeve exposure at every new usable quote.
  Orders remain capped and execute later. It can buy or sell while rebalancing;
  it does not anticipate the subsequent quote.

Exposure differs across these policies and the fly accounts, so a difference in
return is not an isolated causal effect of learning. Nor is fill count alone a
measure of friction: constant target's 32 test fills traded **$333.43** in total,
versus reset's 20 fills trading **$450.34**. Position timing, size, costs and
terminal inventory all matter. The next experiment should evaluate net outcomes,
not optimize for fewer spikes, fewer fills or smaller weight updates alone.

## Study 12

The same references on [study 12's sealed prices](../reports/market-baselines-12.json)
lost in both phases: one-entry holding ended at $996.35 in development and
$995.63 in test; constant target ended at $981.82 and $978.56. All four fly
variants also lost. These remain descriptive comparisons with differing exposure,
not new selection arms. See the [full result and diagnosis](fly-rate-result-12.md).

## Reproduce

The [baseline report](../reports/market-baselines-11.json) retains all 300
reference account marks, executions, coverage and input provenance. Its source
is [market_baselines.py](../paperlab/market_baselines.py). It reads existing study
11 audits or independently reconstructs study 12 prices from its sealed database.
It does not construct or propagate a neural model.

```sh
python3 -m zipfile -e reports/fly-online-evidence-11.zip runs/study-11-evidence
uv run python -m paperlab.market_baselines \
  --root runs/study-11-evidence --out runs/study-11-baselines
PYTHONHASHSEED=0 uv run --extra dev python -m pytest tests/test_market_baselines.py -q
```

Reuse existing unchanged extracted evidence, and choose a new output directory.
For study 12, wait until its sealed `plan.json` and `universe.db` have been
downloaded, then run the same command with `--root runs/rate-results-12` and a
new output path. A missing or altered snapshot fails before publication. This
produces descriptive baselines; the full neural/account audit and registered
development selection remain separate requirements.

Eleven tests cover hand-calculated terminal value, delayed fills, missing and
repeated quotes, fixed-quantity holding versus rebalancing, order/exposure caps,
loss stops, exclusion of future prices, sealed study 12 wiring and corrupted
price archives. Study 12 wiring tests use synthetic data and are not market results.
