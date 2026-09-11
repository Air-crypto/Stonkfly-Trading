# Paper experiment

Source: **historical_candle_proxy**. 8,192 PPO transitions, 247,780 parameters. Seed 42.

| Test policy | Net return | Max drawdown | Fills | Fees |
|---|---:|---:|---:|---:|
| ppo | 0.000% | 0.000% | 0 | $0.00 |
| cash | 0.000% | 0.000% | 0 | $0.00 |
| equal_cap_buy_hold | -1.699% | 2.131% | 5 | $2.99 |
| trend | -7.141% | 7.194% | 101 | $56.30 |

Research run, not a profitability certificate.
Single chronological split and one seed; repeat with preregistered windows and seeds.
Paper fills omit queue position and market impact; historical candle fills are proxies.
News must have been collected at the time. Zero news in old history is intentional.
Test is a one-shot audit. Repeatedly optimizing against it invalidates the holdout.
