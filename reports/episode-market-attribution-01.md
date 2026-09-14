# Market conditions versus learning: eleven completed paper episodes

The recent reduction in marked losses is **not explained by a general improvement in this measured launch cohort**, but it does **not establish that the learner caused the improvement**. Missing-quote inventory marks, selected-token outcomes, and changing exposure remain substantial confounders. No frozen-policy counterfactual was run.

## Market comparison

Reconstructed archived Solana events for eleven windows, including launches the trader did not select. The cohort comprises 91 launches reaching the existing `tick_for` eligibility criteria early enough to observe another 70 seconds. For each, measure indicative USD mid-price change from its first eligible quote to the first eligible quote 60–70 seconds later. Sixty-four have such endpoints; 27 do not. These are bounded observed launches, not all Solana memecoins.

| Window start UTC | Trader marked PnL | Mean 60s market return among observable endpoints | Endpoints / eligible launches | Simulated realized PnL | Basis of inventory without final quotes |
|---|---:|---:|---:|---:|---:|
| 10:02 | -$603.32 | +21.19% | 4 / 7 | +$42.99 | $743.04 |
| 11:02 | -$209.23 | +1.12% | 4 / 6 | -$77.64 | $213.25 |
| 12:02 | -$187.07 | +3.34% | 6 / 10 | -$219.35 | $125.63 |
| 13:02 | -$44.02 | -0.51% | 7 / 11 | +$86.04 | $250.00 |

The observable-endpoint statistic is survivor-conditioned. Including all cohort members and assigning missing endpoints a -100% stress return gives -30.75%, -32.59%, -38.00%, and -36.69% respectively. That sensitivity check also shows no improving broad backdrop. Neither measure is an executable portfolio return: there is no simulated entry latency, allocation, fill size, or fees in this descriptive market statistic. Individual entry times differ, and a 60-second launch return is not the trader's full 15-minute holding horizon. SOL/USD moved only -0.05%, -0.15%, +0.22%, and +0.13% in these windows.

## Accounting and selection confounders

Across all eleven independent accounts, marked PnL is -$2,828.33, split into -$363.15 from simulated sales and -$2,465.18 from unsold inventory marks. Every episode's fill cash flows, paid fees, cash balance, and summed token PnL were independently reconciled.

The realized series in the table is not steadily improving, despite steadily improving total marked PnL. The 10:02 episode held $743.04 of remaining acquisition basis without usable final quotes. The system marks such holdings at zero; that is neither a confirmed sale nor proof the tokens became worthless.

As a deliberately non-executable sensitivity check, carrying the latest previously observed usable quotes for those missing holdings changes the last four marked results to -$192.65, -$39.95, -$101.89, and +$240.30. These stale values are not profits and have not replaced any stored account marks. This demonstrates how heavily quote availability affects the apparent trend, while leaving token selection, timing, and learning unresolved.

Across eleven episodes, Pearson correlation of marked PnL with unavailable-inventory acquisition basis is -0.76; correlation with the observable-endpoint market mean is +0.23 and with the missing-endpoint stress mean is +0.18. These are descriptive small-sample associations, not causal or statistically validated effects. The missing-basis relationship is partly mechanical by construction.

## What remains to establish learning

Run an earlier frozen checkpoint, the current frozen checkpoint, and simple cash/hold controls on identical subsequent market tapes, under identical capital, quote availability, execution costs, and random seeds. Preserve the full native fly state and Q policy for both checkpoints; the existing GSPO pilot is a different actor and cannot substitute. Evaluate forward dates and report per-mint results, missing-quote sensitivity, realized outcomes, drawdown, turnover, and uncertainty. Comparing successive online episodes alone cannot identify a causal learning effect.

This analysis performs no new native inference, cloud training, deployments, or account changes. Reproduce with:

```sh
PYTHONPATH=. python scripts/analyze_episode_markets.py runs/market-attribution-01 reports/episode-market-attribution-01.json
```

The JSON contains all episode metrics, cohort mint identifiers, endpoint coverage, and missing-inventory quote ages. The SQLite sources and ledgers remain ignored runtime artifacts downloaded from completed Modal runs. Replay uses event receipt timestamps and archived contemporaneous FX; no later market data is introduced into cohort eligibility.
