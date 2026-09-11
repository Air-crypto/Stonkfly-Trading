# Verification — September 11, 2026

Status: implementation tested locally; cloud sign-in/deployment pending. No real trades or sustained forward deployment.

## Mechanical checks

- 20 lab tests passed: timestamp leakage, news revisions, execution delay/fees, invalid targets, inventory, risk rejection, restart/deduplication/rollback, actual PPO weight updates, budget reservation and forward fills.
- All 7 upstream neural tests passed with the **full** dataset enabled, including sensory input, reward/aversive stimulation, learning, frozen weights and checkpoint restoration.
- Full connectome checksums verified: 166,700 neurons and 25,582,938 directed edges.
- FinBERT at the pinned revision encoded 45 current RSS headlines (25 CoinDesk, 20 Federal Reserve). Earlier lexical records remain separate revisions.
- A current-book news-on/news-off observation produced different fly input hashes. Both decoded HOLD. This verifies the input pathway, not predictive value.
- Modal Python definition imports successfully with its schedule disabled. Cloud image execution, actual invoice, provider budget configuration and restart on the cloud host remain unverified.

## Historical BTC smoke runs

Dataset: 1344 public 15-minute closed candles. SHA-256 `68f87aaf01903f9d257948c035619bdc017b39b38158ab65ee8dc05eea23f972`.

These experiments intentionally had **zero causally available news observations**: today’s collection occurred after the historical bars. Fees were 60 bps/side, slippage 10 bps and assumed spread 2 bps. Initial paper capital $1,000. All values below are before hosting.

| Compact test run | Net P&L | Return | Fills |
|---|---:|---:|---:|
| PPO seed 7 | $-3.58 | -0.358% | 6 |
| PPO seed 19 | $-8.12 | -0.812% | 17 |
| PPO seed 42 | $0.00 | 0.000% | 0 |
| cash | $0.00 | 0.000% | 0 |
| equal_cap_buy_hold | $-16.99 | -1.699% | 5 |
| trend | $-71.41 | -7.141% | 101 |

Each compact run used 8,192 PPO transitions, a 70/15/15 chronological split and a fixed checkpoint before testing. The test window was about two days. These are short debugging experiments; none beat cash after hosting. No strategy was selected using these results.

| Fly 32-observation window | Net P&L | Changed plastic edges | Runtime |
|---|---:|---:|---:|
| Learning | $-6.80 | 3,524 | 74.6 s |
| Frozen control | $-5.38 | 0 | 71.3 s |

The two fly runs use the same eight-hour market window; this is a different period from the compact test and is not a head-to-head profit ranking. Learning changed weights and performed worse than its frozen control in this sample. The 8-step synthetic fixture changed 2,990 edges; its results are not market evidence.

## Quantization

FP32: 992,337 bytes. INT8: 255,752 bytes. Sampled action agreement: 100% across 129 observations. Local per-observation latency: FP32 0.0160 ms; INT8 0.0151 ms.

INT8 is an optional artifact; the paper runner continues to use float32. Agreement on sampled cash-state observations does not prove identical sequential portfolio decisions. Native fly quantization is not attempted.

## Remaining evidence

Cloud deployment and billed-duration measurement; long forward collection; news-on/off training comparisons on truly available historical news; larger fixed-seed walk-forward study; robust cost stress and uncertainty analysis; checkpoint portability across cloud image changes. Paper fills still omit depth, queue position and market impact. Profitability is unproven.

## Reproduce the full neural test

```sh
STONKFLY_FULL_TEST=1 STONKFLY_DATA="$PWD/data/fly" PYTHONPATH="$PWD/vendor/stonkfly" .venv/bin/python -m pytest vendor/stonkfly/tests/test_neural.py -q
```

Local generated checkpoints and ONNX files are excluded from Git; commands in README recreate them. Committed metrics and paper ledgers are experimental evidence, not account statements.
