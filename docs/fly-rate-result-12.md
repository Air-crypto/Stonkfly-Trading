# Study 12: smaller updates did not improve trading

The registered learning-rate candidate failed development selection. All four
fly variants lost money in development and test, and no policy was promoted.
The complete [independent audit](../reports/fly-rate-result-12.json) covers
16 conditions, 372 neural observations, 18,600 recorded bins and 400 account
marks. These observations repeat 93 eligible asset/time slots across four arms.

Development used 06:00–08:00 UTC and test used 08:00–10:00 UTC on September 13,
2026. Each arm and phase starts with a fresh $1,000: two $250 sleeves, RAY and
STONK, plus $500 idle cash. These two pools do not represent every memecoin.
Fees, spread, slippage and adverse terminal inventory marks are included;
hosting is excluded from the table.

| Policy | Development equity | Test equity |
| --- | ---: | ---: |
| Pristine frozen fly | $982.51 | $978.38 |
| Trained frozen fly | $978.84 | $979.66 |
| Online fly, eta 0.001 | $981.73 | $981.14 |
| Online fly, eta 0.0001 | $979.25 | $980.41 |
| Cash reference | $1,000.00 | $1,000.00 |
| One capped entry per pool, then hold | $996.35 | $995.63 |
| Constant exposure target | $981.82 | $978.56 |

The [simple references](../reports/market-baselines-12.json) are descriptive,
not registered candidate arms. One-entry holding invests only $25 per pool
before fees, so these are not comparisons at equal exposure. Unlike study 11's
test window, both exposure references lost in both phases here.

![Audited accounts, fixed-fill P&L and weight movement](assets/fly-rate-comparison-12.png)

The large temporary equity dips are **missing-quote stress marks**, not observed
price crashes or executed liquidations. RAY supplied 24/24 observations in each
phase; STONK supplied 23/24 in development and 22/24 in test. Coverage passed the
registered minimum of 18 per pool. All terminal quotes were usable. Missing
slots remain missing, and feedback is zero across unpriced gaps.

## What failed

The candidate's smaller updates did not deliver better decisions. The sum of
per-observation weight-update L2 norms across both sleeves fell from 271.52 to
38.36 in development, and from 305.62 to 42.56 in test. This sum is a diagnostic,
not the norm of the net checkpoint change, optimizer loss or backprop gradient.

| Phase / online rate | Fixed-fill midpoint component | Execution and exit costs | Net P&L | Fills |
| --- | ---: | ---: | ---: | ---: |
| Development / 0.001 | −$4.90 | $13.37 | −$18.27 | 11 |
| Development / 0.0001 | −$4.69 | $16.06 | −$20.75 | 15 |
| Test / 0.001 | −$5.34 | $13.53 | −$18.86 | 14 |
| Test / 0.0001 | −$6.08 | $13.51 | −$19.59 | 12 |

In development, lower eta added about $2.69 of friction despite a slightly
better midpoint component. In test, friction was almost unchanged, while
the midpoint component worsened by about $0.75. Fewer fills alone did not help.
The midpoint component holds the actual fills and quantities fixed; it is
**not a zero-cost alternative strategy**. The [diagnostic report](../reports/fly-rate-diagnostics-12.json)
reconciles every mark, and all 105 fills have a preceding decision. Maximum
accounting identity error is below $3e-14.

This rejects the registered claim for this experiment. It does not prove that
eta 0.001 is a profitable strategy, or that a particular rate is universally
better. The next experiment should address the decision/readout and economic
cost of changing exposure, rather than continue a rate sweep on these results.

## Inspect the first changed action

The first action divergence in development RAY occurs at **06:15 UTC**, the
fourth neural observation. Both rates had made the same first three BUY
decisions. The saved price-image and reinforcement/feedback prefixes match
through this point; only the registered learning rate differs.

| Quantity | Eta 0.001 | Eta 0.0001 |
| --- | ---: | ---: |
| Left decoder rate | 36 Hz | 38 Hz |
| Right decoder rate | 42 Hz | 36 Hz |
| Gate spikes | 0 | 1 |
| Final action | HOLD | SELL |

The lower-rate gate neuron **10527** fires once in the **280–290 ms** bin; its
paired standard-rate recording has zero gate spikes. The final SELL is decoded
at 500 ms. Its delayed $25 sale fills at 06:18:57 UTC and incurs $0.692 in fees,
spread and slippage. A later BUY decision at 06:25 UTC fills at 06:29:06 UTC,
adding $0.683. The standard-rate controller holds at both decision times.
These costs are recorded, not an estimate of the benefit from deleting trades:
changing a decision would also change later inventory, feedback and learning.

![First changed action, raw decoder counts and selected edge weights](assets/fly-rate-case-12.png)

The [case record](../reports/fly-rate-case-12.json) includes all 50 bins for the
four decoder neurons and the selected connection's weights and `u/w`. Edge
8022240 is an independent inspected connection, not a proven causal path to the
gate. The live browser check matched its paired weight values and the one-bin
spike difference against the full recording.

Open the included paired recordings without neural computation:

```sh
python3 -m zipfile -e reports/fly-rate-views-12.zip runs/study-12-views
uv run --extra cloud python -m paperlab.debugger serve \
  --backend modal --out runs/study-12-views --port 8772
```

Visit the [first differing action](http://127.0.0.1:8772/?run=development-pool0-trained_online_low_eta&compare=development-pool0-trained_online_carry&step=3&bin=28&neuron=10527&edge=8022240).
The portable archive contains the displayed subset and paired curves. Full-neuron
lookup outside that subset requires the retained complete NPZ recordings.

## Reproduce the diagnostics

The [68-file evidence archive](../reports/fly-rate-evidence-12.zip) includes
the full audit report, per-condition audits, projections, completed receipts,
summaries, sealed plan and baseline report. It reproduces the account and
decision diagnostics from existing audited evidence. It omits the raw market
databases and neural recordings, so this is not a repeat of the full array audit.

```sh
python3 -m zipfile -e reports/fly-rate-evidence-12.zip runs/study-12-evidence
uv run --extra plots python -m paperlab.fly_rate_report \
  --root runs/study-12-evidence/recordings --audit runs/study-12-evidence/audit \
  --baselines runs/study-12-evidence/baselines.json \
  --out runs/study-12-diagnostics --plot
PYTHONHASHSEED=0 uv run --extra dev --extra plots python -m pytest tests/test_fly_rate_report.py -q
```

Use a new output directory. With the full downloads and audits retained, add
`--case` to export the case directly from the raw arrays. The
[full download/audit instructions](fly-rate-market.md#download-audit-and-inspect-the-completed-comparison)
remain separate. No command above constructs or propagates a local brain.

The [completion record](../reports/fly-rate-completion-12.json) preserves archive
hashes, cloud closure and budget evidence. All 67 frozen execution files remained
unchanged. The temporary study app was stopped at 12:31:50 UTC after all captures
completed and the worker was verified idle; the separate paper-lab deployment
was retained. The study's conservative compute ledger accounted for **$0.90664**
against its $3 allowance. That excludes a claim about the final provider invoice.
