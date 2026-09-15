# Unseen evaluation and result reconciliation

The September 15 status check found three completed policies in the 40-policy
prospective cohort `evaluation-1789504902`. They used the same later tape,
`solana-online-20260915-204717`:

| Policy | P&L from $1,000 | Native observations |
| --- | ---: | ---: |
| Preselected latest checkpoint, `solana-online-20260915-194714` | $0.00 | 123 |
| Untrained | -$358.34 | 123 |
| Cash | $0.00 | 0 |

The trained policy chose 115 hold actions and 8 buy actions, with **zero fills**.
It did not demonstrate profitable trading. Both neural policies retained their
original weights during evaluation. The untrained policy's results depend on
the existing paper execution and valuation assumptions.

At 22:51 UTC, training had completed 44 windows and accumulated 5,717 native
observations and 4,486 readout updates. Its latest training episode had 124 native
observations, 96 readout updates, 7 fills, and ended at $996.26 (-$3.74).
The shared worker estimate was $21.52983, not a provider invoice.

## Pause and repair

Evaluation was disabled with `MissingDurableEvaluationResult` on cash call
`fc-01M2KEYV6K39EEMFD0KWF1ZJJD`. The returned completed result and the persisted
`completed.json` matched exactly. The dispatcher refreshed its mounted volume
only at startup, then polled the worker and checked a potentially older filesystem
snapshot. A completion between reload and polling can therefore falsely trigger
this pause. Added a reload immediately after a completed call returns.

Added an explicit recovery operation under the coordinator lease and existing
dispatcher budget. It only accepts this exact missing-file error, verifies the
expected call ID and persisted owner, matches the complete returned result to the
saved file and sealed policy/tape, and preserves the original control in a
recovery record. Other pauses and mismatches remain blocked. It never reruns or
overwrites the completed policy.

Deployed the dispatcher repair and ran recovery call
`fc-01M2KMDC7SF4G40RN713JGB0K8`, which returned
`verified_result_reconciled` for the cash policy. The frozen evaluator, test tape,
checkpoint cohort, scoring rules, and $100 monthly authorization remain unchanged.

Validation: **75 targeted regression tests passed**. Coverage includes completion
appearing only after reload, pending calls, exact-result recovery, owner/result
mismatches, protected audit pauses, missing files and repeat recovery rejection.
The companion JSON records the subsequent cloud state.

At 22:54:53 UTC the evaluator was enabled with no error and waiting for a training
gap. Training run `solana-online-20260915-225431` was dispatched as
`fc-01M2KMF2PXQNBZJSW4XC1WBV6N` and held the shared worker lease. The worker ledger
was $21.84898 including the active reservation. Three of 40 evaluation policies
were complete; the remaining scores were not yet available.
