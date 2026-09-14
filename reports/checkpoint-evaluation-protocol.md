# Retained checkpoints and prospective frozen evaluation

Deployed Modal app: `fly-paper-checkpoint-eval`, environment `main`.

Every completed hourly training window retains its completion/state record and checkpoint provenance. Windows with native observations save `fly-final.npz` and `head-final.pt` (including optimizer and RNG); no-inference windows may explicitly reference unchanged parent checkpoints. The registry records verified SHA-256 hashes under `/checkpoint-eval/checkpoints/`; it does not copy large native checkpoints, overwrite earlier records, or delete them. Subsequent scheduled checks register newly completed episodes. Any changed file fails evaluation.

The original v1 prospective batch was `evaluation-1789398507`: 12 retained episode checkpoints plus pristine/untrained, cash, and always-long controls. At the full-system audit pause it had no tape or policy attempts. It is being superseded without rewriting its plan. The corrected `frozen_common_opportunities_v2` protocol requires a new sealed cutoff and a subsequent v2 market window. The first *completed* online window whose start is strictly after that cutoff supplies the common tape; policy results are not yet claimed here.

## Schedule and costs

The cloud coordinator checks at minutes 7, 22, 37 and 52 each hour. Subsequent batch cutoffs are eligible daily at 00:00 UTC; a new batch waits for the prior comparison to finish. Every retained checkpoint available before that batch's cutoff is included. Large cohorts can take multiple days; no scores are promised by a particular hour.

Each policy is one bounded, non-retrying, maximum-20-minute worker call. It uses the existing shared worker lease and $85 worker allocation within the authorized $100 total monthly cap, with the existing 25% reservation margin. It launches only when the live trainer is idle and its next scheduled window is more than 1,250 seconds away. Failed or ambiguous attempts require review; no prior attempt is overwritten. A budget refusal pauses this evaluator for review. Live training pauses are respected. These jobs run on Modal without a laptop; assistant notifications require the local app.

## Controlled comparison

- Fresh $1,000 paper portfolio for each policy; $250 remaining acquisition-cost limit per token and 1% observed SOL reserve per-fill limit.
- Identical archived market events, contemporaneous FX, quote guards, starting cash, fees, spread, slippage, missing-quote zero marks, and delayed-fill rules.
- The v2 tape records the exact applied-event cursor, observation cutoff and FX availability. Replay excludes events received but not yet applied at the snapshot, respects failed FX fetches, and preserves the recorded later decision issuance time for common inference latency.
- `solana_observed_entry_exit_v2` separates observable marks, new-entry guards and risk-reducing exit guards. Full indicative marks and next-fill capacity sensitivity are reported separately; neither proves exchange executability.
- Identical recorded token-attention opportunities. This isolates conditional buy/sell behavior; it **does not measure token selection quality** or a completely independent end-to-end trader.
- Each earlier checkpoint's full fly and Q weights are frozen. Neural transient state is standardized without deleting learned weights. Q evaluation is greedy (epsilon zero) for every checkpoint; the original live learner used exploration.
- Freezing sets the native weight-change input to zero. This is a documented difference from the online-training feature distribution; the evaluation does not claim otherwise.
- The untrained reference uses pristine native weights and a seed-13 random Q head. Its initial native/head files are saved with its evaluation. It is distinct from a previously trained pre-episode checkpoint.
- The always-long control requests LONG at every common opportunity, subject to the same cash and liquidity limits; it is not a frictionless buy-and-hold index.
- Source, tape and checkpoint hashes are verified. Training ending at or after the evaluation tape start is rejected. No checkpoint receives updates from evaluation.

Each policy records step-by-step decisions, inputs/Q values, fills and account marks, fees, drawdown, native inference count, opportunity coverage/exclusions, and an independent recorded-fill accounting audit. A final comparison includes every policy, preserving all losses. Frozen Q/native weights are checked for exact equality. No evaluation resets or modifies the actual live training account. Multiple checkpoints on one tape are not independent market samples; checkpoint selection requires a later untouched confirmation tape.

## Verification performed

The initial v1 implementation passed 33 focused tests and an archived cash/always-long mechanics smoke. Those checks did not cover the defects found in the full-system audit and do not validate prospective trading performance. The v2 evaluator/quote focused suite passed 22 tests at implementation, including 9 evaluator cases; the native cloud smoke and corrected prospective deployment require their separate observed verification. See [the evaluation audit](system-audit-evaluation-20260914.md) for corrections and remaining interpretation limits.
