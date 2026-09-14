# Retained checkpoints and prospective frozen evaluation

Deployed Modal app: `fly-paper-checkpoint-eval`, environment `main`.

Every completed hourly training window already retains `fly-final.npz`, `head-final.pt` (including optimizer and RNG), and its completion record. The new registry records their SHA-256 hashes under `/checkpoint-eval/checkpoints/`; it does not copy large native checkpoints, overwrite earlier records, or delete them. Subsequent scheduled checks register newly completed episodes. Any changed file fails evaluation.

The first prospective batch is `evaluation-1789398507`: 12 retained episode checkpoints plus pristine/untrained, cash, and always-long controls, for 15 policies. The cohort cutoff is Unix time `1789398507.2657042`. The first *completed* online window whose start is strictly after that cutoff supplies the common tape. No result has been reported for this future batch yet.

## Schedule and costs

The cloud coordinator checks at minutes 7, 22, 37 and 52 each hour. Subsequent batch cutoffs are eligible daily at 00:00 UTC; a new batch waits for the prior comparison to finish. Every retained checkpoint available before that batch's cutoff is included. Large cohorts can take multiple days; no scores are promised by a particular hour.

Each policy is one bounded, non-retrying, maximum-20-minute worker call. It uses the existing shared worker lease and $85 worker allocation within the authorized $100 total monthly cap, with the existing 25% reservation margin. It launches only when the live trainer is idle and its next scheduled window is more than 1,250 seconds away. Failed or ambiguous attempts require review; no prior attempt is overwritten. A budget refusal pauses this evaluator for review. Live training pauses are respected. These jobs run on Modal without a laptop; assistant notifications require the local app.

## Controlled comparison

- Fresh $1,000 paper portfolio for each policy; $250 remaining acquisition-cost limit per token and 1% observed SOL reserve per-fill limit.
- Identical archived market events, contemporaneous FX, quote guards, starting cash, fees, spread, slippage, missing-quote zero marks, and delayed-fill rules.
- Identical recorded token-attention opportunities. This isolates conditional buy/sell behavior; it **does not measure token selection quality** or a completely independent end-to-end trader.
- Each earlier checkpoint's full fly and Q weights are frozen. Neural transient state is standardized without deleting learned weights. Q evaluation is greedy (epsilon zero) for every checkpoint; the original live learner used exploration.
- The untrained reference uses pristine native weights and a seed-13 random Q head. Its initial native/head files are saved with its evaluation. It is distinct from a previously trained pre-episode checkpoint.
- The always-long control requests LONG at every common opportunity, subject to the same cash and liquidity limits; it is not a frictionless buy-and-hold index.
- Source, tape and checkpoint hashes are verified. Training ending at or after the evaluation tape start is rejected. No checkpoint receives updates from evaluation.

Each policy records step-by-step decisions, fills and account marks, fees, drawdown, native inference count, and an independent recorded-fill accounting audit. A final comparison includes every policy, preserving all losses. Frozen Q/native weights are checked for exact equality. No evaluation resets or modifies the actual live training account.

## Verification performed

33 focused tests passed, including immutable hash detection, exclusion of overlapping windows/later checkpoints, delayed fills, cash accounting, frozen native fixtures and zero Q updates. An archived market smoke check passed for cash and always-long controls. Those retrospective smoke outputs are mechanics checks, not prospective policy performance or evidence of profitable learning. The deployed coordinator registered 12 checkpoints and persisted the first cohort while waiting for an unseen market window. Full native evaluation remains to be observed on that future tape.
