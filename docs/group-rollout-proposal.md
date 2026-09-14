# Group rollout experiment

The bounded pilot is implemented in `paperlab/group_replay.py` and
`group_replay_cloud.py`. It is an unscheduled, separate paper experiment. The live
Q learner and native plasticity continue under their existing coordinator.

The [completed pilot and all twelve sample alternatives](../reports/group-replay-01.md)
include measured losses, gradients, weights, fills and held-out comparisons.
Average held-out loss was lower after training, but cash still performed better;
the policy was not promoted. Reproduce the local accounting audit with
`python -m paperlab.group_replay_audit runs/group-replay-20260914-01` after downloading
that immutable run directory from the Modal `fly-paper-lab-state` volume under
`/group-replay/group-replay-20260914-01`.

The first pilot preregisters up to four first-eligible launches from
`solana-online-20260913-230259` for training and up to four later launches from
`solana-online-20260913-232717` for testing, excluding every mint in the earlier
archive. Each episode has a fixed 60-second horizon at five-second intervals;
unavailable quotes and losing episodes remain in the data. These are retrospective
pilot windows, not a new sealed future holdout. No settings are tuned on test PnL.

The full pristine fly encodes causal market frames with zero reward and frozen
synapses, resetting between mints. A 900-parameter stochastic actor receives these
features plus branch-specific account state and remaining time. Twelve trajectories
form each group; eight passes use two Adam updates per group. The objective uses
length-normalized sequence probability ratios, group-normalized rewards, clipping,
entropy regularization and a KL guard. Zero-variance groups skip training.

Each branch starts with $1,000 simulated cash and at most 0.25% target exposure.
Actions are HOLD, EXIT, 0.15% target and 0.25% target. Orders are capped at $2.50
notional, with a 1.25% fee per side, 1% slippage per side and the indicative
one-percent bid/ask spread. The original stricter $25-based liquidity admission
guard is retained. Execution requires a later usable receipt within fifteen seconds.
Terminal inventory is conservatively marked with exit costs, not automatically sold;
an unavailable terminal quote values that inventory at zero.

Reward is liquidation PnL minus 25% of maximum dollar drawdown and 0.1% of mean
marked dollar exposure. Outputs include the pre-training manifest, causal feature
cache, initial/trained actor checkpoints, all training and evaluation trajectories,
losses, per-layer gradient norms, weight deltas, sequence ratios, entropy and KL.
Matched evaluations compare initial and trained actors, cash and buy-and-hold.
The current Q-head comparison is deferred because its reward-conditioned encoder,
account inputs and action/risk settings differ; it would not be a matched comparison.

The cloud job acquires the existing shared worker lease, reserves its full timeout
within the $85 worker allocation of the $100 monthly authorization, has no automatic
retries, and cannot overwrite a run. A timeout/failure retains the reservation. Native
encoding is blocked outside Modal. There is no automatic promotion or recurring
GSPO training; the pilot measures mechanics and indicative outcomes, not profitability.

## Design rationale

The user proposed twelve alternative action sequences for the same new-token market
segment. This is a reasonable replay experiment after the live paper accounting and
learning-feedback checks pass. It is not twelve independent market observations.

[GSPO](https://qwenlm.github.io/blog/gspo/) uses group-relative sequence rewards and
length-normalized sequence likelihood ratios with a clipped policy objective. It
does not simply copy the winning trajectory. Applying it here would be a trading
adaptation, not a validated use of the language-model algorithm.

1. Freeze a behavior policy for each rollout batch. Sample twelve action sequences
   from the same initial portfolio state, using only information available at each
   recorded receipt. Actions cover flat/hold/exit and discrete position-size levels.
2. Replay the same chronological receipts for every sequence with independent
   portfolio state and identical starting cash, horizon, fees, spread, slippage,
   admission rules, delayed-fill rules, and risk limits. Include the cash baseline.
   Missing quotes never imply an executable sale; simulation fills remain indicative.
3. Score net liquidation return with an explicit downside/exposure penalty. Compare
   equal starting capital and risk limits so the reward does not just favor a
   larger bet. Preserve failed and losing rollouts as well as winners.
4. Normalize within each group; skip the policy update for tied/near-zero-variance
   rewards. Store behavior log probabilities and masks. Apply a bounded, clipped
   sequence policy update, with entropy/KL diagnostics and held-out rollback gates.
5. Begin with a small probabilistic actor head and a frozen, market-only fly feature
   encoder. The current Q-learning head is not already a GSPO actor. Its portfolio
   features must be recomputed separately in every branch. Fly features can only be
   shared if they are causal and independent of branch actions/rewards; do not reuse
   reward-conditioned neural traces from the live agent as counterfactual features.
   This first phase would not update the native fly synapses using GSPO.
6. Keep all segments of a mint in one split and reserve later, unseen launches/dates
   before tuning. Compare against the current Q learner, frozen policy, simple
   baselines and cash on equal data and compute. Bootstrap uncertainty by token/day,
   not by treating twelve correlated branches as twelve independent samples.

Twelve head-level simulations should be benchmarked under a small explicit cloud
reservation first. Do not run twelve full fly brains every five seconds or add an
unbudgeted recurring job. The total monthly cloud authorization remains $100.
Extra replay increases action experience, not market diversity, and cannot repair
missing execution data or establish profitable live trading by itself.
