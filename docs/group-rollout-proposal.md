# Proposed group rollout experiment (not deployed)

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
