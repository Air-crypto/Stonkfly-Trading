# Frozen forward experiment launch

Started September 17, 2026, at 15:58:27 Chicago time. The 72-hour endpoint is
September 20 at 15:58:27 Chicago time, subject to the budget, storage, and
worker-health guards. See the [saved launch evidence](frozen-forward-launch-20260917.json)
and [experiment protocol](../docs/frozen-forward.md).

- Selected checkpoint: `solana-online-20260916-005626`.
- New continuous paper capital: $1,000; learning disabled, zero exploration.
- Comparison accounts: always-long on matched decision opportunities, and cash.
- Dedicated prepaid compute allowance: $17.50 inside the original $100 authorization.
- Current trainer and frozen replay apps remain deployed; their existing spending
  guards will see this reservation and may pause them sooner.
- The obsolete `fly-paper-lab` app was stopped to free schedule capacity.

The first cloud session completed 14 neural observations. Both native and readout
weights were unchanged; inherited readout updates stayed at 4,853, with zero new
backprop updates. Both account fill reconstructions passed. The trained policy
made no fills and retained $1,000. This is startup evidence, not profitable-trading
evidence.

The scheduled coordinator automatically started session two. Its complete opening
state equals session one's saved continuation, including the always-long account's
four holdings and $12.3456667 cumulative fees. A subsequent snapshot showed 23
cumulative neural observations and the trained account still flat at $1,000.

The always-long endpoint valuation included unavailable quotes marked at zero.
That is not a realized loss or evidence that the trained strategy outperformed an
executable baseline. Future reporting must separate missing marks, inventory,
realized proceeds, fees, and the one-fill capacity-stress scenario.

Validation: 93 relevant tests passed, including 14 forward-run tests. Sealed replay
source files were not modified. Coordinator guards also stop on an ambiguous or
corrupt result and a call exceeding its dispatch/execution allowance. GitHub CI is
separate from these local checks and was still running when the launch was recorded.

Runtime CPU/memory estimates use a 2x margin over the standard rates listed on
[Modal's pricing page](https://modal.com/pricing), checked September 17. Standard
preemptible workers use one core and 8 GiB. Preemptions pause for review; they do
not silently reset the account or erase a partial session.
