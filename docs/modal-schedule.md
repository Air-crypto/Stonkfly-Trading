# Modal repeating schedule

> User pause, September 19, 2026: the entire experiment is paused until the user explicitly requests resumption. All eight remaining deployed Modal apps in the only environment (`main`) were stopped; verification found zero active apps and zero workers. The frozen forward app was already stopped. The local training monitor is also paused. Saved volumes, checkpoints, account ledgers and unresolved session 43 evidence are preserved. Do not redeploy, repair-and-resume, or restart on a budget reset without a new user instruction. The schedule below is historical configuration, not an active schedule.

Deployment inventory verified September 16, 2026, around 00:50 Chicago time.
This describes the observed configuration, not an assurance that every future
scheduled invocation succeeds. The [deployment snapshot](../reports/modal-schedules-20260916.json)
records schedules returned by Modal's app-layout and function lookup APIs.

September 17 update: the obsolete `fly-paper-lab` app was stopped to free schedule
capacity. The new [frozen forward test](frozen-forward.md) runs independently with
a prepaid budget and continuous paper accounts. Its coordinator checks each minute;
a three-minute verification session precedes rolling 15-minute sessions for up to
72 hours. Training/evaluation keep their existing schedules and budget guards.

## Active training and evaluation

| Job | Repeating cadence | Work |
| --- | --- | --- |
| `fly-paper-solana-online.coordinator` | Hourly minutes 02, 17, 32 and 47 | Reconcile the previous run and start a due training window when the shared worker and budget permit. |
| `fly-paper-solana-online.worker` | Dispatched approximately hourly; no direct cron | Resume the previous checkpoint and run 900 seconds of live paper trading and training, with a 1,200-second hard timeout. |
| Live Pump/PumpSwap collection | During the training window | Subscribe to confirmed events and target five-second neural observations. Collection stops between windows. |
| `fly-paper-checkpoint-dispatch.coordinator` | Every minute | Reconcile completed calls, give due training priority, and dispatch one frozen evaluation at a time when enough time remains. |
| `fly-paper-checkpoint-eval.coordinator` | Hourly minutes 07, 22, 37 and 52 | Register checkpoints, seal new cohorts and future tapes, and aggregate completed comparisons. |
| `fly-paper-checkpoint-eval.worker` | Dispatched between training windows; no direct cron | Score one policy on the sealed recorded tape with learning disabled. Hard timeout: 1,200 seconds. |

Minute positions are the same in Chicago and UTC. All deployed crons use UTC.
Daily cohort eligibility is midnight UTC, currently 19:00 Chicago during daylight
saving time (18:00 during standard time). A new cohort waits for the active one
to finish, so eligibility is not a guaranteed start time.

Training carries forward fly weights and the Q readout, including its saved
optimizer/RNG state. Each new training episode resets paper cash to $1,000 and
clears positions. Completed outcomes and prior losses remain archived. Evaluation
also starts each policy with $1,000, keeps weights frozen, and uses a market
episode that starts strictly after the checkpoint cohort cutoff. New plans
preselect the latest eligible checkpoint before viewing returns.

Training, native evaluation and preparation serialize through the shared worker
lease. The dispatcher starts a new evaluation only when at least 1,250 seconds
(20 minutes 50 seconds) remain before the next training window is due. Existing
evaluations can finish during that interval; otherwise the worker can be idle.
Training starts may drift later than exactly one hour apart. The next due time
is based on the actual dispatch time, not a fixed wall-clock hour.

## Other deployed apps

| App/function | Observed schedule and behavior |
| --- | --- |
| `fly-paper-lab.worker` | Stopped September 17; obsolete five-minute schedule removed. |
| `fly-paper-lab.universe_collector` | Stopped September 17 along with the legacy app; it had not been providing fresh continuous coverage. |
| `fly-paper-solana-5s.worker` | Manual bounded pilot, no repeating schedule. |
| `fly-paper-group-replay.worker` | Manual 12-branch group replay experiment, no repeating schedule. |
| `fly-paper-acceleration-study.worker` | Manual replay/cadence study, no repeating schedule. |
| `fly-paper-checkpoint-smoke.worker` | Manual evaluation diagnostic, no repeating schedule. |
| `fly-paper-universe-probe.worker` | Manual feed diagnostic, no repeating schedule. |

Recovery, cadence-change and prospective-request operator functions have no cron.
All inventoried functions have zero minimum containers. A deployed app can be
available without a running worker. The legacy app was stopped September 17; its old locks and data were preserved.

The compact PPO trader, news ingestion and 12-branch replay are not part of the
current recurring Solana training loop. New tokens between collection windows
are not guaranteed to be captured. Five-second observations do not imply a
24-hour, every-token collector.

## Budget and background operation

The monthly authorization remains $100: an $85 shared worker allocation and $15
for collector/other overhead, including a $5 dispatcher allocation. The worker
ledger rejects new reservations at 75% of its allocation ($63.75), retaining
headroom. These are conservative internal estimates, not provider billing.
Budget guards can pause training/evaluation before the overall ceiling. Training
budget exhaustion defers it to the following UTC month; evaluation failures and
review pauses are not blindly retried.

All the Modal jobs above run independently of the laptop. No local Codex
monitor is required for their scheduled operation. Durable controls are
`/state/solana-online/control.json` and `/state/checkpoint-eval/control.json`.
