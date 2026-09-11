# Cloud operation and economics

Deployment target: Modal, one scheduled worker, no continuously running laptop. The checked-in schedule is disabled until deployed explicitly. The worker starts every 15 minutes, ingests RSS, encodes newly seen titles with pinned FinBERT, samples one BTC-USD book and advances independent fly/compact paper ledgers. It persists data and checkpoints on a Modal Volume, then scales down. ETH-USD is supported as a separate experiment; do not change products inside an existing state directory/volume.

## Deployment

The account owner must complete Modal sign-in and its CLI token authorization. No account credentials are part of the repository. Creating an account or accepting its terms belongs to the owner.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev,cloud,news,export]'
.venv/bin/modal setup
# One bounded cloud bootstrap: prepare verified fly data and an initial compact checkpoint.
PAPERLAB_FLY=1 .venv/bin/modal run cloud.py --prepare
# Inspect that invocation's runtime, memory and bill before starting the schedule.
PAPERLAB_SCHEDULE=1 PAPERLAB_FLY=1 .venv/bin/modal deploy cloud.py
```

The worker first collects 64 forward snapshots (about 16 hours) for price-feature warm-up. Fly plasticity then updates from its own paper results; the compact bootstrap is used initially. After 400 snapshots, a fresh compact candidate is trained daily on the chronological training partition of the observed archive. No candidate is chosen by test profit. Current news is never backdated into the historical bootstrap.

Use `modal app list` and the Modal console for the deployed app and function logs. `latest.json` and the SQLite ledgers in `fly-paper-lab-state` contain model decisions, fills and budget estimates. Download files through the Modal CLI/console for inspection; don't mount another writer or edit a database while the worker runs. The upstream native checkpoint includes a binary build hash; an incompatible image must fail restoration rather than silently reset its learned state.

To stop scheduled compute, remove the schedule and redeploy:

```sh
PAPERLAB_SCHEDULE=0 PAPERLAB_FLY=1 .venv/bin/modal deploy cloud.py
```

Changing `PAPERLAB_FLY` or product against the existing runtime state is rejected. Create a separately named app and volume for a different experiment. Do not delete existing state to work around that check.

## Spending limits

User budget: **$100 for September 2026**, then **$20–40/month**. The code uses a $40 steady-state ceiling for planning, reserves a conservative worst-case compute allowance before each invocation, retains the reservation on a crash, and reconciles successful elapsed runtime. Its estimate doubles published CPU/RAM rates and adds 30 seconds per run for overhead. It stops starting substantive work at 75% of the monthly ceiling, leaving headroom.

This is **not a provider-enforced bill cap**. It excludes unknown image-build costs, chargeable network/storage, API subscriptions and unrelated account activity. Configure any available provider billing limit/alerts and check the actual invoice. A `budget_stopped` result requires removing the schedule; even a skipped invocation can have startup costs. No paid news or market-data subscription is enabled.

Resources are 2 CPU cores and 16 GiB RAM for the full-fly configuration, with a 600-second invocation timeout and one container. The compact-only configuration uses 4 GiB. Published rates checked September 11, 2026 imply approximately $0.00006172 per second for the full configuration before extra costs. For 2,880 invocations in a 30-day month:

| Average billed duration per 15-minute invocation | Approximate CPU + RAM / month |
|---|---:|
| 10 seconds | $1.78 |
| 60 seconds | $10.67 |
| 180 seconds | $32.00 |
| 300 seconds | $53.33 |

These are workload scenarios, not measured cloud quotes. Initial local fly observations took roughly 1–2.2 seconds each, excluding some initialization/checkpoint overhead. Local timing cannot predict a different cloud CPU or a long-running neural state's activity. Bootstrap and daily retraining add work. Fit the schedule to measured billed duration; use compact inference and less frequent fly research if the fly exceeds the steady budget. Do not depend on promotional credits to make the strategy profitable.

## Trading must cover operating costs

At **$1,000 paper capital**, $20 and $40 monthly hosting require **2% and 4% monthly return after trading costs**, respectively, merely to break even. The $100 first-month research budget is 10% of that bankroll and should be evaluated as an experiment expense rather than a promised trading return. At $5,000 deployed capital, $20–40 requires 0.4–0.8%. Larger capital can still lose more dollars. Paper cash is simulated and does not add to the proposed later real allocation.

Forward records include net-P&L scenarios subtracting $20/$40/$100 monthly overhead prorated on a 30-day basis. These are scenarios, not actual hosting invoices; replace them with measured costs when assessing profitability. Allocate the total research hosting bill once across a comparison, not once to each policy when summing portfolio results. The initial experiments have not demonstrated an edge and do not cover these overheads.

Sources: [Modal scheduled functions](https://modal.com/docs/guide/cron), [Volumes and persistence](https://modal.com/docs/guide/volumes), [current pricing](https://modal.com/pricing).
