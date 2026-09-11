# Stonkfly Trading

A paper-only research lab comparing the full Stonkfly spiking fly network with a **247,780-parameter PPO policy**, using market data and timestamped news. No real-order endpoint, wallet or exchange credentials are used.

The fly is a sparse spiking network, not a transformer. The compact policy is an MLP actor/critic, not an SLM. An optional, separately pinned FinBERT transformer encodes headline sentiment. News features enter the compact policy numerically and the fly through an explicit visual adapter.

## Run

Python 3.12 and a C++ compiler are required. Run from this source checkout so the preserved upstream files are available.

For a locked install, use `uv sync --locked --python 3.12 --all-extras`. The lockfile selects CPU-only PyTorch on Linux. The pip alternative below uses the system platform's default PyTorch wheel; install PyTorch from its CPU index first on a Linux CPU server.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev,cloud,news,export]'
.venv/bin/python -m pytest -q
.venv/bin/paperlab fetch --product BTC-USD --days 14 --out data/btc.jsonl
.venv/bin/paperlab news --db data/news.db
.venv/bin/paperlab train --data data/btc.jsonl --news-db data/news.db --steps 8192 --out runs/ppo
```

Train the actual retained fly graph (about 1.1 GB of public source downloads; plan for 16 GB RAM):

```sh
.venv/bin/paperlab fly-prepare --fly-data data/fly
.venv/bin/paperlab fly-replay --data data/btc.jsonl --news-db data/news.db --fly-data data/fly --steps 32 --out runs/fly-learning
.venv/bin/paperlab fly-replay --data data/btc.jsonl --news-db data/news.db --fly-data data/fly --steps 32 --frozen --out runs/fly-frozen
```

The fly uses Stonkfly's candidate dopamine-modulated plasticity, not PPO. Its retained graph and fixed spike decoder are unchanged. Changing synaptic weights does not establish profitable learning. See [architecture and evaluation](docs/architecture.md), [cloud operation and budgets](docs/cloud.md), and [verification results](reports/verification.md).

Default paper capital is $1,000, with a 50% maximum target allocation and $100 maximum paper order. Costs are assumptions, not a verified exchange fee tier: 60 bps per side plus 10 bps slippage. Reports value open inventory at bid less estimated exit costs. Current headlines are never retroactively inserted into old price history.

`vendor/stonkfly` preserves [nftechie/stonkfly](https://github.com/nftechie/stonkfly) at commit `78ef3e05ab0fa086032098558d893667068944a0`, including its MIT license and third-party notices. This lab imports only its neural/data/display modules; its live trading implementation is not wired into the lab.

**Status:** deployed on Modal with a five-minute paper-data schedule and six-hourly PPO retraining after 400 observations. The new cadence uses a separate forward archive; the original 15-minute experiment is preserved. Closed-candle price context enables earlier paper decisions without adding historical fills or forward-training samples. Full decision, loss, gradient and fly-weight diagnostics are described in [observability](docs/observability.md). Sustained profitability has not been established.
