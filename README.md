# Stonkfly Trading

A paper-only research lab comparing the full Stonkfly spiking fly network with a **247,780-parameter PPO policy**, using market data and timestamped news. No real-order endpoint, wallet or exchange credentials are used.

The [dynamic memecoin experiment](docs/memecoin-universe.md) adds free live launch
discovery, multi-chain pool sampling, separate $1,000 portfolios, isolated fly
contexts, and pooled PPO training. Coverage is bounded and DEX fills are explicitly
indicative simulations. Use `PAPERLAB_UNIVERSE=1` when deploying this experiment;
the original BTC archive remains available.

The fly is a sparse spiking network, not a transformer. The compact policy is an MLP actor/critic, not an SLM. An optional, separately pinned FinBERT transformer encodes headline sentiment. News features enter the compact policy numerically and the fly through an explicit visual adapter.

## Inspect the fly

The [interactive circuit debugger](docs/fly-debugger.md) shows recorded spikes,
voltages, connection updates, and fixed decoder outputs. Test synthetic prices,
news, neuron stimulation, and selected connection restoration. The simulation
retains the full graph; the display is a labeled subset with full-neuron lookup
in generated recordings. Paper accounts remain separate from diagnostic assays.

![Fly circuit debugger](docs/assets/fly-debugger.png)

After installing below, open the included recordings without model compute:

```sh
uv run python -m paperlab.debugger serve --out examples/fly-debugger
```

Visit <http://127.0.0.1:8765>. With Modal deployed and authenticated, use
`uv run python -m paperlab.debugger serve --backend modal --out runs/cloud-debugger`
to run new synthetic assays and inspect full cloud traces. See the guide for
[controlled market replays](docs/fly-debugger.md#separate-market-memory-updates-and-reinforcement),
source verification, saved-call recovery, and run steps.

The diagnostics have identified concrete limitations:

- The original chart can render small and large percentage moves identically.
  An experimental fixed return encoding preserves the distinction but failed
  the [fourth market comparison](docs/fly-debugger.md#fourth-market-replay-result).
- In that study, original training mode produced an extra BUY that cost $1.35
  relative to frozen on the pool with fresh quotes. The
  [eight-arm replay](docs/fly-debugger.md#market-pulse-factorial-results) reproduced
  both reference controls and isolated the extra BUY to the combination of
  ongoing plasticity and recorded reinforcement. Starting with earlier trained
  memory did not change those three observations' spike counts or actions.
- Freezing weights removed that extra BUY in this diagnostic. Whether trained,
  frozen inference improves fresh market results remains a separate test;
  no model change has been promoted to the regular paper trader.

![All eight memory, update, and pulse conditions on identical inputs](docs/assets/fly-market-pulse-01.png)

After starting the included viewer, [jump to the extra gate spike](http://127.0.0.1:8765/?run=pulse01-trained_online_recorded&step=2&bin=37&neuron=10527&compare=pulse01-trained_frozen_recorded).
The link selects neuron 10527 at the 1,380 ms bin boundary. Use previous/next
spike buttons and the moment link to inspect and share any displayed neuron.
The guide retains earlier experiments, all outcomes, and their limitations.
These visualizations diagnose the model; they do not demonstrate profitable trading.

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

The original BTC mode supports a five-minute paper-data schedule and six-hourly
PPO retraining after 400 observations. The multi-pool mode has separate collection,
eligibility, training gates, and ledgers documented above. Full BTC decision, loss,
gradient and fly-weight diagnostics are described in [observability](docs/observability.md).
Sustained profitability has not been established for either experiment.
