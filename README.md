# Stonkfly Trading

A paper-only research lab comparing the full **166,700-neuron Stonkfly network**
with a **247,780-parameter PPO policy**, using market data and timestamped news.
There is no real-order endpoint, wallet, or exchange credential in this lab.

The fly is a sparse spiking network. The compact policy is an MLP actor/critic.
A separately pinned FinBERT transformer can encode headline sentiment; those
features enter the compact policy numerically and the fly through a visual adapter.
The [dynamic memecoin experiment](docs/memecoin-universe.md) discovers current and
new launches and samples DEX pools. Coverage is bounded; it does not trade every
Solana token. The controlled comparison below uses two fixed pools, ALL and baton.

## Latest completed comparison

[Study 11](docs/fly-online-comparison.md#final-audited-result) completed all sixteen
conditions with **360 audited neural observations and 18,000 recorded bins**.
Those observations repeat 90 eligible asset/time slots across four variants.
Each variant starts development and held-out test with a fresh $1,000 account:
two $250 pool allocations plus $500 idle cash.

| Fly condition | Development equity | Held-out test equity |
| --- | ---: | ---: |
| Pristine memory, frozen | $946.73 | $986.34 |
| Paper-trained memory, frozen | $942.86 | $976.36 |
| Paper-trained memory, online updates | $964.45 | $972.13 |
| Online updates, reset KC/DAN rate histories | $972.03 | $984.93 |

**No condition passed development, and no policy was promoted.** Reset reduced
losses relative to online carry in both phases, but trailed pristine frozen in
test. All aggregate results lost money. Simulated adverse fees, spread and
slippage are included; hosting is excluded. Two pools and four hours do not
establish monthly profitability.

![Complete audited development and held-out equity comparison](docs/assets/fly-online-11-comparison.png)

Red crosses mark missing quotes: the affected inventory is valued at $0 as a
stress mark. Those downward spikes are not observed price crashes. Gaps remain
missing in neural plots and do not generate fabricated learning observations.
The [result record](reports/fly-online-result-11.json) preserves the input hashes,
receipts, audits, costs and two disclosed recovery amendments. The original
failed third condition and preempted partial recording remain preserved and
excluded; six completed recordings were reused.

## Inspect the fly

The [interactive debugger](docs/fly-debugger.md) connects recorded activity to the
model's fixed BUY/SELL/HOLD decoder. It provides:

- Neuron spikes, membrane voltage, input timing and accumulated decoder counts.
- Connection weights and stored `u/w`, separately from source firing highlights.
- Paired recordings matched by neuron, connection and market timestamp, with
  first/next differences and shareable links to a selected observation and bin.
- Full-neuron lookup from retained arrays, plus controlled tests of synthetic
  prices, news, neuron stimulation and selected connection restoration.

The simulation retains the complete graph; the displayed circuit is a labeled
subset. A highlighted weight update is not proof that its source just fired,
and synchronized plots do not establish a causal pathway.

![Source firing and changed connection weights in the circuit debugger](docs/assets/fly-source-spike-changed.png)

![Held-out reset and carry recordings: spikes, voltage, weights and stored memory](docs/assets/fly-online-11-test-pool0-pair.png)

After installing below, open the included recordings without model compute:

```sh
uv run python -m paperlab.debugger serve --out examples/fly-debugger
```

Visit <http://127.0.0.1:8765>. To submit new bounded synthetic assays to an
already deployed and authenticated Modal worker, use:

```sh
uv run python -m paperlab.debugger serve --backend modal --out runs/cloud-debugger
```

The [debugger guide](docs/fly-debugger.md) documents controls, full cloud traces,
saved-call recovery, controlled market replays and all earlier experiments.
Diagnostic assays remain separate from paper accounts and policy promotion.

## Reproduce the financial and learning charts

The included [evidence archive](reports/fly-online-evidence-11.zip) contains 51
exact report, plan, summary, audit and receipt files, compressed to about 1.7 MB.
No graph download, credential or new model run is needed to regenerate all five
PNG/SVG figures:

```sh
python3 -m zipfile -e reports/fly-online-evidence-11.zip runs/study-11-evidence
uv run --extra plots python -m paperlab.fly_online_figure \
  --root runs/study-11-evidence --out runs/study-11-figures
```

Use new output directories to preserve earlier evidence. The plotter verifies
all sixteen audit/receipt relationships and the original development selection
before rendering. The archive transports existing audits; repeating the raw
neural audit requires the separately retained full recordings and graph data.

![Held-out ALL decisions, later fills, gate activity and connection updates](docs/assets/fly-online-11-test-pool0.png)

Each panel separates new decisions from fills executing earlier decisions.
Weight movement is a recorded plasticity update, not an optimizer loss or
backprop gradient. The learning-history share is an algebraic decomposition,
not causal credit. [All 90 paired steps](docs/fly-online-paired-steps-11.md) link
back to the complete-recording viewer; its setup requires the retained arrays
as described in the [comparison guide](docs/fly-online-comparison.md).

## What the debugging has established

Stored paper training changed the incoming weights of all six plastic
recipients, but those recipients often remained silent. Controlled stimulation
exposed memory-dependent firing and decisions without demonstrating better
trading. The [recipient activity and stimulation results](docs/fly-debugger.md#inspect-the-quiet-learned-pathway)
retain every control and outcome.

The [learning-drive reconstruction](docs/fly-debugger.md#why-a-quiet-source-connection-can-still-update)
then found that current DAN activity can combine with old KC activity to update
a connection whose source is currently silent. Removing KC history reproduces
the first different weight update in two recorded examples; that one-bin
calculation does not determine later full-network behavior.

The next [selective KC/DAN assay](docs/fly-selective-traces.md) tests each history
separately across twelve full-network conditions. Its worker is prepared and
its completed-study gate is now satisfied. Native selective trajectories and
any subsequent prospective financial test remain pending. No profitable fix is
claimed from changed spikes or weights alone.

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
