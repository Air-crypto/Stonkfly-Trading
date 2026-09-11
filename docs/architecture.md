# Architecture and experiment protocol

## What is trained

**Fly:** the full checksum-locked MaleCNS graph retained by Stonkfly, with 166,700 neurons and 25,582,938 directed connections. It uses a sparse spiking LIF simulation and candidate dopamine-modulated KC→MBON plasticity on 7,835 edges. A fixed neuronal decoder chooses BUY, SELL or HOLD. This is not a transformer, language model, biological prediction oracle, or validated profitable system. The unmodified upstream implementation and caveats are in `vendor/stonkfly`.

**Compact:** an MLP with 64 inputs, two 464-unit tanh layers, a three-action head and a scalar value head: 247,780 trainable parameters. PPO collects fresh on-policy rollouts, uses GAE and clipped updates, and optimizes changes in net marked equity with a small incremental-drawdown penalty. Actions target 0%, 25% or 50% asset exposure. A 250k-parameter numerical policy is practical; a language model of that size would require aggressive task restriction and would not replace a broadly pretrained news encoder.

**News:** FinBERT is an optional BERT transformer, independent of both policies, pinned to `ProsusAI/finbert@4556d13015211d73dccd3fdd39d39232506f3e43`. It supplies positive/negative/neutral probabilities. The default CLI fallback uses a deliberately basic word list. The cloud configuration enables FinBERT. Hashed title features preserve a small amount of token identity, with count, age and sentiment producing 50 news inputs. These are not fact verification or reliable event forecasts. FinBERT is not retrained by this project.

The compact observation combines those 50 values with 14 causal market/portfolio features. The fly sees a 320×180 market image with a fixed news strip. Colors encode sentiment probabilities; gray cells encode hashed features and freshness. This is an engineered sensory interface, not evidence that the fly reads headlines. The graph and neural decoder are unchanged; only the external observation display differs from upstream.

## Causality and execution

News is an append-only SQLite archive of title revisions. Each item records publication time, first-seen time, encoding completion time and encoder revision. It is available only after all three clocks permit it. Repeated collection does not refresh an old item. A URL contributes its newest available revision; future revisions cannot alter an earlier observation. Feed failures are recorded explicitly. Text is never executed or used as an agent instruction.

In historical replay, a decision uses completed observations through time t and fills at the next recorded close, with assumed spread, fees and slippage. Coinbase historical candles do not reconstruct an order book. This is a deliberately labeled proxy and not executable-price proof.

Forward paper mode records public REST best bid/ask snapshots. A decision's timestamp is taken **after** inference, and a simulated fill must use a later snapshot. An expired decision or excessive spread is rejected. There is no live order sender. Partial fills, queue position, depth and market impact remain unmodeled, so forward paper results are still optimistic in some respects. Volume features are zero for book snapshots, unlike historical candle volume; bootstrap-to-forward distribution shift must be measured.

Accounting uses Decimal cash/quantity, bounded order notional and inventory, and bid liquidation value less estimated exit costs. A loss stop rejects new exposure without inventing a replacement trade. It does not guarantee a maximum loss; open inventory remains exposed. Portfolio state and pending decisions survive restarts in one transaction. Fly checkpoints are written under a new name before the database references them. The Modal worker and filesystem lock allow only one writer.

Fly reinforcement comes from the subsequent change in its own net portfolio equity. That delayed reward is an engineered signal and may have weak temporal credit assignment. It includes market movement while holding inventory, not only realized sales. Frozen-fly controls receive the same observation and reinforcement protocol with plasticity disabled.

## Evaluation

1. Split observations chronologically into 70% train, 15% validation and 15% test, with fresh cash at each evaluation boundary. Feature warm-up uses only earlier observed prices. A checkpoint is fixed before its test results are calculated.
2. Compare fixed seeds 7, 19 and 42; report every run. Compare with cash, an equal-exposure capped buy-and-hold entry, and a simple trend rule under the same modeled fees and order cap. Do not select the most flattering seed or infer monthly returns from a short sample.
3. Compare learning and frozen fly on identical windows. For a learned-checkpoint evaluation, use `--checkpoint` and `--frozen` on a later window; never initialize a historical evaluation from weights trained on later data. Upstream checkpoint validation binds to native build/data hashes, so train and restore inside the same cloud image.
4. After collecting timestamped news, rerun the same preregistered protocol with `--no-news`. Today's headlines cannot be used to claim improved performance on old candles. Current historical smoke experiments have zero causally available news.
5. Keep an append-only forward record of each model hash and switch. Compact retraining follows a fixed six-hourly schedule after 400 observed snapshots (roughly 33 hours at five-minute intervals in the new independent archive), automatically replacing only the paper policy. Repeated rolling test reports overlap; the persistent forward ledger, rather than repeated historical tests, measures the adaptive system.
6. Run for multiple market conditions, examine turnover, drawdown, tail events, baseline-relative return, model-switch effects and cost sensitivity. Use untouched future windows and block-bootstrap uncertainty estimates before judging an edge. These extended statistical evaluations are not yet implemented or completed.

No backtest, fixed number of days, Sharpe ratio or passing unit test automatically authorizes real trading. The proposed later $1k–$5k allocation is conditional on evidence; this repository does not implement live execution.

## Quantization

The optional export produces FP32 and INT8 ONNX models and reports size, local latency, logit changes and sampled action agreement. The initial check shrank weights about fourfold but did not establish a material cost benefit. INT8 is not enabled automatically. It needs full sequential-trajectory checks and benchmarks on the target cloud CPU. The fly uses its original float32 native kernel and is not quantized.

Sources: [Stonkfly model](https://github.com/nftechie/stonkfly/blob/main/docs/model.md), [FinBERT model card](https://huggingface.co/ProsusAI/finbert), [PPO paper](https://arxiv.org/abs/1707.06347), [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
