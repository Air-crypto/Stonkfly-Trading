# Solana launch collection and five-second live paper training

This is a new, bounded training pilot. It preserves studies 11 and 12 and does
not turn their failed results into a selected strategy. The worker runs in Modal
after the laptop closes; the local debugger remains a viewer of saved studies.

## Data and scope

The design follows the direct program-log subscription in
[solid-octo-engine](https://github.com/Air-crypto/solid-octo-engine/blob/ef7bb562c0fa5820e4f8f6969cc5a12833e1263a/src/adapters/pump-events.ts).
It uses one public Solana `logsSubscribe` connection at **confirmed** commitment
for the Pump.fun program. There is no wallet, signer, transaction submission,
paid PumpPortal trade subscription or token-supplied URL fetch.

The reference repo pins SDK 1.36.0. Current live events include additional fields;
the new Python decoder pins the official Pump SDK **2.0.0** event definitions and
checks the complete payload. Unknown or malformed layouts cannot become prices.
The committed on-chain fixture reproduces this discovery; it is not a trading
result. Program-stack checks exclude data emitted by other invoked programs,
failed transactions are rejected, and event identity is signature plus log index.

All decoded create/trade/complete events received during the bounded window go
to SQLite, including rejected and disappearing launches. The in-memory working
set is capped at 512 launches and 256 recent trades each; the selected launch is
pinned. At 200,000 archived events, collection stops. Disconnects, decoding
errors, evictions and duplicates are counted. There is **no backfill or claim of
complete Solana coverage**. Existing coins and other launchpads are outside this
first cohort; non-SOL quote tokens are retained but not traded.

Every five seconds the trader inspects the latest received data. It does not
convert an unchanged cached price into a new training sample. Model runtime,
initialization, network latency, missing prices and missed five-second deadlines
are reported. Collection stays on a separate thread during inference.

## Paper mechanics and learning

The first eligible launch by observed creation receipt is selected without future
outcomes. It must be 30 seconds–one hour old, observed near creation, have recent
two-sided activity, usable native-SOL reserves, and enough real SOL reserves
that a $25 order is at most 1%. Mayhem-mode launches are excluded. These checks
are data admission rules, not token safety or sellability certification.

Each independent pilot starts with **$1,000 paper cash**, with a 2.5% maximum
exposure and $25 order cap. Only one token is selected per pilot. It is not
replaced if it disappears. Wallet/token authorities and exact Token-2022
extensions are not reconstructed; this prevents live-readiness claims.

Prices are indicative USD per token base unit from event virtual reserves and
an observed Coinbase SOL/USD reference (refreshed once per minute, expires after
90 seconds). No token decimal count is assumed. Simulated execution adds a
100-bps spread, 100-bps per-side slippage and 125-bps per-side fees. It is not an
exact bonding-curve swap, sellability check, AMM-impact or MEV model. A pending
trade requires a genuinely later receipt than decision publication and expires
after 15 seconds. Missing markets retain quantities and stress-mark inventory
at zero rather than invent a sale. Learning credit is cut across unavailable or
long-gapped observations.

After 12 distinct five-second sampled observations, the **full native fly** runs
500 neural milliseconds per observation with its original plasticity enabled.
Its reward/aversive pulse uses the change in paper liquidation equity. Native
learning reports weight deltas; it does not have a backpropagation loss.

A separate 22-input, 32-hidden-unit, two-output neural readout learns FLAT versus
2.5%-LONG targets with online one-step Q-learning, Huber loss, Adam, clipped
gradients and 10% paper exploration. Inputs combine market/account state with
fly activity. Rewards are paper equity changes divided by $25, clipped to [-1,1];
discounting scales with actual elapsed time. Logs include predictions, chosen
targets, loss, TD error, gradient norms and weight deltas. This is an experimental
online Q learner, not a claim of convergence or a proven profitable policy.

The old fixed fly decoder is logged for diagnosis. Cash and a capped one-entry
holding account are descriptive references. They do not establish an independent
frozen-fly or compact-model control. News is disabled in this pilot so the launch
feed and action learner can be checked first. Native state and readout optimizer
are checkpointed at completion; pilot IDs are immutable and cannot be restarted
to erase losses. Cross-window resumption is not yet implemented.

## Cloud operation and spending

```sh
PAPERLAB_FLY=1 PAPERLAB_UNIVERSE=1 PAPERLAB_SCHEDULE=0 \
  uv run --extra cloud modal deploy solana_cloud.py
# Dispatch once. Save the returned call ID; an observer timeout is not a retry signal.
uv run --extra cloud python -c 'import modal; print(modal.Function.from_name("fly-paper-solana-5s", "worker", environment_name="main").spawn("solana-live-13", seconds=900).object_id)'
uv run --extra cloud python -m paperlab.solana_watch solana-live-13 --out runs/solana-live-13-status.json
PYTHONHASHSEED=0 uv run --extra dev --extra cloud python -m pytest tests/test_solana_live.py -q
```

Do not reuse that run ID if it already exists. The deployment has **no automatic
recurring schedule**. A call runs up to 15 minutes of collection/training and then
saves its checkpoints; its hard function timeout is 20 minutes. This pilot must
demonstrate cloud feed health, fresh-data decisions, real native updates and its
achieved cadence before ongoing training is enabled.

The 2-CPU/8-GiB worker shares the existing writer lease and $25 monthly worker
ledger with `fly-paper-lab`. Its collector can continue separately; the old
trader yields while this worker owns the lease. Before starting, the pilot
reserves the full hard timeout at 3x nonpreemptible prices and a 2x uncertainty
margin: **$0.3191544**. Crashes retain that reservation. Successful completion
settles elapsed time. This is a conservative compute ledger, not an invoice;
image builds and storage are not included. Existing provider usage/spend limits
are unchanged. Full-fly inference every five seconds around the clock has not
been shown to fit the $20–40 steady monthly target.

`latest.json` and JSON logs expose progress. The final closed `events.db`,
`decisions.jsonl`, `fx.jsonl`, per-observation native arrays and final checkpoints
remain in `fly-paper-lab-state/solana-live/<run-id>/`. Read the database after
completion; intermediate volume publication is not a sealed SQLite snapshot.
The status reader performs no cloud dispatch and no neural computation.

## Next acceptance gates

1. Verify the actual cloud entrypoint, parser, selected token, delayed paper fills,
   native changes, trainable-readout gradients and measured observation cadence.
2. Add durable cross-window continuation without resetting balances or dropping
   failed tokens, then enable bounded ongoing training within measured costs.
3. Run many launch cohorts; compare fly+readout against market-only readout,
   frozen fly, compact policy and cash under identical data and costs.
4. Freeze candidates before entirely later launches. Test news separately and
   validate executable prices before any live-money consideration.

References: [Solana logsSubscribe](https://solana.com/docs/rpc/websocket/logssubscribe),
[official Pump SDK](https://github.com/pump-fun/pump-sdk),
[Modal resource prices](https://modal.com/pricing).
