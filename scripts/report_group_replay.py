"""Render measured replay diagnostics; does not launch training or native inference."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def report(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    result=json.loads((source/'completed.json').read_text())
    audit=json.loads((source/'audit.json').read_text())
    updates=json.loads((source/'updates.json').read_text())
    trajectories=[json.loads(line) for line in (source/'training-trajectories.jsonl').read_text().splitlines()]
    actual=[u for u in updates if u['status']=='updated']
    skipped_groups=len({u['group'] for u in updates if u['status']=='skipped_zero_reward_variance'})
    heldout=[r for r in result['evaluation'] if r['split']=='test']
    passes=defaultdict(list)
    for t in trajectories:passes[t['repeat']].append(t['pnl_usd'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axes[0,0];ax.plot([r['loss'] for r in actual],color='#2563eb')
    ax.set(title='Actor loss (group-normalized; not trading return)',xlabel='Gradient update',ylabel='Objective loss')
    ax=axes[0,1];ax.plot([r['gradient_norm'] for r in actual],color='#7c3aed')
    ax.set(title='Gradient norm before clipping',xlabel='Gradient update',ylabel='L2 norm')
    ax=axes[1,0];ax.plot([i+1 for i in passes],[np.mean(passes[i]) for i in passes],marker='o',color='#0891b2')
    ax.axhline(0,color='gray',linewidth=1)
    ax.set(title='Training replay PnL (same four markets reused)',xlabel='Training pass',ylabel='Mean USD per isolated episode')
    ax=axes[1,1]
    ax.bar([r['policy'] for r in heldout],[r['mean_pnl_usd'] for r in heldout],color=['#2563eb','#94a3b8','#16a34a','#d97706'])
    ax.axhline(0,color='gray',linewidth=1)
    ax.set(title='Held-out mean PnL: four mints, one day',ylabel='USD per $1,000 / 60-second episode')
    fig.suptitle('GSPO-inspired fly paper replay — mechanics pilot, indicative fills',fontsize=15)
    fig.savefig(destination.with_suffix('.png'),dpi=160);plt.close(fig)
    sample=[t for t in trajectories if t['group']==0]
    lines=['# GSPO-inspired paper replay: measured pilot','',
        f"Run: `{source.name}`. Paper-only. **{result['promotion']}**",'',
        f"Completed {result['groups']} groups × 12 branches = {result['training_trajectories']} training trajectories, "
        f"{result['gradient_updates']} gradient updates. {skipped_groups} groups had no reward variance and skipped training.",
        f"The {result['actor_parameters']}-parameter actor used frozen pristine full-fly features. "
        f"Native observations: {result['encoder']['native_observations']}; native weight delta: zero. "
        f"Actor checkpoint L2 change: {result['actor_weight_delta_l2']:.6f}.",'',
        'Four earlier mints trained the actor; four later mints, excluded from the entire earlier archive, tested it. '
        'Each branch starts with its own $1,000 cash, targets at most 0.25% exposure, and lasts 60 seconds. '
        'These are isolated token episodes, not a combined portfolio or a forecast of monthly returns.', '',
        f"![Measured diagnostics]({destination.stem}.png)",'',
        '## Matched evaluation','',
        'PnL includes assumed entry fees, spread, slippage and liquidation costs. Missing terminal quotes value inventory at zero; '
        'open inventory is marked, not fabricated as a sale. Twelve seeds reduce action-sampling noise, not market uncertainty.','',
        '| Split | Policy | Mean PnL (USD) | Mean reward | Mean fills | Mean paid fees (USD) |',
        '|---|---|---:|---:|---:|---:|']
    for r in result['evaluation']:
        lines.append(f"| {r['split']} | {r['policy']} | {r['mean_pnl_usd']:+.6f} | {r['mean_reward']:+.6f} | {r['mean_fills']:.2f} | {r['mean_fees_usd']:.6f} |")
    lines += ['', '## Weight updates and verification','',
        f"First / last objective loss: {result['loss_first']:.8f} / {result['loss_last']:.8f}. "
        f"Mean pre-clipping gradient norm: {result['gradient_norm_mean']:.6f}. "
        'Group-normalized loss values do not establish convergence or profitable learning.',
        f"Independent Decimal audit passed: {audit['trajectories']} trajectories, {audit['rows']} account rows, "
        f"{audit['fills']} fills. The audit checks receipt timing, fees, cash, inventory, branch features, rewards, "
        'aggregate results and actual actor checkpoint changes.',
        f"Wall time: {result['wall_seconds']:.1f}s. Conservative compute estimate: ${result['budget']['estimated_compute_usd']:.4f}. "
        f"Shared worker ledger after settlement: ${result['budget']['monthly_reserved_usd']:.4f}; this is not a provider invoice.", '',
        '## All twelve alternatives from the first training group','',
        f"Mint: `{sample[0]['mint']}`. Actions shown as seconds from episode start. "
        'HOLD steps are omitted for readability; every timestep and rejected/expired order remains in the raw ledger.', '',
        '| Branch | Requested orders | Fills | Net PnL (USD) | Reward |','|---:|---|---:|---:|---:|']
    for t in sample:
        start=t['rows'][0]['now']
        orders=', '.join(f"{r['now']-start:g}s {r['action_name']}" for r in t['rows'][:-1] if r['action']!=0) or 'HOLD throughout'
        lines.append(f"| {t['branch']+1} | {orders} | {t['fills']} | {t['pnl_usd']:+.6f} | {t['reward']:+.6f} |")
    lines += ['', '## Limits and next evidence gate','',
        'This pilot does not update native fly synapses, ingest news, change the live trader or schedule recurring GSPO jobs. '
        'A matched current-Q comparison is deferred because its encoder, reward conditioning and account/action settings differ. '
        'There is no market-only ablation yet, so any improvement cannot be attributed to the fly features. '
        'Reserve-based quotes are indicative; actual execution, latency, failed transactions and market impact remain unvalidated.',
        'Before promotion: preregister a new forward collection across more independent dates and mints, retain failed and dead tokens, '
        'compare against cash, frozen/random, market-only and matched simple policies, then evaluate outcomes after realistic execution and hosting costs. '
        'Do not tune on or reuse these test results as a fresh holdout.', '',
        f"Raw local artifacts: `{source.resolve()}`. Includes all rollout rows, masks, behavior probabilities, losses, gradients, "
        'sequence ratios, clipping, entropy/KL, feature provenance and model checkpoints.', '',
        'Algorithm reference: [Qwen GSPO](https://qwenlm.github.io/blog/gspo/). This is a trading adaptation, not a claim of validated GSPO trading performance.', '']
    destination.with_suffix('.md').write_text('\n'.join(lines))
    for name in ('completed.json','manifest.json','audit.json','updates.json'):
        shutil.copyfile(source/name,destination.with_name(destination.stem+'-'+name))
    print(destination.with_suffix('.md').resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source');parser.add_argument('destination')
    args=parser.parse_args();report(args.source,args.destination)
