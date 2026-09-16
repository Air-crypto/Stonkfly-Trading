"""Plot retained online training results without compounding reset episodes."""
import json
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault('MPLCONFIGDIR', '/tmp/stonkfly-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/training-performance-20260916'
data = json.loads(Path(str(BASE) + '-source.json').read_text())
rows = data['rows']
legacy = [r for r in rows if r['account_mode'] == 'continuous']
episodes = [r for r in rows if r['account_mode'] == 'fresh_training_episode']
assert len(rows) == data['scheduler_completed_windows']
assert len(episodes) == data['episode_totals']['episodes']
assert abs(sum(r['episode_pnl_usd'] for r in episodes)
           - data['episode_totals']['sum_episode_pnl_usd']) < 1e-6
for r in episodes:
    assert r['initial_cash_usd'] == 1000
    assert abs(r['ending_equity_usd'] - 1000 - r['episode_pnl_usd']) < 1e-7

current = [r for r in episodes if r['quote_protocol'] == 'solana_all_observed_quotes_v3']
stats = dict(completed_windows=len(rows), reset_episodes=len(episodes),
             first_10_current_setup_mean_pnl_usd=mean(r['episode_pnl_usd'] for r in current[:10]),
             last_10_current_setup_mean_pnl_usd=mean(r['episode_pnl_usd'] for r in current[-10:]),
             positive_reset_episodes=sum(r['episode_pnl_usd'] > 0 for r in episodes),
             latest_episode_pnl_usd=episodes[-1]['episode_pnl_usd'],
             cumulative_profit_claim=False)
Path(str(BASE) + '-summary.json').write_text(json.dumps(stats, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'text.parse_math': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#d3d9e0', 'text.color': '#26364a',
                     'axes.labelcolor': '#455468', 'xtick.color': '#657487',
                     'ytick.color': '#657487'})
fig, (old, ax) = plt.subplots(1, 2, figsize=(11, 5.6),
                             gridspec_kw={'width_ratios': [1, 4.6]})
fig.subplots_adjust(left=.075, right=.975, top=.72, bottom=.23, wspace=.40)
fig.suptitle('Paper performance across training', x=.075, y=.97,
             ha='left', fontsize=20, fontweight='bold')
fig.text(.075, .88, 'Current all-Pump setup: average loss per episode', fontsize=11)
fig.text(.075, .815,
         f"First 10: −${abs(stats['first_10_current_setup_mean_pnl_usd']):,.0f}"
         f"    →    Latest 10: −${abs(stats['last_10_current_setup_mean_pnl_usd']):,.0f}",
         fontsize=17, fontweight='bold', color='#194d81')

old.plot([r['window'] for r in legacy], [r['ending_equity_usd'] for r in legacy],
         color='#73849b', marker='o', markersize=4, linewidth=1.5)
old.set_title('First 5 windows\nContinuous account', fontsize=10, loc='left', pad=12)
old.set_ylabel('Ending balance')
old.set_xlabel('Training window')
old.set_xticks([1, 3, 5])
old.set_ylim(890, 960)
old.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:,.0f}'))
old.grid(axis='y', color='#edf0f4')

x = list(range(1, len(episodes) + 1))
y = [r['episode_pnl_usd'] for r in episodes]
ax.axvspan(.5, 12.5, color='#f1f2f5', zorder=0)
ax.axhline(0, color='#61758c', linewidth=1, zorder=1)
ax.plot(x, y, color='#b9c5d5', linewidth=1.2, zorder=2, label='Each episode')
ax.scatter(x, y, c=['#19856a' if p > 0 else '#c97869' for p in y], s=23, zorder=3)
# Never smooth across a quote/accounting protocol change.
trend = [float('nan')] * len(episodes)
for i in range(4, len(episodes)):
    sample = episodes[i-4:i+1]
    if len({r['quote_protocol'] for r in sample}) == 1:
        trend[i] = mean(r['episode_pnl_usd'] for r in sample)
ax.plot(x, trend, color='#194d81', linewidth=2.7, label='5-episode average', zorder=4)
ax.axvline(13.5, color='#9daebe', linewidth=1, linestyle='--')
ax.text(6.5, .96, 'Before accounting fixes', ha='center', va='top',
        transform=ax.get_xaxis_transform(), fontsize=8, color='#727e8d')
ax.text(14.3, .96, 'All-Pump setup →', ha='left', va='top',
        transform=ax.get_xaxis_transform(), fontsize=8, color='#506783')
ax.set_title('44 reset episodes  •  Fresh $1,000 each  •  Windows 6–49',
             fontsize=10, loc='left', pad=12)
ax.set_ylabel('Episode profit / loss')
ax.set_xlabel('Training episode')
ax.set_xlim(.5, len(episodes)+.7)
ax.set_ylim(-780, 275)
ax.set_xticks([1, 10, 20, 30, 40, 44])
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'−${abs(v):,.0f}' if v < 0 else f'${v:,.0f}'))
ax.grid(axis='y', color='#edf0f4')
ax.legend(frameon=False, loc='lower right', fontsize=9)
fig.text(.075, .115,
         'Recorded inventory marks after simulated trading fees; unsold positions are included. Cloud costs excluded.',
         fontsize=9, color='#657487')
fig.text(.075, .072,
         'Separate episodes, not compounded returns. Markets and protocols changed; this is not held-out proof of learning.',
         fontsize=9, color='#657487')
fig.text(.975, .026, 'Snapshot: ' + data['captured_at'][:16].replace('T', ' ') + ' UTC',
         fontsize=8, color='#8290a0', ha='right')
fig.savefig(str(BASE) + '.png', dpi=160, facecolor='white')
fig.savefig(str(BASE) + '.svg', facecolor='white')
print(json.dumps(stats, indent=2))
