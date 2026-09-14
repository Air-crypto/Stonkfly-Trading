"""Descriptive market attribution; no policy replay or causal learning claim."""
import json, sqlite3, bisect, statistics, argparse
from pathlib import Path
from paperlab.solana_events import Feed
from paperlab.solana_paper import tick_for
from scripts.attribute_paper_episodes import attribute

def analyze(path):
    rows=[json.loads(x) for x in (path/'decisions.jsonl').read_text().splitlines()]
    end=rows[-1]['at']; fx=[json.loads(x) for x in (path/'fx.jsonl').read_text().splitlines()];times=[x['at'] for x in fx]
    feed=Feed(path);starts={};outcomes={};symbols={}
    db=sqlite3.connect('file:'+str(path/'events.db')+'?mode=ro',uri=True)
    for (body,) in db.execute('select body from events where received<=? order by received,log_index',(end,)):
        event=json.loads(body);now=event['received'];feed.accept(event)
        mint=event.get('mint') or feed.pools.get(event.get('pool'))
        if mint not in feed.tokens:continue
        token=feed.tokens[mint];symbols[mint]=token['created'].get('symbol','')
        if mint in outcomes:continue
        idx=bisect.bisect_right(times,now)-1
        if idx<0:continue
        t,reason=tick_for(token,now,fx[idx]['sol_usd'],fx[idx]['at'])
        if reason or not t:continue
        if mint not in starts:
            if now+70>end:continue
            starts[mint]=(t.ts,t.mid)
        elif t.ts>=starts[mint][0]+60 and t.ts<=starts[mint][0]+70:
            outcomes[mint]=t.mid/starts[mint][1]-1
    db.close()
    values=list(outcomes.values());stress=[outcomes.get(m,-1.) for m in starts]
    attr=attribute(path/'decisions.jsonl');lastquotes={}
    for row in rows:
        lastquotes.update(row['marks'])
        for execution in row['executions']:
            if execution['tick'].get('available'):lastquotes[execution['mint']]=execution['tick']
        n=row.get('neural')
        if n and n.get('tick',{}).get('available'):lastquotes[n['mint']]=n['tick']
    missing_basis=0;stale_value=0;missing=[]
    for mint,p in rows[-1]['portfolio']['positions'].items():
        if float(p['qty']) and mint not in rows[-1]['marks']:
            missing_basis+=float(p['basis']);q=lastquotes.get(mint);value=float(p['qty'])*q['bid']*.99*.9875 if q else 0
            stale_value+=value;missing.append(dict(mint=mint,basis=float(p['basis']),last_observed_value=value,quote_age=end-q['ts'] if q else None))
    return dict(run=path.name,pnl=attr['pnl_usd'],realized=attr['realized_usd'],unrealized=attr['unrealized_usd'],fees=attr['paid_fees_usd'],
                missing_basis=missing_basis,missing_positions=missing,last_quote_pnl_diagnostic=attr['pnl_usd']+stale_value,
                eligible_launches=len(starts),observed_60s=len(values),missing_60s=len(starts)-len(values),
                median_60s_return=statistics.median(values) if values else None,mean_60s_return=statistics.mean(values) if values else None,
                stress_60s_mean=statistics.mean(stress) if stress else None,sol_return=fx[-1]['sol_usd']/fx[0]['sol_usd']-1,
                market_tokens=[dict(mint=m,symbol=symbols[m],start=starts[m][0],return_60s=outcomes.get(m)) for m in starts])

def correlation(xs,ys):
    mx,my=statistics.mean(xs),statistics.mean(ys)
    den=(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**.5
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den if den else None

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    results=[analyze(x) for x in sorted(a.source.glob('solana-online-*'))]
    out=dict(episodes=results,method='All observed launches reaching tick_for eligibility, independent of trader admission. First eligible quote to first eligible quote at 60-70 seconds; no future endpoint filtering of cohort. Raw indicative USD mid returns, not an executable trading strategy. Missing endpoint also reported as a -100% stress scenario. End-of-episode unavailable holdings separately retain original zero marks.',
             limitations=['One day; small number of episodes.','Cohort entry lacks simulated execution delay, fees, or portfolio allocation: descriptive market metric only.','Eligible endpoint conditioning biases survivor-only returns; stress returns conflate illiquidity, missing data and loss.','Frozen policy replay on identical markets not performed; no causal learning estimate.','Last quote valuation is stale diagnostic only; never changes stored accounts.'],
             correlations={k:correlation([e['pnl'] for e in results],[e[k] for e in results]) for k in ['missing_basis','stress_60s_mean','mean_60s_return']})
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    for e in results:print(e['run'][-6:],*[round(e[k],3) for k in ['pnl','realized','missing_basis','last_quote_pnl_diagnostic','mean_60s_return','stress_60s_mean','sol_return']],e['eligible_launches'],e['missing_60s'])
    print(out['correlations'])
