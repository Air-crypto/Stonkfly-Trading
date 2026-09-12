"""Read-only observation coverage for the registered study 11 market window.

This inspects a closed collector snapshot. It does not seal inputs, compute
actions or returns, choose a model, or alter the scheduled experiment.
"""
import argparse
from bisect import bisect_right
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

from .core import atomic_json, digest
from .fly_market_study import quote_at
from .fly_online_cloud import expected_sources
from .fly_online_protocol import MIN_OBSERVATIONS, PHASE_STEPS
from .multi import MIN_CONTEXT, read_archive
from .universe import Pool


def coverage(registration, archive):
    r=registration;archive=Path(archive)
    if (r.get('study')!='11' or r.get('phase_steps')!=PHASE_STEPS or r.get('decision_seconds')!=300
            or len(r.get('cohort',[]))!=2 or len(set(r['cohort']))!=2
            or r['development_start']%300 or r['test_start']!=r['development_start']+PHASE_STEPS*300
            or r['end']!=r['test_start']+PHASE_STEPS*300):
        raise ValueError('Expected the registered study 11 time grid and two fixed pools')
    raw={}
    with closing(sqlite3.connect(f'file:{archive.resolve()}?mode=ro',uri=True)) as db:
        if db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:raise ValueError('Invalid snapshot database')
        last=db.execute("SELECT max(json_extract(payload,'$.observed')) FROM observations").fetchone()[0]
        if not isinstance(last,(float,int)) or not math.isfinite(last):raise ValueError('Snapshot has no observed endpoint')
        # Do not classify the still-open receipt minute or any future decision.
        cutoff=min(r['end'],math.floor(last/60)*60)
        for key in r['cohort']:
            receipts=[]
            for slot,payload,reason in db.execute('SELECT slot,payload,reason FROM observations WHERE key=? ORDER BY slot',(key,)):
                p=Pool(**json.loads(payload))
                if (p.key!=key or slot!=int(p.observed//60) or p.rejection()!=reason
                        or receipts and p.observed<=receipts[-1].observed):
                    raise ValueError('Raw receipt identity, chronology or eligibility differs')
                receipts.append(p)
            raw[key]=receipts
    _,series=read_archive(archive,cutoff)
    pools={}
    for key in r['cohort']:
        ticks=series.get(key,[]);receipts=raw[key];raw_times=[p.observed for p in receipts]
        past=[t for t in ticks if t.ts<=r['development_start'] and t.available]
        phases={}
        for phase in ('development','test'):
            rows=[];last_quote=0
            for i in range(PHASE_STEPS+1):
                stamp=r[phase+'_start']+300*i
                row={'index':i,'decision_ts':stamp,'terminal':i==PHASE_STEPS,
                     'quote_ts':None,'quote_available':None,'raw_receipt_ts':None,'raw_receipt_age_seconds':None,
                     'raw_rejection':None,'neural_observation_expected':None,'status':'awaiting_snapshot'}
                if stamp<=cutoff:
                    ri=bisect_right(raw_times,stamp)-1
                    receipt=receipts[ri] if ri>=0 else None
                    if receipt:
                        row.update(raw_receipt_ts=receipt.observed,raw_receipt_age_seconds=stamp-receipt.observed,
                                   raw_rejection=receipt.rejection())
                    if ticks and ticks[0].ts<=stamp:
                        _,q=quote_at(ticks,stamp);observed=not row['terminal'] and q.available and q.ts>last_quote
                        if observed:last_quote=q.ts
                        reason='terminal_mark' if row['terminal'] else 'observed' if observed else (
                            'stale_or_missing_quote' if receipt is None or stamp-receipt.observed>180 else
                            'ineligible_quote' if not q.available else 'no_new_quote')
                        row.update(quote_ts=q.ts,quote_available=q.available,
                                   neural_observation_expected=observed,status=reason)
                    else:row.update(neural_observation_expected=False,status='missing_prior_context')
                rows.append(row)
            decisions=rows[:-1];observed=sum(row['neural_observation_expected'] is True for row in decisions)
            pending=sum(row['neural_observation_expected'] is None for row in decisions)
            gate=('met' if observed>=MIN_OBSERVATIONS else 'below_threshold') if pending==0 else (
                'threshold_unreachable' if observed+pending<MIN_OBSERVATIONS else 'pending')
            phases[phase]={'observed':observed,'missing':PHASE_STEPS-observed-pending,'pending':pending,
                           'minimum_observations':MIN_OBSERVATIONS,'maximum_possible_observations':observed+pending,
                           'coverage_gate':gate,'rows':rows}
        pools[key]={'symbol':receipts[-1].symbol if receipts else None,
                    'context_observations_before_development_in_snapshot':len(past),
                    'context_requirement_met':len(past)>=MIN_CONTEXT,'phases':phases}
    return {'status':'provisional_input_coverage','snapshot_sha256':digest(archive),'snapshot_end':last,
            'assessed_through':cutoff,'pools':pools,'source_sha256':digest(__file__),
            'interpretation':'Coverage of a fixed downloaded snapshot, not the final sealed plan. Future/open-minute slots remain unknown; missing registered pools are retained. No news audit, neural activity, action, return, model selection or promotion is computed.'}


def run(registration, arming, archive, output):
    registration,arming,archive,output=map(Path,(registration,arming,archive,output))
    if output.exists():raise ValueError('Preserve the earlier coverage report')
    receipt=json.loads(arming.read_text())
    if digest(registration)!=receipt['registration_sha256']:raise ValueError('Registration differs from cloud arming evidence')
    if expected_sources()!=receipt['source_hashes']:raise ValueError('Execution sources differ from cloud arming evidence')
    result=coverage(json.loads(registration.read_text()),archive)
    result.update(registration_sha256=digest(registration),arming_evidence_sha256=digest(arming),
                  execution_sources_match_arming=True)
    atomic_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('registration','arming','archive','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=run(a.registration,a.arming,a.archive,a.out)
    print(json.dumps({'status':r['status'],'snapshot_end':r['snapshot_end'],'assessed_through':r['assessed_through'],
        'pools':{key:{'context_ready':v['context_requirement_met'],**{phase:{k:x for k,x in row.items() if k!='rows'}
                 for phase,row in v['phases'].items()}} for key,v in r['pools'].items()}},indent=2))


if __name__=='__main__':main()
