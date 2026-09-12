import copy
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from paperlab.core import atomic_json,digest
from paperlab.fly_market_study import signature,learned_state,memory_signature
from paperlab.fly_paper_inputs import seal,validate,news_stamps,SnapshotNews,audit_news
from paperlab.fly_paper_protocol import MEMORY_FIELDS
from paperlab.news import News


def documents():
    return (json.loads(Path('reports/fly-market-study-09-preregistration.json').read_text()),
            json.loads(Path('reports/fly-paper-memory-audit-01.json').read_text()))


def make_news(path,start):
    news=News(path)
    news.add('https://example.invalid/known','growth rally','fixture',start-60,start-60,encoded=start-60)
    news.add('https://example.invalid/late-seen','crash loss','fixture',start-60,start+300,encoded=start+300)
    news.add('https://example.invalid/late-encoded','hack ban','fixture',start-60,start-60,encoded=start+600)
    news.add('https://example.invalid/future-publication','surge gain','fixture',start+900,start-60,encoded=start-60)
    news.db.close()


def test_sealer_preserves_registered_prices_and_all_three_news_clocks(tmp_path):
    from paperlab.universe import Pool,Store
    r,a=documents();start=r['development_start'];archive=tmp_path/'prices.db';store=Store(archive)
    for i in range(240):
        ts=start-120*60+i*60
        store.add([Pool(key,key.split(':')[0],key.split(':')[1],f'token{k}',f'TEST{k}',
                       1 if ts<=r['end'] else 99999,100000,10000,20,20,ts-3600,ts,source='synthetic') for k,key in enumerate(r['cohort'])])
    store.db.close();news=tmp_path/'news.db';make_news(news,start)
    env=seal(archive,news,'reports/fly-market-study-09-preregistration.json','reports/fly-paper-memory-audit-01.json',
             'reports/fly-market-study-08-plan.json',tmp_path/'plan.json')
    validate(env);audit_news(env,news)
    assert all(t['ts']<=r['end'] and t['bid']<10 for ticks in env['plan']['series'].values() for t in ticks)
    recorded=SnapshotNews(news)
    try:
        assert len(recorded.rows(start))==1
        assert len(recorded.rows(start+300))==2
        assert len(recorded.rows(start+600))==3
        assert len(recorded.rows(start+900))==4
    finally:recorded.db.close()
    bad=copy.deepcopy(env);stamp=next(iter(bad['plan']['news_features']));bad['plan']['news_features'][stamp][0]+=.1;bad['sha256']=signature(bad['plan'])
    with pytest.raises(ValueError,match='eligible archived'):audit_news(bad,news)
    with pytest.raises(ValueError,match='overwrite'):
        seal(archive,news,'reports/fly-market-study-09-preregistration.json','reports/fly-paper-memory-audit-01.json','reports/fly-market-study-08-plan.json',tmp_path/'plan.json')
    with sqlite3.connect(archive) as db:
        db.execute("DELETE FROM observations WHERE json_extract(payload,'$.observed')>=?",(r['end'],))
    with pytest.raises(ValueError,match='endpoint'):
        seal(archive,news,'reports/fly-market-study-09-preregistration.json','reports/fly-paper-memory-audit-01.json','reports/fly-market-study-08-plan.json',tmp_path/'incomplete.json')


def native_fixture(tmp_path,study="09"):
    from paperlab.fly_trace import TraceLab,synthetic_ticks
    from paperlab.fly_paper_memory import array_hash
    r,a=documents()
    if study=='10':r=json.loads(Path('reports/fly-market-study-10-preregistration.json').read_text())
    lab=TraceLab(Path(os.environ['FLY_TRACE_DATA']));b=lab.brain
    from stonkfly.neural.brain import PARAMETERS
    # Explicit synthetic learned-memory fixture. Real checkpoint/ledger export is tested separately.
    package=tmp_path/'memory';package.mkdir();series={}
    for i,key in enumerate(r['cohort']):
        memory=learned_state(b);memory['u'][0]=.01;memory['w'][0]=.02
        memory['weights']=(b.baseline_plastic*(1+memory['w'])).astype(b.weight.dtype)
        file=package/f'pool{i}-memory.npz';np.savez_compressed(file,**memory)
        source=a['pools'][key];source.update(memory_sha256=memory_signature(memory),memory_file_sha256=digest(file))
        source['source_metadata'].update(build=b.build,parameters=PARAMETERS,
            graph_ids_sha256=array_hash(b.ids),graph_ptr_sha256=array_hash(b.ptr),graph_post_sha256=array_hash(b.post),
            plastic_edges_sha256=array_hash(b.circuit['edges']),configuration_sha256=b.configuration_signature())
        r['source_memories'][key]={k:source[k] for k in MEMORY_FIELDS}
        ticks=synthetic_ticks('reversal',8);shift=r['development_start']-ticks[99].ts
        ticks=[replace(t,ts=t.ts+shift,received_at=t.ts+shift,product=key) for t in ticks if t.ts+shift<=r['end']]
        ticks[100]=replace(ticks[100],available=False)
        if i==1:
            for j in (99,100,101):ticks[j]=replace(ticks[j],available=False)
        series[key]=[asdict(t) for t in ticks]
    atomic_json(package/'audit.json',a);r['training_audit_sha256']=digest(package/'audit.json')
    news=tmp_path/'news.db';make_news(news,r['development_start']);reader=SnapshotNews(news)
    try:vectors={str(float(t)):reader.features(t).tolist() for t in news_stamps(r,series)}
    finally:reader.db.close()
    p={'registration':r,'training_audit':a,'series':series,'news_features':vectors,'snapshot_sha256':'a'*64,
       'snapshot_end':r['end'],'news_snapshot_sha256':digest(news)}
    return {'plan':p,'sha256':signature(p)},package,news


@pytest.mark.skipif(not os.environ.get('FLY_TRACE_DATA'),reason='requires prepared full graph')
def test_native_checkpoint_comparison_replays_ledgers_counts_and_news(tmp_path):
    from paperlab.fly_paper_study import run
    from paperlab.fly_paper_audit import audit
    env,package,news=native_fixture(tmp_path);root=tmp_path/'run'
    result=run(env,package,news,Path(os.environ['FLY_TRACE_DATA']),root)
    report,verified=audit(env,result,root)
    assert all(report['verification'].values())
    assert sum(p['full_count_decoders_verified'] for arms in verified['pools'].values() for phases in arms.values() for p in phases.values())==32
    assert len(list(root.glob('pool*/view.json')))==12
    raw=json.loads((root/'results.json').read_text());key=env['plan']['registration']['cohort'][0]
    for bad in ('fill','position','side','quote','news','time','learning','equity'):
        changed=copy.deepcopy(raw);row=changed[key]['trained_input_reset']['test']['rows'][1]
        if bad=='fill':row['fill']['status']='hold'
        if bad=='position':row['broker']['cash']='99999'
        if bad=='side':row['event']['side']='SELL' if row['event']['side']!='SELL' else 'BUY'
        if bad=='quote':row['quote_ts']-=1
        if bad=='news':row['event']['news_features'][0]+=.1
        if bad=='time':row['event']['market_decision_ts']+=1
        if bad=='learning':row['event']['plasticity_enabled']=True
        if bad=='equity':row['equity']+=1
        atomic_json(root/'results.json',changed)
        with pytest.raises(ValueError):audit(env,result,root)
    atomic_json(root/'results.json',raw)
    final=root/'boundaries/pool0-trained_input_reset-test/final-counts.npz'
    with np.load(final) as saved:counts=saved['counts'].copy()
    altered=counts.copy();altered[0]+=1;np.savez_compressed(final,counts=altered)
    with pytest.raises(ValueError,match='Full neuron counts'):audit(env,result,root)
    np.savez_compressed(final,counts=counts)
    view_path=root/'pool0-trained_input_reset/view.json';view=json.loads(view_path.read_text())
    changed=copy.deepcopy(view);changed['report']['training_exposure']['positive_rewards']+=1
    atomic_json(view_path,changed)
    with pytest.raises(ValueError,match='training exposure'):audit(env,result,root)
    atomic_json(view_path,view)
    # Leave a valid fixture available for browser QA.
    report,verified=audit(env,result,root);atomic_json(tmp_path/'report.json',report);atomic_json(tmp_path/'audit.json',verified)

    # A fresh observer process has not loaded the graph or configured upstream imports.
    import subprocess,sys
    subprocess.run([sys.executable,'-c',
        'import json,sys; from pathlib import Path; from paperlab.fly_paper_audit import audit; '
        'r=Path(sys.argv[1]); report,_=audit(json.loads((r/"protocol.json").read_text()), '
        'json.loads((r/"summary.json").read_text()),r); assert all(report["verification"].values())',
        str(root)],check=True)
