import json
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from paperlab.core import atomic_json,digest
from paperlab.fly_paper_memory import array_hash,exposure,export,verify_checkpoint


def fixture(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'stonkfly.neural.brain',SimpleNamespace(MODEL='test-model',PARAMETERS={'dt':.1}))
    b=SimpleNamespace(ids=np.array([10,20,30]),ptr=np.array([0,1,3,4]),post=np.array([1,0,2,1]),
        circuit={'edges':np.array([0,2])},eta=.001,weight=np.ones(4,dtype=np.float32),baseline_plastic=np.ones(2,dtype=np.float32),
        fields=['memory_u','memory_w','v'],memory_u=np.zeros(2),memory_w=np.zeros(2),v=np.zeros(3,dtype=np.float32),
        build={'model':'test-model','source_sha256':'a'*64,'binary_sha256':'b'*64,'flags':['-O3']},
        configuration_signature=lambda:{'initial_weights':'fixed'})
    m={'model':'test-model','parameters':{'dt':.1},'eta':.001,'graph_ids_sha256':array_hash(b.ids),
       'graph_ptr_sha256':array_hash(b.ptr),'graph_post_sha256':array_hash(b.post),'plastic_edges_sha256':array_hash(b.circuit['edges']),
       'configuration_sha256':b.configuration_signature(),'build':{**b.build,'binary_sha256':'c'*64},'weights_frozen':False,'cursor':10000}
    arrays={'weight':np.array([1.2,1,.9,1],dtype=np.float32),'memory_u':np.array([.05,-.1]),'memory_w':np.array([.2,-.1]),'v':np.full(3,-60,dtype=np.float32)}
    last={'slot':2,'brain_ms':1000,'reward':-.5,'memory_sha256':array_hash(arrays['weight'][b.circuit['edges']])}
    path=tmp_path/'checkpoint.npz'
    def save():np.savez_compressed(path,metadata=json.dumps(m),**arrays)
    save()
    return b,m,arrays,last,path,save


def test_export_verifies_weights_without_importing_cloud_dynamics(tmp_path,monkeypatch):
    b,m,arrays,last,path,_=fixture(tmp_path,monkeypatch)
    before={k:getattr(b,k).copy() for k in ('weight',*b.fields)}
    state,metadata=verify_checkpoint(b,path,digest(path),last)
    np.testing.assert_array_equal(state['weights'],arrays['weight'][b.circuit['edges']])
    assert metadata['build']['binary_sha256']!=b.build['binary_sha256']
    for k,v in before.items():np.testing.assert_array_equal(v,getattr(b,k))


@pytest.mark.parametrize('bad',['file_hash','graph','source','flags','config','clock','frozen','nonplastic','efficacy','ledger','nan','shape'])
def test_checkpoint_export_rejects_changed_provenance_or_weights(tmp_path,monkeypatch,bad):
    b,m,arrays,last,path,save=fixture(tmp_path,monkeypatch)
    if bad=='graph':m['graph_ids_sha256']='e'*64
    if bad=='source':m['build']['source_sha256']='e'*64
    if bad=='flags':m['build']['flags']=['-O0']
    if bad=='config':m['configuration_sha256']={'changed':True}
    if bad=='clock':m['cursor']+=1
    if bad=='frozen':m['weights_frozen']=True
    if bad=='nonplastic':arrays['weight'][1]+=1
    if bad=='efficacy':arrays['memory_w'][0]+=.1
    if bad=='ledger':last['memory_sha256']='d'*64
    if bad=='nan':arrays['v'][0]=float('nan')
    if bad=='shape':arrays['v']=np.zeros(4,dtype=np.float32)
    save()
    with pytest.raises(ValueError):verify_checkpoint(b,path,'0'*64 if bad=='file_hash' else digest(path),last)


def ledger(db,last):
    db.executescript('CREATE TABLE state(name TEXT PRIMARY KEY,payload TEXT); CREATE TABLE ledger(slot INTEGER,name TEXT,payload TEXT);')
    lanes=[{'pool':pool,'assigned':300,'checkpoint':f'/state/meme-pools-v1/fly-{i}-2.npz'} for i,pool in enumerate(('one','two'))]
    db.execute('INSERT INTO state VALUES(?,?)',('fly',json.dumps(lanes)))
    for slot in (1,2):
        events=[{'sleeve':i,'pool':pool,'detail':{'brain_ms':slot*500,'memory':{'sha256':last['memory_sha256']},
                'learning_diagnostics':{'learning_enabled':True,'equity_reward_usd':1 if slot==1 else -.5}}} for i,pool in enumerate(('one','two'))]
        db.execute('INSERT INTO ledger VALUES(?,?,?)',(slot,'fly',json.dumps({'sleeves':events})))
    db.commit();return lanes


def test_exposure_requires_complete_history_and_both_reward_signs(tmp_path,monkeypatch):
    *_,last,path,save=fixture(tmp_path,monkeypatch)
    db=sqlite3.connect(':memory:');ledger(db,last)
    assert len(exposure(db,'one',0,300))==2
    raw=json.loads(db.execute('SELECT payload FROM ledger WHERE slot=2').fetchone()[0])
    raw['sleeves'][0]['detail']['learning_diagnostics']['equity_reward_usd']=1
    db.execute('UPDATE ledger SET payload=? WHERE slot=2',(json.dumps(raw),))
    with pytest.raises(ValueError,match='positive and negative'):exposure(db,'one',0,300)
    db.execute('DELETE FROM ledger WHERE slot=1')
    with pytest.raises(ValueError,match='complete'):exposure(db,'one',0,300)
    db.close()


def test_two_checkpoint_export_links_database_and_rejects_overwrite(tmp_path,monkeypatch):
    b,m,arrays,last,path,save=fixture(tmp_path,monkeypatch)
    dbpath=tmp_path/'paper.db';db=sqlite3.connect(dbpath);lanes=ledger(db,last)
    capture={'captured_at':1000,'cohort':['one','two'],'pools':{}}
    for i,pool in enumerate(capture['cohort']):
        target=tmp_path/f'pool{i}-paper-checkpoint.npz';target.write_bytes(path.read_bytes())
        capture['pools'][pool]={'sleeve':i,'assigned':300,'remote_checkpoint':lanes[i]['checkpoint'],
          'checkpoint_sha256':digest(target),'events':exposure(db,pool,i,300),'observations':2}
    db.close();capture['ledger_sha256']=digest(dbpath);atomic_json(tmp_path/'capture.json',capture)
    out=tmp_path/'export';r=export(tmp_path,out,b)
    assert all(p['positive_rewards']==p['negative_rewards']==1 and p['changed_weights']==2 for p in r['pools'].values())
    for i in (0,1):
        with np.load(out/f'pool{i}-memory.npz') as saved:assert set(saved.files)=={'weights','u','w'}
    with pytest.raises(ValueError,match='overwrite'):export(tmp_path,out,b)
    capture['pools']['one']['events'][0]['reward']=100;atomic_json(tmp_path/'capture.json',capture)
    with pytest.raises(ValueError,match='independent ledger'):export(tmp_path,tmp_path/'bad',b)
    assert not (tmp_path/'bad').exists()
