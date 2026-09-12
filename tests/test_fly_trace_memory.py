import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from paperlab.fly_trace_memory import enrich,enrich_frame


def fixture(root):
    counts=np.zeros((50,3),dtype=np.int32);counts[3,0]=1;counts[7,2]=2
    u=np.tile(np.array([.1,.2],dtype=np.float32),(50,1));w=u*.5;weights=1+w
    a={'neuron_ids':np.array([101,102,103]),'plastic_edges':np.array([7,9]),'plastic_pre':np.array([0,1]),'plastic_post':np.array([1,2]),
       'ms':np.arange(10,501,10,dtype=float),'counts':counts,'weights':weights,'initial_weights':np.ones(2,dtype=np.float32),'u':u,'w':w}
    frame={'event':{'brain_ms':500.,'spike_sha256':hashlib.sha256(counts.sum(axis=0,dtype=np.int32).tobytes()).hexdigest()},
        'times_ms':a['ms'].tolist(),'counts':counts.tolist(),'plastic_weights':weights[:,[1]].tolist(),'plastic_initial':[1.],
        'memory_u_l2':np.linalg.norm(u,axis=1).tolist(),'memory_w_l2':np.linalg.norm(w,axis=1).tolist()}
    v={'report':{'events':[frame['event']],'graph':{'neurons':3,'plastic_edges':2}},'nodes':[{'index':i,'id':str(i+101)} for i in range(3)],
        'edges':[{'id':9,'source':1,'target':2,'plastic_index':1}],'plastic_selection':[1],'frames':[frame]}
    p=root/'step-01.npz';np.savez_compressed(p,**a);src=root/'source.json';src.write_text(json.dumps(v));return v,a,p,src


def test_enrichment_preserves_original_and_copies_the_identified_connection(tmp_path):
    view,a,p,src=fixture(tmp_path);original=src.read_bytes();out=tmp_path/'enriched.json'
    meta=enrich(src,tmp_path,out);result=json.loads(out.read_text())
    assert src.read_bytes()==original and result['report']==view['report']
    f=result['frames'][0]
    assert f['event']==view['frames'][0]['event'] and f['plastic_weights']==view['frames'][0]['plastic_weights']
    assert f['plastic_u']==a['u'][:,[1]].tolist() and f['plastic_w']==a['w'][:,[1]].tolist()
    assert meta['source_view_sha256']==hashlib.sha256(original).hexdigest()
    assert meta['trace_sha256']['step-01.npz']==hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='never overwrite'):enrich(src,tmp_path,src)
    with pytest.raises(ValueError,match='never overwrite'):enrich(src,tmp_path,out)


def test_partial_enrichment_is_explicit_and_never_fills_missing_memory_with_zero(tmp_path):
    v,_,_,src=fixture(tmp_path);v['frames'].append(v['frames'][0]);v['report']['events'].append(v['report']['events'][0]);src.write_text(json.dumps(v))
    out=tmp_path/'enriched.json'
    with pytest.raises(ValueError,match='Missing full trace'):enrich(src,tmp_path,out)
    assert not out.exists()
    meta=enrich(src,tmp_path,out,allow_partial=True)
    assert meta['included_observations']==[1] and meta['missing_observations']==[2]
    assert 'plastic_u' not in json.loads(out.read_text())['frames'][1]


@pytest.mark.parametrize('mutation',['time','full_spikes','identity','endpoint','weight','memory','nonfinite','shape','graph'])
def test_enrichment_rejects_a_mismatched_full_trace(tmp_path,mutation):
    v,a,p,_=fixture(tmp_path)
    if mutation=='time':a['ms'][0]+=1
    if mutation=='full_spikes':a['counts'][0,0]+=1
    if mutation=='identity':a['plastic_edges'][1]=10
    if mutation=='endpoint':a['plastic_post'][1]=0
    if mutation=='weight':a['weights'][0,1]+=1
    if mutation=='memory':a['u'][0,1]+=1
    if mutation=='nonfinite':a['w'][0,1]=float('nan')
    if mutation=='shape':a['u']=a['u'][:,:1]
    if mutation=='graph':v['report']['graph']['plastic_edges']=3
    np.savez_compressed(p,**a)
    with pytest.raises(ValueError):enrich_frame(v,v['frames'][0],p)
