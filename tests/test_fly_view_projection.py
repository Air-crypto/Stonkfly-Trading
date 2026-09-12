import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from paperlab.fly_trace_memory import enrich_frame
from paperlab.fly_view_projection import (audit_image, audit_view, array_digest,
    frame, load_graph, norm_series, run, topology)


def fixture():
    g=SimpleNamespace(ids=np.arange(10,15),ptr=np.array([0,2,3,3,4,4]),
        post=np.array([2,1,2,4]),weight=np.array([2.,-.25,1.,.5],dtype=np.float32),
        types=np.array(['KC','DAN','MBON','output','visual']),
        groups={'visual':np.array([4]),'KC':np.array([0]),'DAN':np.array([1]),
                'MBON':np.array([2]),'output':np.array([3])},
        circuit={'edges':np.array([0]),'dan':np.array([1])})
    counts=np.zeros((50,5),dtype=np.int32);counts[::3,0]=1;counts[5:15,1]=2
    weights=np.linspace(2,3,50,dtype=np.float32)[:,None];u=np.arange(50,dtype=np.float64)[:,None]/100
    a={'neuron_ids':g.ids,'plastic_edges':g.circuit['edges'],'plastic_pre':np.array([0]),
        'plastic_post':np.array([2]),'ms':np.arange(10,501,10,dtype=np.float64),'counts':counts,
        'voltage':np.full((50,5),-50,dtype=np.float32),'weights':weights,'u':u,'w':u/2,
        'kc':u+1,'dan':u+2,'initial_weights':np.array([2.],dtype=np.float32)}
    event={'brain_ms':500.,'total_spikes':int(counts.sum()),'spike_sha256':array_digest(counts.sum(axis=0,dtype=np.int32))}
    f={'step':1,'event':event,'times_ms':a['ms'].tolist(),'counts':counts.tolist(),'voltage':a['voltage'].tolist(),
        'total_spikes':counts.sum(axis=1).tolist(),'groups':{k:counts[:,v].sum(axis=1).tolist() for k,v in g.groups.items()},
        'weight_delta_l2':np.linalg.norm(weights-a['initial_weights'],axis=1).tolist(),
        'weight_from_pristine_l2':np.linalg.norm(weights-g.weight[[0]],axis=1).tolist(),
        'changed_edges':np.count_nonzero(weights-a['initial_weights'],axis=1).tolist(),
        'changed_from_pristine':np.count_nonzero(weights-g.weight[[0]],axis=1).tolist(),
        'plastic_weights':weights.tolist(),'plastic_initial':a['initial_weights'].tolist(),
        'plastic_u':u.tolist(),'plastic_w':a['w'].tolist(),
        'kc_mean_hz':a['kc'].mean(axis=1).tolist(),'dan_mean_hz':a['dan'].mean(axis=1).tolist(),
        'memory_u_l2':np.linalg.norm(u,axis=1).tolist(),'memory_w_l2':np.linalg.norm(a['w'],axis=1).tolist()}
    v={'report':{'graph':{'neurons':5,'edges':4,'plastic_edges':1}},
        'nodes':[{'id':str(identity),'index':i,'type':str(g.types[i]),'group':str(g.types[i])} for i,identity in enumerate(g.ids)],
        'edges':[{'id':i,'source':source,'target':int(g.post[i]),'weight':float(g.weight[i]),'plastic_index':0 if i==0 else None}
            for i,source in enumerate([0,0,1,3])], 'plastic_selection':[0]}
    return g,v,f,a


def test_valid_projection_uses_actual_topology_and_arrays():
    g,v,f,a=fixture();columns=topology(v,g);total,errors=frame(v,f,a,g,columns,1)
    np.testing.assert_array_equal(total,a['counts'].sum(axis=0));assert max(errors.values())<1e-7


@pytest.mark.parametrize('field',['total_spikes','weight_delta_l2','weight_from_pristine_l2',
    'changed_edges','changed_from_pristine','kc_mean_hz','dan_mean_hz'])
def test_previously_unchecked_chart_fields_cannot_disagree_with_trace(tmp_path,field):
    g,v,f,a=fixture();f[field][0]+=1;path=tmp_path/'step.npz';np.savez(path,**a)
    # Reproduce the gap: selected-cell/connection enrichment alone accepts this.
    assert enrich_frame(v,f,path)['plastic_u']==a['u'].tolist()
    with pytest.raises(ValueError):frame(v,f,a,g,topology(v,g),1)


@pytest.mark.parametrize('change',['target','source','weight','missing','duplicate','plastic_flag'])
def test_nonplastic_connections_are_checked_too(tmp_path,change):
    g,v,f,a=fixture();edge=v['edges'][1]
    if change=='target':edge['target']=4
    elif change=='source':edge['source']=3
    elif change=='weight':edge['weight']=5
    elif change=='missing':v['edges'].pop(1)
    elif change=='duplicate':v['edges'].append(copy.deepcopy(edge))
    else:edge['plastic_index']=0
    if change!='plastic_flag':
        path=tmp_path/'step.npz';np.savez(path,**a)
        enrich_frame(v,f,path)  # The original helper intentionally only covers plastic edges.
    with pytest.raises(ValueError):topology(v,g)


@pytest.mark.parametrize('change',['type','group','index','id','selection','duplicate_node'])
def test_neuron_and_selection_identity_is_strict(change):
    g,v,_,_=fixture()
    if change=='selection':v['plastic_selection']=[0,0]
    elif change=='duplicate_node':v['nodes'][1]=copy.deepcopy(v['nodes'][0])
    elif change=='index':v['nodes'][0]['index']=0.0
    else:v['nodes'][0][change]='wrong'
    with pytest.raises(ValueError):topology(v,g)


@pytest.mark.parametrize('change',['counts','voltage','group','plastic_w','memory_u_l2','ordinal','time','event_total','nan_voltage'])
def test_frame_corruption_is_rejected(change):
    g,v,f,a=fixture()
    if change=='group':f['groups']['DAN'][0]+=1
    elif change=='ordinal':f['step']=2
    elif change=='time':f['times_ms'][0]+=1
    elif change=='event_total':f['event']['total_spikes']+=1
    elif change=='nan_voltage':a['voltage'][0,0]=float('nan')
    elif change=='memory_u_l2':f[change][0]+=1
    else:f[change][0][0]+=1
    with pytest.raises(ValueError):frame(v,f,a,g,topology(v,g),1)


def test_retimed_spikes_preserve_totals_but_change_highlights():
    g,v,f,a=fixture();a['counts'][0,0]-=1;a['counts'][1,0]+=1
    assert array_digest(a['counts'].sum(axis=0,dtype=np.int32))==f['event']['spike_sha256']
    with pytest.raises(ValueError,match='spike counts'):frame(v,f,a,g,topology(v,g),1)


def test_norm_roundoff_is_distinct_from_a_wrong_curve():
    delta=np.array([[1,2,3]],dtype=np.float32);value=np.float32(np.linalg.norm(delta.astype(float)))
    norm_series([float(np.nextafter(value,np.float32(0)))],delta,'norm')
    for wrong in ([float(value)*1.01],[float('nan')],[-1],[]):
        with pytest.raises(ValueError):norm_series(wrong,delta,'norm')


@pytest.mark.parametrize('change',[None,'hash','saved','displayed'])
def test_sealed_image_matches_both_saved_and_displayed_png(tmp_path,change):
    rgb=np.arange(12,dtype=np.uint8).reshape(2,2,3);path=tmp_path/'input.png';Image.fromarray(rgb).save(path)
    f={'event':{'input_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()},'input_png':base64.b64encode(path.read_bytes()).decode()}
    if change=='hash':f['event']['input_sha256']='wrong'
    if change in ('saved','displayed'):
        different=rgb+1;buf=io.BytesIO();Image.fromarray(different).save(buf,format='PNG')
        if change=='saved':path.write_bytes(buf.getvalue())
        else:f['input_png']=base64.b64encode(buf.getvalue()).decode()
    if change is None:audit_image(f,path,rgb)
    else:
        with pytest.raises(ValueError):audit_image(f,path,rgb)


def test_retained_projection_audit_never_constructs_native_brain(tmp_path,monkeypatch):
    recordings=os.environ.get('FLY_CREDIT_RESET_RECORDINGS');data=os.environ.get('FLY_TRACE_DATA')
    if not recordings or not data:pytest.skip('Requires retained full arrays and prepared graph; no neural propagation')
    from paperlab.fly import configure
    configure(data)
    from stonkfly.neural.brain import MemoryBrain
    def forbidden(*a,**kw):raise AssertionError('Projection checks must not construct a neural model')
    monkeypatch.setattr(MemoryBrain,'__init__',forbidden)
    r=run(Path(recordings)/'audit.json',Path(recordings)/'artifacts',data,tmp_path/'out')
    assert len(r['views'])==6 and sum(v['bins'] for v in r['views'].values())==900
    assert r['neural_observations']==r['cloud_submissions']==0
    for v in r['views'].values():assert v['all_topology_verified'] and v['all_plotted_series_verified']
