import copy
import hashlib

import numpy as np
import pytest

from paperlab.core import digest
from paperlab.fly_paper_selection import expand


def fixture(root):
    counts=np.zeros((50,3),dtype=np.int32);counts[3,0]=1;counts[7,2]=2
    memory={'u':np.array([.1,.2],dtype=np.float32),'w':np.array([.05,.1],dtype=np.float32),
            'weights':np.array([1.05,1.1],dtype=np.float32)}
    arrays={k:np.tile(v,(50,1)) for k,v in memory.items()}
    arrays.update(neuron_ids=np.array([101,102,103]),plastic_edges=np.array([7,9]),
        plastic_pre=np.array([0,2]),plastic_post=np.array([1,1]),ms=np.arange(10,501,10,dtype=float),
        counts=counts,voltage=np.full((50,3),-60,dtype=np.float32),initial_weights=memory['weights'])
    arrays['voltage'][:,2]=-55
    np.savez_compressed(root/'step-01.npz',**arrays)
    boundary={p+'__'+k:np.array([1.,2.,3.],dtype=np.float32) for p in ('before','after') for k in ('v','g')}
    np.savez_compressed(root/'boundary-01.npz',**boundary)
    event={'brain_ms':500.,'spike_sha256':hashlib.sha256(counts.sum(axis=0,dtype=np.int32).tobytes()).hexdigest(),
           'activity_boundary':{'artifact_sha256':digest(root/'boundary-01.npz')}}
    frame={'event':event,'times_ms':arrays['ms'].tolist(),'counts':counts[:,:2].tolist(),
        'voltage':arrays['voltage'][:,:2].tolist(),'plastic_initial':memory['weights'][:1].tolist(),
        'memory_u_l2':np.linalg.norm(arrays['u'],axis=1).tolist(),'memory_w_l2':np.linalg.norm(arrays['w'],axis=1).tolist(),
        'activity_state':{p+'_'+k:boundary[p+'__'+k][:2].tolist() for p in ('before','after') for k in ('v','g')}}
    frame.update({out:arrays[k][:,:1].tolist() for k,out in [('weights','plastic_weights'),('u','plastic_u'),('w','plastic_w')]})
    nodes=[{'id':str(i+101),'index':i,'type':'fixture','group':'KC' if i!=1 else 'MBON','total_spikes':int(counts[:,i].sum())} for i in range(3)]
    view={'report':{'events':[event],'graph':{'neurons':3,'plastic_edges':2},'native_build':{'source':'fixture'},
        'upstream_commit':'fixture','plan_sha256':'a'*64},'nodes':copy.deepcopy(nodes[:2]),'selection':'Original subset.',
        'edges':[{'id':7,'source':0,'target':1,'weight':1.,'plastic_index':0}],'plastic_selection':[0],'frames':[frame]}
    reference=copy.deepcopy(view);reference['nodes']=nodes;reference['nodes'][2]['total_spikes']=999
    reference['edges'].append({'id':9,'source':2,'target':1,'weight':1.,'plastic_index':1})
    return view,reference,arrays,memory


def test_adds_missing_endpoint_and_this_runs_actual_memory_and_boundary(tmp_path):
    view,reference,arrays,memory=fixture(tmp_path);old=copy.deepcopy(view)
    out,meta=expand(view,reference,tmp_path,['9'],np.ones(2),memory,tmp_path)
    assert view==old and out['report']==view['report']
    assert meta['added_neurons']==['103'] and meta['added_edges']==['9']
    assert out['nodes'][-1]['total_spikes']==2  # Never use the reference brain's 999 spikes.
    assert out['edges'][-1]==reference['edges'][-1]
    f=out['frames'][0]
    assert f['counts']==arrays['counts'].tolist() and f['voltage']==arrays['voltage'].tolist()
    assert f['plastic_weights']==arrays['weights'].tolist() and f['plastic_u']==arrays['u'].tolist() and f['plastic_w']==arrays['w'].tolist()
    assert f['activity_state']['before_v']==[1.,2.,3.] and f['event']==old['frames'][0]['event']
    assert meta['trace_sha256']['step-01.npz']==digest(tmp_path/'step-01.npz')
    assert meta['boundary_sha256']['boundary-01.npz']==digest(tmp_path/'boundary-01.npz')


def test_reuses_existing_endpoint_and_connection_without_duplicate_columns(tmp_path):
    view,reference,_,memory=fixture(tmp_path)
    out,meta=expand(view,reference,tmp_path,['7'],np.ones(2),memory,tmp_path)
    assert out['nodes']==view['nodes'] and out['edges']==view['edges'] and out['plastic_selection']==[0]
    assert meta['added_neurons']==meta['added_edges']==[]


@pytest.mark.parametrize('mutation',['new_weight','new_endpoint','neuron_identity','pristine_weight','voltage','spikes','memory','boundary','missing_boundary','other_study','duplicate_edge','mixed_duplicate'])
def test_rejects_mismatched_data_instead_of_fabricating_a_paired_trace(tmp_path,mutation):
    view,reference,arrays,memory=fixture(tmp_path);boundaries=tmp_path;edges=['9']
    if mutation=='new_weight':arrays['weights'][0,1]+=1
    if mutation=='new_endpoint':arrays['plastic_pre'][1]=0
    if mutation=='neuron_identity':reference['nodes'][2]['id']='404'
    if mutation=='pristine_weight':reference['edges'][1]['weight']=2.
    if mutation=='voltage':arrays['voltage'][0,0]-=1
    if mutation=='spikes':arrays['counts'][0,2]+=1
    if mutation=='memory':view['frames'][0]['plastic_u'][0][0]+=1
    if mutation=='boundary':(tmp_path/'boundary-01.npz').write_bytes(b'corrupt')
    if mutation=='missing_boundary':boundaries=None
    if mutation=='other_study':reference['report']['plan_sha256']='b'*64
    if mutation=='duplicate_edge':edges=['9','9']
    if mutation=='mixed_duplicate':edges=[9,'9']
    np.savez_compressed(tmp_path/'step-01.npz',**arrays)
    with pytest.raises(ValueError):expand(view,reference,tmp_path,edges,np.ones(2),memory,boundaries)
