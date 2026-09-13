"""Future registration and exact cloud-witness checks; no remote work or brain."""
import copy
import json
from pathlib import Path

import pytest

from paperlab.core import atomic_json,digest
from paperlab.fly_market_study import signature
from paperlab.fly_rate_cloud import prepare,verify_arming
from paperlab.fly_rate_inputs import verify_reference


@pytest.fixture
def prepared(tmp_path):
    audit=json.loads(Path('reports/fly-rate-checkpoint-memory-audit-12.json').read_text())
    now=audit['captured_at']+60
    registration=tmp_path/'registration.json';reference=tmp_path/'reference'
    result=prepare(registration,reference,now=now)
    return result,registration,reference,now


def test_prepare_uses_verified_checkpoint_and_future_windows(prepared):
    result,path,reference,now=prepared;r=result['registration']
    assert now+900<=r['development_start']<now+1200
    assert r['development_start']%300==0 and r['end']-r['development_start']==14400
    assert result['registration_sha256']==digest(path)
    audit=json.loads((reference/'audit.json').read_text())
    assert verify_reference(r,audit,reference)
    assert r['preparation_source_sha256']==digest('paperlab/fly_rate_cloud.py')
    assert result['source_sha256']['paperlab/fly_rate_cloud.py']==r['preparation_source_sha256']
    with pytest.raises(ValueError,match='Preserve'):
        prepare(path,reference,now=now)


@pytest.mark.parametrize('start',[0,1789277101,float('nan'),float('inf'),True])
def test_prepare_refuses_late_unaligned_or_invalid_start(tmp_path,start):
    with pytest.raises(ValueError,match='ten minutes'):
        prepare(tmp_path/'reg',tmp_path/'ref',start=start,now=1789277100)
    assert not (tmp_path/'ref').exists()


@pytest.mark.parametrize('corruption',[None,'source','signature','bytes','late','owner','content','nan'])
def test_cloud_witness_matches_local_bytes_sources_time_and_owner(prepared,corruption):
    result,path,_,now=prepared;r=result['registration'];sources=result['source_sha256']
    files={'preregistration.json':copy.deepcopy(r),'source-hashes.json':copy.deepcopy(sources),
           'armed.json':{'registration_sha256':digest(path),'registration_signature':signature(r),
                         'source_sha256':signature(sources),'armed_at':now+1,'call_id':'fc-one','input_id':'in-one'}}
    hashes={'preregistration.json':digest(path)}
    if corruption=='source':files['source-hashes.json']['cloud.py']='f'*64
    if corruption=='signature':files['armed.json']['registration_signature']='f'*64
    if corruption=='bytes':hashes['preregistration.json']='f'*64
    if corruption=='late':files['armed.json']['armed_at']=r['development_start']
    if corruption=='owner':files['armed.json']['call_id']=''
    if corruption=='content':files['preregistration.json']['recorded_at']-=1
    if corruption=='nan':files['armed.json']['armed_at']=float('nan')
    checked=verify_arming(files,hashes,path,sources)
    assert checked['verified']==(corruption is None)


def test_witness_cannot_be_confirmed_without_prepared_registration(tmp_path):
    assert not verify_arming({}, {}, tmp_path/'missing', {})['verified']
