import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from paperlab.cloud_debug import validate_request
from paperlab.fly_market_pulse import ARMS, input_sequences, same_memory, validate


def payload():
    return {'protocol':json.loads(Path('reports/fly-market-pulse-protocol-01.json').read_text()),
            'reference_json':Path('reports/fly-market-study-04.json').read_text(),
            'market_plan':json.loads(Path('reports/fly-market-study-04-plan.json').read_text())}


def test_registered_factorial_plan_preserves_all_eight_controls():
    p=payload();protocol,reference,plan=validate(p)
    assert len(ARMS)==8
    assert {tuple(a.values()) for a in ARMS.values()}=={(m,l,s) for m in ('pristine','trained') for l in (False,True) for s in ('none','recorded')}
    assert validate_request({'run_id':'assay-pulse-test','pulse_plan':p})==p
    assert protocol['pool'] in plan['cohort']


@pytest.mark.parametrize('mutation',['path','arms','reference','plan','pool','missing_observation','pulse_schedule'])
def test_pulse_plan_rejects_changed_or_incomplete_reference(mutation):
    p=payload()
    if mutation=='path':p['path']='/state/account.db'
    if mutation=='arms':p['protocol']['arms'].pop('trained_online_none')
    if mutation=='reference':p['reference_json']+=' '
    if mutation=='plan':p['market_plan']['plan']['start']+=300
    if mutation=='pool':p['protocol']['pool']='absent'
    if mutation in ('missing_observation','pulse_schedule'):
        r=json.loads(p['reference_json']);rows=r['phase_diagnostics'][p['protocol']['pool']]['online_original']['test']['decisions']
        if mutation=='missing_observation':rows[0]['neural']=None
        else:rows[2]['neural']['stimulus']='none'
        p['reference_json']=json.dumps(r)
        p['protocol']['reference_report_sha256']=hashlib.sha256(p['reference_json'].encode()).hexdigest()
    with pytest.raises(ValueError):validate(p)


def test_replay_inputs_must_reconcile_to_quote_and_image_hash(monkeypatch):
    p=payload();protocol,reference,plan=validate(p)
    rgb=np.zeros((180,320,3),dtype=np.uint8)
    monkeypatch.setattr('paperlab.fly_market_pulse.frame',lambda *args:rgb)
    reference=copy.deepcopy(reference)
    for phase in ('training','test'):
        for row in reference['phase_diagnostics'][protocol['pool']]['online_original'][phase]['decisions'][:3]:
            row['neural']['input_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
    sequences=input_sequences(plan,reference,protocol['pool'])
    assert all(len(rows)==3 for rows in sequences.values())
    row=reference['phase_diagnostics'][protocol['pool']]['online_original']['test']['decisions'][0]
    row['neural']['input_sha256']='altered'
    with pytest.raises(ValueError,match='image'):input_sequences(plan,reference,protocol['pool'])
    row['quote_ts']-=1
    with pytest.raises(ValueError,match='quote'):input_sequences(plan,reference,protocol['pool'])


def test_frozen_memory_check_includes_efficacy_not_just_weights():
    from types import SimpleNamespace
    b=SimpleNamespace(weight=np.array([1.,2.]),circuit={'edges':np.array([0,1])},memory_u=np.zeros(2),memory_w=np.zeros(2))
    state={'weights':b.weight.copy(),'u':b.memory_u.copy(),'w':b.memory_w.copy()}
    assert same_memory(b,state)
    b.memory_u[0]=1
    assert not same_memory(b,state)


def test_cloud_pulse_observer_resumes_one_call_and_rejects_new_payload(tmp_path,monkeypatch):
    import modal
    from types import SimpleNamespace
    from paperlab.fly_market_pulse import cloud_run
    calls=[]
    class Remote:
        object_id='fc-pulse-test'
        def get(self,timeout=0):raise TimeoutError('pending')
    remote=Remote()
    def spawn(**kw):calls.append(kw);return remote
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**kw:SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(modal.FunctionCall,'from_id',lambda identity:remote)
    p=payload();cloud_run(p,tmp_path);cloud_run(p,tmp_path)
    assert len(calls)==1
    p['protocol']['registered_at']+=1
    with pytest.raises(ValueError,match='another payload'):cloud_run(p,tmp_path)
    assert len(calls)==1


def test_uncertain_pulse_submission_never_resubmits(tmp_path,monkeypatch):
    import modal
    from types import SimpleNamespace
    from paperlab.fly_market_pulse import cloud_run
    calls=[]
    def spawn(**kw):calls.append(kw);raise ConnectionError('lost response')
    monkeypatch.setattr(modal.Function,'from_name',lambda *a,**kw:SimpleNamespace(spawn=spawn))
    with pytest.raises(ConnectionError):cloud_run(payload(),tmp_path)
    with pytest.raises(RuntimeError,match='outcome unknown'):cloud_run(payload(),tmp_path)
    assert len(calls)==1
