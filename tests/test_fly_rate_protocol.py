"""Prospective rate comparison rules, with explicitly synthetic registrations."""
import copy
from dataclasses import asdict

import pytest

from paperlab.fly_paper_protocol import MEMORY_FIELDS, NEWS
from paperlab.fly_rate_protocol import (ARMS,CANDIDATE,COHORT_POLICY,HYPOTHESIS,INFERENCE,
    SELECTION,HOSTING_ALLOCATION,chunk_order,chunk_name,development_choice,validate_registration)
from paperlab.multi import DEX_COSTS


def fixture():
    source={'memory_sha256':'a'*64,'memory_file_sha256':'b'*64,'checkpoint_sha256':'c'*64,
            'last_slot':5999,'observations':20,'positive_rewards':9,'negative_rewards':10}
    audit={'captured_at':1800001,'pools':{k:copy.deepcopy(source) for k in ('solana:a','solana:b')}}
    r={'schema':4,'kind':'paper_checkpoint_comparison','study':'12','arms':copy.deepcopy(ARMS),
       'phase_steps':24,'decision_seconds':300,'costs':asdict(DEX_COSTS),'hypothesis':HYPOTHESIS,
       'inference_protocol':INFERENCE,'selection_rule':SELECTION,'news_protocol':NEWS,
       'cohort_policy':COHORT_POLICY,'python_hash_seed':'0','recorded_at':1800010,
       'development_start':1800300,'test_start':1807500,'end':1814700,'training_cutoff':1800000,
       'checkpoint_capture_at':1800001,'parent_end':1790000,'cohort':list(audit['pools']),
       'source_memories':{k:{f:v[f] for f in MEMORY_FIELDS} for k,v in audit['pools'].items()}}
    r.update({k:'d'*64 for k in ('parent_report_sha256','mechanism_audit_sha256',
        'training_audit_sha256','capture_result_sha256','cohort_selection_sha256')})
    return r,audit


def test_registered_candidate_changes_only_eta_from_online_control():
    assert {**ARMS[CANDIDATE],'eta':.001}==ARMS['trained_online_carry']
    assert len(chunk_order())==16 and [c[0] for c in chunk_order()]==['development']*8+['test']*8
    for c in chunk_order():assert chunk_name(*c)
    assert HOSTING_ALLOCATION==pytest.approx(1/9)
    validate_registration(*fixture())


@pytest.mark.parametrize('field',['eta','reset','hypothesis','late','seed','cohort','memory','cutoff','test','provenance'])
def test_registration_rejects_protocol_and_timing_drift(field):
    r,a=fixture()
    if field=='eta':r['arms'][CANDIDATE]['eta']=.00001
    elif field=='reset':r['arms'][CANDIDATE]['activity_reset']='reset_rates'
    elif field=='hypothesis':r['hypothesis']='old stimulation hypothesis'
    elif field=='late':r['recorded_at']=r['development_start']
    elif field=='seed':r['python_hash_seed']='random'
    elif field=='cohort':r['cohort'][1]='solana:survivor'
    elif field=='memory':r['source_memories']['solana:a']['memory_sha256']='e'*64
    elif field=='cutoff':r['training_cutoff']-=300
    elif field=='test':r['test_start']-=300
    else:r['capture_result_sha256']='missing'
    with pytest.raises(ValueError):validate_registration(r,a)


def test_candidate_must_beat_cash_hosting_and_each_control_with_equal_coverage():
    equity={a:990. for a in ARMS};counts={a:[24,18] for a in ARMS}
    equity[CANDIDATE]=1000+HOSTING_ALLOCATION
    assert development_choice(equity,counts) is None
    equity[CANDIDATE]+=1
    assert development_choice(equity,counts)==CANDIDATE
    for control in ('pristine_frozen','trained_frozen','trained_online_carry'):
        other={**equity,control:equity[CANDIDATE]}
        assert development_choice(other,counts) is None
    assert development_choice(equity,{a:[24,17] for a in ARMS}) is None
    with pytest.raises(ValueError,match='coverage'):
        development_choice(equity,{**counts,CANDIDATE:[24,24]})
    with pytest.raises(ValueError,match='Finite'):
        development_choice({**equity,CANDIDATE:float('nan')},counts)
