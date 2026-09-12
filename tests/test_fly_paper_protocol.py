import json
from pathlib import Path

import pytest

from paperlab.core import digest
from paperlab.fly_paper_protocol import validate_registration,development_choice


def fixture():
    return (json.loads(Path('reports/fly-market-study-09-preregistration.json').read_text()),
            json.loads(Path('reports/fly-paper-memory-audit-01.json').read_text()))


def test_registered_checkpoint_comparison_pins_reward_exposure_before_evaluation():
    r,a=fixture();validate_registration(r,a)
    assert r['training_audit_sha256']==digest('reports/fly-paper-memory-audit-01.json')
    assert r['parent_plan_sha256']==json.loads(Path('reports/fly-market-study-08-plan.json').read_text())['sha256']
    assert r['parent_report_sha256']==digest('reports/fly-market-study-08.json')
    assert [p['observations'] for p in r['source_memories'].values()]==[82,72]


@pytest.mark.parametrize('bad',['late','cutoff','overlap','arms','news','selection','costs','cohort','memory','exposure','provenance'])
def test_changed_checkpoint_registration_is_rejected(bad):
    r,a=fixture();pool=r['cohort'][0]
    if bad=='late':r['recorded_at']=r['development_start']
    if bad=='cutoff':r['training_cutoff']+=300
    if bad=='overlap':r['test_start']-=300
    if bad=='arms':r['arms']['trained_frozen']['activity_reset']='conductance'
    if bad=='news':r['news_protocol']='headlines fetched during test'
    if bad=='selection':r['selection_rule']='use the best test return'
    if bad=='costs':r['costs']['fee_bps']=0
    if bad=='cohort':r['cohort']=[pool,pool]
    if bad=='memory':r['source_memories'][pool]['memory_sha256']='a'*64
    if bad=='exposure':
        r['source_memories'][pool]['positive_rewards']=0;a['pools'][pool]['positive_rewards']=0
    if bad=='provenance':r['training_audit_sha256']='missing'
    with pytest.raises(ValueError):validate_registration(r,a)


@pytest.mark.parametrize('carry,reset,pristine,expected',[
    (999,998,999,None),(1001,1002,1003,None),(1001,1002,1000,'trained_input_reset'),
    (1002,1001,1000,'trained_frozen'),(1002,1002,1000,'trained_frozen'),
    (1002,1002+5e-10,1000,'trained_frozen'),(1000,1000,999,None)])
def test_selection_uses_development_controls_and_registered_tie_order(carry,reset,pristine,expected):
    e={'pristine_frozen':pristine,'pristine_input_reset':pristine,'trained_frozen':carry,'trained_input_reset':reset}
    assert development_choice(e)==expected
    e['pristine_input_reset']=max(carry,reset)+1
    assert development_choice(e) is None
    e['trained_frozen']=float('nan')
    with pytest.raises(ValueError):development_choice(e)
