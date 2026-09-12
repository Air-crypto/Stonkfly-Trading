import json
from pathlib import Path

import pytest

from paperlab.core import atomic_json, digest
from paperlab.fly_online_coverage import coverage, run
from paperlab.fly_online_cloud import expected_sources
from paperlab.universe import Pool, Store


@pytest.fixture
def registration():
    return json.loads(Path('reports/fly-market-study-11-preregistration.json').read_text())


def snapshot(path,r,end,*,omit=lambda key,t:False,bad=lambda key,t:False):
    store=Store(path);start=r['development_start']
    for t in range(start-120*60,end+1,60):
        for index,key in enumerate(r['cohort']):
            if omit(key,t):continue
            pool=Pool(key,'solana',key.split(':')[1],f'token{index}',f'pool{index}',1,50000,
                      0 if bad(key,t) else 2000,5,5,start-20000,t)
            store.add([pool])
    store.db.close()
    return path


def test_future_slots_are_unknown_and_elapsed_quotes_are_usable(registration,tmp_path):
    r=registration;start=r['development_start'];p=snapshot(tmp_path/'snapshot.db',r,start+7*300+60)
    result=coverage(r,p)
    for pool in result['pools'].values():
        assert pool['context_requirement_met']
        d=pool['phases']['development'];t=pool['phases']['test']
        assert (d['observed'],d['missing'],d['pending'])==(8,0,16)
        assert (t['observed'],t['missing'],t['pending'])==(0,0,24)
        assert t['rows'][0]['quote_ts'] is None and t['rows'][0]['neural_observation_expected'] is None
        assert d['coverage_gate']=='pending'


def test_later_receipts_do_not_change_prior_assessments_and_terminal_is_not_neural(registration,tmp_path):
    r=registration;start=r['development_start']
    a=coverage(r,snapshot(tmp_path/'early.db',r,start+7*300+60))
    b=coverage(r,snapshot(tmp_path/'complete.db',r,r['end']+60))
    for key in r['cohort']:
        assert a['pools'][key]['phases']['development']['rows'][:8]==b['pools'][key]['phases']['development']['rows'][:8]
        for phase in b['pools'][key]['phases'].values():
            assert (phase['observed'],phase['missing'],phase['pending'])==(24,0,0)
            assert phase['coverage_gate']=='met' and phase['rows'][-1]['status']=='terminal_mark'
            assert phase['rows'][-1]['neural_observation_expected'] is False


def test_fresh_ineligible_quotes_can_make_coverage_threshold_unreachable(registration,tmp_path):
    r=registration;start=r['development_start'];key=r['cohort'][0]
    p=snapshot(tmp_path/'snapshot.db',r,start+7*300+60,bad=lambda k,t:k==key and start<=t<start+7*300)
    d=coverage(r,p)['pools'][key]['phases']['development']
    assert (d['observed'],d['missing'],d['pending'])==(1,7,16)
    assert d['coverage_gate']=='threshold_unreachable' and d['maximum_possible_observations']==17
    assert d['rows'][0]['status']=='ineligible_quote'
    assert d['rows'][0]['raw_rejection']=='insufficient_two_sided_activity'
    assert d['rows'][0]['raw_receipt_age_seconds']==0


def test_gap_marker_does_not_hide_age_of_actual_receipt(registration,tmp_path):
    r=registration;start=r['development_start'];key=r['cohort'][0]
    p=snapshot(tmp_path/'snapshot.db',r,start+300,omit=lambda k,t:k==key and t>start-240)
    row=coverage(r,p)['pools'][key]['phases']['development']['rows'][0]
    assert row['status']=='stale_or_missing_quote' and row['raw_receipt_age_seconds']==240
    assert row['raw_receipt_ts']==start-240 and row['quote_ts']==start-60
    assert not row['quote_available']


def test_missing_fixed_pool_is_retained_as_missing(registration,tmp_path):
    r=registration;key=r['cohort'][0]
    p=snapshot(tmp_path/'snapshot.db',r,r['development_start']+60,omit=lambda k,t:k==key)
    result=coverage(r,p);assert set(result['pools'])==set(r['cohort'])
    pool=result['pools'][key]
    assert not pool['context_requirement_met'] and pool['symbol'] is None
    assert pool['phases']['development']['rows'][0]['status']=='missing_prior_context'


def test_tampered_receipt_eligibility_is_rejected(registration,tmp_path):
    p=snapshot(tmp_path/'snapshot.db',registration,registration['development_start']+60)
    store=Store(p);store.db.execute("UPDATE observations SET reason='changed'");store.db.commit();store.db.close()
    with pytest.raises(ValueError,match='eligibility'):coverage(registration,p)


def test_registration_is_pinned_and_earlier_report_is_preserved(registration,tmp_path):
    p=snapshot(tmp_path/'snapshot.db',registration,registration['development_start']+60)
    r=tmp_path/'registration.json';a=tmp_path/'arming.json';out=tmp_path/'result.json'
    atomic_json(r,registration);atomic_json(a,{'registration_sha256':'0'*64})
    with pytest.raises(ValueError,match='Registration differs'):run(r,a,p,out)
    atomic_json(a,{'registration_sha256':digest(r),'source_hashes':{}})
    with pytest.raises(ValueError,match='Execution sources differ'):run(r,a,p,out)
    atomic_json(a,{'registration_sha256':digest(r),'source_hashes':expected_sources()});run(r,a,p,out);sha=digest(out)
    with pytest.raises(ValueError,match='Preserve'):run(r,a,p,out)
    assert digest(out)==sha
