"""Fresh interpreter, budgeting and actual cloud-entrypoint ordering checks."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import pytest

from paperlab.core import atomic_json
from paperlab.fly_rate_runtime import ALLOWANCE,MAX_RESERVATION,budgeted_execute,run_seeded,seal_seeded
from paperlab.fly_rate_schedule import DIRECTORY
from test_fly_rate_pipeline import sealed


def test_sealing_reconstructs_identical_inputs_in_fresh_seed_zero_process(sealed):
    env,root,ref=sealed
    result=seal_seeded(root/'universe.db',root/'news.db',root/'registration.json',ref/'audit.json',ref,root/'child-plan.json')
    assert result==env


def test_chunk_process_receives_seed_rate_arm_and_bounded_timeout(tmp_path,monkeypatch):
    calls=[];out=tmp_path/'chunk/artifacts'
    def run(command,**kw):
        calls.append(command);assert kw['env']['PYTHONHASHSEED']=='0' and kw['timeout']==450
        assert command[command.index('--arm')+1]=='trained_online_low_eta'
        assert command[command.index('--seconds')+1]=='420'
        out.mkdir(parents=True);atomic_json(out/'summary.json',{'synthetic':True})
    monkeypatch.setattr('paperlab.fly_rate_runtime.subprocess.run',run)
    assert run_seeded({'fixture':True},'memory','news','data',out,stage='test',pool_index=0,
                      arm='trained_online_low_eta',development={'fixture':True},seconds=420)=={'synthetic':True}
    assert len(calls)==1 and (out.parent/'development-inputs.json').exists()


def test_budget_reservation_is_committed_before_dispatch_and_settled_once(tmp_path):
    events=[]
    def due(*args,**kw):
        cost=json.loads((tmp_path/DIRECTORY/'costs.json').read_text())['fc-one']
        assert cost['status']=='reserved' and cost['charged_usd']==pytest.approx(MAX_RESERVATION)
        assert events[-1]=='commit' and 560<kw['deadline']-time.monotonic()<=570
        return {'status':'paper_rate_collecting'}
    result=budgeted_execute(tmp_path,'unused','unused','unused',call_id='fc-one',input_id='in-one',
                            commit=lambda:events.append('commit'),due=due)
    assert result['budget']['monthly_limit_usd']==25 and result['budget']['study_allowance_usd']==ALLOWANCE
    assert 0<result['budget']['estimated_compute_usd']<MAX_RESERVATION
    assert events==['commit','commit']
    with pytest.raises(ValueError,match='reuse'):
        budgeted_execute(tmp_path,'','','',call_id='fc-one',input_id='in-one',commit=lambda:None,due=due)


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt])
def test_interruption_keeps_worst_case_cost_and_halts_further_work(tmp_path,failure):
    attempts=[]
    def due(*a,**kw):attempts.append(1);raise failure('synthetic error')
    with pytest.raises(failure):
        budgeted_execute(tmp_path,'','','',call_id='fc-fail',input_id='in-fail',commit=lambda:None,due=due)
    cost=json.loads((tmp_path/DIRECTORY/'costs.json').read_text())['fc-fail']
    assert cost['charged_usd']==MAX_RESERVATION and cost['status']=='failed_or_interrupted'
    assert budgeted_execute(tmp_path,'','','',call_id='fc-next',input_id='in-next',commit=lambda:None,due=due)['status']=='paper_rate_halted'
    assert attempts==[1]


def test_study_allowance_prevents_another_reservation(tmp_path):
    atomic_json(tmp_path/DIRECTORY/'costs.json',{'fc-old':{'charged_usd':ALLOWANCE-.01}})
    assert budgeted_execute(tmp_path,'','','',call_id='fc-next',input_id='in-next',commit=lambda:None)['status']=='study_budget_stopped'
    assert not (tmp_path/'budget.json').exists()


def test_cloud_entrypoint_acquires_shared_lease_and_reloads_before_budgeted_work(tmp_path,monkeypatch):
    import cloud as base
    monkeypatch.setattr(base,'FULL_FLY',True);monkeypatch.setattr(base,'UNIVERSE',True)
    source=Path('rate_market_cloud.py').read_text();(tmp_path/'rate_market_cloud.py').write_text(source)
    (tmp_path/'reports').mkdir();(tmp_path/'runs/rate-market-reference-12').mkdir(parents=True)
    (tmp_path/'reports/fly-rate-market-registration-12.json').write_text('{}')
    spec=importlib.util.spec_from_file_location('rate_cloud_entrypoint_fixture',tmp_path/'rate_market_cloud.py')
    module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,module);spec.loader.exec_module(module)
    assert module._project_root(False,'/root/rate_market_cloud.py')==Path('/opt/paperlab')
    assert module._project_root(True,tmp_path/'rate_market_cloud.py')==tmp_path.resolve()
    events=[]
    class Volume:
        def __init__(self,name):self.name=name
        def reload(self):events.append('reload-'+self.name)
        def commit(self):events.append('commit')
    monkeypatch.setattr(module,'volume',Volume('state'));monkeypatch.setattr(module,'discovery_volume',Volume('discovery'))
    monkeypatch.setattr(module.modal,'current_function_call_id',lambda:'fc-cloud')
    monkeypatch.setattr(module.modal,'current_input_id',lambda:'in-cloud')
    def exclusive(name,fn):
        assert name=='worker';events.append('lease');return fn()
    def execute(*args,**kw):
        assert events==['lease','reload-state','reload-discovery']
        assert args==('/state','/discovery','/opt/paperlab/reports','/opt/paperlab/runs/rate-market-reference-12')
        assert kw['call_id']=='fc-cloud' and kw['input_id']=='in-cloud'
        kw['commit']();return {'synthetic':True}
    monkeypatch.setattr(module,'exclusive',exclusive)
    monkeypatch.setattr('paperlab.fly_rate_runtime.budgeted_execute',execute)
    assert module.worker.local()=={'synthetic':True} and events[-1]=='commit'
