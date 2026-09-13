"""Bounded seed-zero subprocesses and shared-budget accounting for study 12."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .budget import reserve,settle
from .core import atomic_json,digest

MAX_RESERVATION = .1608936
ALLOWANCE = 3.


def seal_seeded(archive,news,registration,audit,reference,output):
    script=('from paperlab.fly_rate_inputs import seal;import sys;'
            'seal(*sys.argv[1:])')
    subprocess.run([sys.executable,'-c',script,*map(str,(archive,news,registration,audit,reference,output))],
                   check=True,timeout=60,env={**os.environ,'PYTHONHASHSEED':'0'})
    return json.loads(Path(output).read_text())


def run_seeded(envelope,memory,news,data,output,*,stage,pool_index,arm,development=None,seconds=480):
    if not 1<=seconds<=480:raise ValueError('Bound the complete phase capture to 480 seconds')
    output=Path(output);atomic_json(output.parent/'inputs.json',envelope)
    script=('from paperlab.fly_rate_inputs import require_seed;require_seed();'
            'from paperlab.fly_rate_study import main;main()')
    command=[sys.executable,'-c',script,'--plan',str(output.parent/'inputs.json'),
             '--memory',str(memory),'--news',str(news),'--fly-data',str(data),'--out',str(output),
             '--stage',stage,'--pool',str(pool_index),'--arm',arm,'--seconds',str(seconds)]
    if development is not None:
        atomic_json(output.parent/'development-inputs.json',development)
        command+=['--development',str(output.parent/'development-inputs.json')]
    subprocess.run(command,check=True,timeout=min(510,seconds+30),env={**os.environ,'PYTHONHASHSEED':'0'})
    return json.loads((output/'summary.json').read_text())


def budgeted_execute(state,discovery,specifications,memory,*,call_id,input_id,commit,
                     run=run_seeded,due=None):
    from .fly_rate_schedule import DIRECTORY,execute_due
    if due is None:due=execute_due
    state=Path(state);root=state/DIRECTORY
    if not call_id or not input_id:raise ValueError('Study calls need actual cloud ownership')
    if (root/'halt.json').exists():
        return {'status':'paper_rate_halted','halt_sha256':digest(root/'halt.json')}
    if (root/'summary.json').exists():
        return {'status':'paper_rate_study_completed','summary_sha256':digest(root/'summary.json')}
    path=root/'costs.json';costs=json.loads(path.read_text()) if path.exists() else {}
    if call_id in costs:raise ValueError('Do not reuse an accounted study call')
    if sum(v['charged_usd'] for v in costs.values())+MAX_RESERVATION>ALLOWANCE:
        return {'status':'study_budget_stopped','allowance_usd':ALLOWANCE}
    reservation=reserve(state/'budget.json',True,seconds=1800,startup_seconds=30,
                        memory_gib=8,limit_override=25)
    if reservation is None:return {'status':'budget_stopped'}
    if abs(reservation['reserve']-MAX_RESERVATION)>1e-12:raise ValueError('Unexpected study reservation')
    reservation['rate']*=3;reservation['startup_seconds']=10
    costs[call_id]={'status':'reserved','charged_usd':reservation['reserve'],'reserved_usd':reservation['reserve'],
                    'input_id':input_id,'reserved_at':reservation['started'],'nonpreemptible':True,'price_multiplier':3}
    atomic_json(path,costs);commit()
    try:
        result=due(state,discovery,specifications,memory,call_id=call_id,input_id=input_id,
                   commit=commit,run=run,deadline=time.monotonic()+570)
        if result is None:raise ValueError('Study 12 registration is missing')
        budget=settle(state/'budget.json',reservation,time.time()-reservation['started'])
        costs[call_id].update(status='settled',charged_usd=budget['estimated_compute_usd'],
                             result_status=result['status'],completed_at=time.time())
        atomic_json(path,costs)
        result={**result,'call_id':call_id,'input_id':input_id,'budget':{**budget,
            'reserved_usd':reservation['reserve'],'study_accounted_usd':sum(v['charged_usd'] for v in costs.values()),
            'study_allowance_usd':ALLOWANCE,'nonpreemptible':True,'price_multiplier':3}}
        atomic_json(root/'last-call.json',result);commit();return result
    except BaseException as exc:
        costs[call_id].update(status='failed_or_interrupted',error_type=type(exc).__name__)
        atomic_json(path,costs)
        atomic_json(root/'halt.json',{'status':'halted','call_id':call_id,'input_id':input_id,
            'at':time.time(),'error_type':type(exc).__name__,'error':str(exc)[:300]})
        commit();raise
