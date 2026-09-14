"""Unscheduled, narrowly scoped maintenance for the audited paper services."""
import modal
app=modal.App('fly-paper-audit-maintenance')
volume=modal.Volume.from_name('fly-paper-lab-state')
writers=modal.Dict.from_name('fly-paper-lab-writers')
image=modal.Image.debian_slim(python_version='3.12')

@app.function(image=image,volumes={'/state':volume},cpu=.125,memory=512,timeout=120,retries=0,single_use_containers=True)
def control(action: str, service: str, reason: str):
    import json,time,hashlib
    from pathlib import Path
    if service not in ('solana-online','checkpoint-eval') or action not in ('pause','resume','supersede_unexecuted'):
        raise ValueError('Invalid maintenance operation')
    key='solana-online-coordinator' if service=='solana-online' else 'checkpoint-eval-coordinator'
    owner=dict(call_id=modal.current_function_call_id(),started=time.time(),maintenance=True)
    if not writers.put(key,owner,skip_if_exists=True):return dict(status='coordinator_busy')
    try:
        volume.reload();path=Path('/state')/service/'control.json';c=json.loads(path.read_text());now=time.time()
        if action=='pause':
            if c.get('audit_pause'):return dict(status='already_paused',control=c)
            if not c.get('enabled'):raise ValueError('Preexisting pause must not be overwritten')
            c['audit_pause']=dict(at=now,reason=reason,previous_status=c.get('status'),previous_enabled=c['enabled'])
            c['enabled']=False;c['status']='audit_maintenance_pause'
        elif action=='supersede_unexecuted':
            if service!='checkpoint-eval' or not c.get('audit_pause') or c.get('pending'):raise ValueError('Must be audit paused and idle')
            base=path.parent/c['batch']
            if any(x.is_dir() for x in base.iterdir()):raise ValueError('Evaluation attempt exists; preserve batch')
            archive=base/'superseded.json'
            if archive.exists():raise ValueError('Never overwrite supersession record')
            archive.write_text(json.dumps(dict(at=now,reason=reason,plan_sha256=hashlib.sha256((base/'plan.json').read_bytes()).hexdigest()),indent=2))
            c.setdefault('superseded_batches',[]).append(dict(batch=c['batch'],reason=reason,at=now))
            c['batch']=None;c['next_batch_at']=0
        else:
            if not c.get('audit_pause') or c['enabled']:raise ValueError('Only own maintenance pause may be resumed')
            if c.get('budget_paused_until',0)>now:raise ValueError('Budget pause remains blocking')
            if c.get('error'):raise ValueError('Operational failure requires independent review')
            if c.get('pending'):
                try:outcome=modal.FunctionCall.from_id(c['pending']['call_id']).get(timeout=0)
                except TimeoutError:raise ValueError('Wait for active run before deployment verification')
                if outcome.get('status')!='completed':raise ValueError('Pending run did not complete successfully')
            c.setdefault('audit_maintenance_history',[]).append(dict(**c.pop('audit_pause'),resumed_at=now))
            c['enabled']=True;c['status']='audit_maintenance_resumed'
        temp=path.with_suffix('.audit-partial');temp.write_text(json.dumps(c,indent=2)+'\n');temp.replace(path);volume.commit()
        result=dict(status='updated',action=action,service=service,control=c);print(json.dumps(result));return result
    finally: writers.pop(key)
