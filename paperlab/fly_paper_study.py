"""Frozen future evaluation of verified paper-trained memory; no account writes."""
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .core import Tick,atomic_json,digest
from .fly_market_activity import dynamic_state
from .fly_market_memory_audit import read_memory
from .fly_market_study import phase,learned_state,restore_learned,memory_signature
from .fly_paper_inputs import validate,SealedNews,audit_news
from .fly_paper_memory import array_hash
from .fly_paper_protocol import development_choice
from .fly_trace import TraceLab

SOURCE_FILES=('fly_paper_study.py','fly_paper_inputs.py','fly_paper_protocol.py','fly_paper_memory.py',
              'fly_paper_audit.py','fly_market_study.py','fly_trace.py','fly.py','fly_visual.py','news.py',
              'fly_market_activity.py','fly_market_activity_audit.py','fly_market_pulse.py',
              'fly_market_restoration.py','fly_market_memory_audit.py','core.py','multi.py',
              'fly_activation_protocol.py','fly_paper_stimulation.py')
PHASES=('development','test')


def trace_name(i,arm,phase):return f'pool{i}-{arm}'+('-development' if phase=='development' else '')


def imported_memory(brain,plan,memory_root):
    from stonkfly.neural.brain import PARAMETERS
    root=Path(memory_root);r=plan['registration'];audit=plan['training_audit']
    if digest(root/'audit.json')!=r['training_audit_sha256'] or json.loads((root/'audit.json').read_text())!=audit:
        raise ValueError('Imported training audit differs from registration')
    states={};edges=brain.circuit['edges']
    for i,key in enumerate(r['cohort']):
        source=audit['pools'][key];metadata=source['source_metadata'];pinned=r['source_memories'][key]
        expected={'graph_ids_sha256':array_hash(brain.ids),'graph_ptr_sha256':array_hash(brain.ptr),
                  'graph_post_sha256':array_hash(brain.post),'plastic_edges_sha256':array_hash(edges),
                  'configuration_sha256':brain.configuration_signature(),'eta':brain.eta,'model':brain.build['model'],'parameters':PARAMETERS}
        if any(metadata.get(k)!=v for k,v in expected.items()) or any(metadata['build'].get(k)!=brain.build.get(k) for k in ('model','source_sha256','flags')):
            raise ValueError('Imported memory graph, kernel or configuration differs')
        path=root/f'pool{i}-memory.npz'
        if digest(path)!=pinned['memory_file_sha256']:raise ValueError('Imported memory file hash differs')
        state=read_memory(path)
        if memory_signature(state)!=pinned['memory_sha256']:raise ValueError('Imported memory arrays differ')
        for name,reference in (('weights',brain.baseline_plastic),('u',brain.memory_u),('w',brain.memory_w)):
            if state[name].shape!=reference.shape or state[name].dtype!=reference.dtype:raise ValueError('Imported memory shape or dtype differs')
        if not np.array_equal(state['weights'],(brain.baseline_plastic*(1+state['w'])).astype(brain.weight.dtype)):
            raise ValueError('Imported stored efficacy does not reproduce the weights')
        states[key]=state
    return states


def run(envelope,memory_root,news_archive,data,output):
    p=validate(envelope);r=p['registration'];root=Path(output)
    if root.exists():raise ValueError('Refuse to overwrite a checkpoint comparison')
    # Check news provenance before loading the full graph or simulating a decision.
    news_audit=audit_news(envelope,news_archive)
    root.mkdir(parents=True);atomic_json(root/'protocol.json',envelope)
    shutil.copyfile(news_archive,root/'news.db')
    deadline=time.monotonic()+480;lab=TraceLab(data);b=lab.brain
    imported=imported_memory(b,p,memory_root);pristine=learned_state(b)
    np.savez_compressed(root/'pristine-memory.npz',**pristine)
    np.savez_compressed(root/'initial-dynamics.npz',**dynamic_state(b))
    np.savez_compressed(root/'neuron-ids.npz',neuron_ids=b.ids)
    (root/'imported').mkdir();shutil.copyfile(Path(memory_root)/'audit.json',root/'imported/audit.json')
    for i in range(len(r['cohort'])):shutil.copyfile(Path(memory_root)/f'pool{i}-memory.npz',root/f'imported/pool{i}-memory.npz')
    results={key:{name:{} for name in r['arms']} for key in r['cohort']};news=SealedNews(p['news_features'])
    for phase_name in PHASES:
        if phase_name=='test':
            equities={arm:500+sum(results[key][arm]['development']['equity'] for key in r['cohort']) for arm in r['arms']}
            selection={'selected':development_choice(equities),'development_equity':equities,'cash_equity':1000,
                       'plan_sha256':envelope['sha256'],'test_simulated_before_selection':False}
            atomic_json(root/'selection.json',selection)
        for i,key in enumerate(r['cohort']):
            ticks=[Tick(**t) for t in p['series'][key]]
            for name,declared in r['arms'].items():
                state=imported[key] if declared['memory']=='paper_trained' else pristine
                restore_learned(b,state)
                arm={**declared,'view':'original','online':False,'train':False}
                trace=root/trace_name(i,name,phase_name);boundary=root/'boundaries'/f'pool{i}-{name}-{phase_name}'
                outcome=phase(lab,ticks,r[phase_name+'_start'],3,arm,False,deadline,
                              trace_output=trace,activity_output=boundary,news=news)
                if outcome['initial_memory_sha256']!=memory_signature(state) or outcome['final_memory_sha256']!=memory_signature(state):
                    raise ValueError('Frozen paper comparison changed imported memory')
                results[key][name][phase_name]=outcome
                np.savez_compressed(boundary/'final-counts.npz',counts=b.counts)
                if (trace/'view.json').exists():
                    view=json.loads((trace/'view.json').read_text());view['report'].update(
                        evaluation_phase=phase_name,memory_origin=declared['memory'],plan_sha256=envelope['sha256'],
                        training_exposure=r['source_memories'][key] if declared['memory']=='paper_trained' else None)
                    atomic_json(trace/'report.json',view['report']);atomic_json(trace/'view.json',view)
                atomic_json(root/'results.json',results)
                print(f'paper_checkpoint_study phase={phase_name} pool={i} arm={name} equity={outcome["equity"]:.6f}',flush=True)
    summary={'status':'paper_checkpoint_study_completed','plan_sha256':envelope['sha256'],
             'registration':r,'selection':selection,'initial_capital':1000,'active_sleeves':2,'costs':r['costs'],
             'total_equity':{name:{phase_name:500+sum(results[key][name][phase_name]['equity'] for key in r['cohort']) for phase_name in PHASES} for name in r['arms']},
             'graph':{'neurons':b.n,'edges':len(b.post)},'native_build':b.build,
             'initial_dynamics_sha256':digest(root/'initial-dynamics.npz'),'neuron_ids_sha256':digest(root/'neuron-ids.npz'),
             'pristine_memory_sha256':memory_signature(pristine),'pristine_memory_file_sha256':digest(root/'pristine-memory.npz'),
             'code_sha256':{name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES},'news_audit':news_audit,
             'limitation':'Frozen comparison of previously paper-trained memory on two future registered intervals. Simulated DEX execution, not monthly profitability. All activity and phase cash start fresh; no policy is automatically promoted.'}
    atomic_json(root/'summary.json',summary);return summary
