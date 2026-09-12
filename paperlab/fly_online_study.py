"""One complete registered pool/arm/phase, with every native bin retained."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .core import Tick, atomic_json, digest
from .fly_market_activity import dynamic_state
from .fly_market_study import learned_state, memory_signature, phase, restore_learned, signature
from .fly_online_protocol import ARMS, HOSTING_ALLOCATION, chunk_name, chunk_order, development_choice
from .fly_paper_inputs import SealedNews, audit_news, validate
from .fly_paper_study import SOURCE_FILES as PAPER_SOURCES, imported_memory
from .fly_trace import TraceLab

SOURCE_FILES = tuple(dict.fromkeys((*PAPER_SOURCES, 'fly_online_protocol.py', 'fly_online_study.py',
    'fly_online_audit.py', 'fly_credit_audit.py', 'fly_credit_reset_audit.py', 'fly_trace_memory.py',
    'fly_paper_price_audit.py')))


def select_development(envelope, chunks):
    p = validate(envelope)
    if p['registration']['study'] != '11':
        raise ValueError('Expected study 11 inputs')
    expected = {chunk_name(*c) for c in chunk_order() if c[0] == 'development'}
    if set(chunks) != expected:
        raise ValueError('Complete all eight development chunks before selection')
    equities = {arm: 500. for arm in ARMS}; counts = {arm: [] for arm in ARMS}
    provenance = {}
    for stage, pool, arm in chunk_order()[:8]:
        name = chunk_name(stage, pool, arm); result = chunks[name]
        if (result.get('status') != 'paper_online_chunk_completed' or result.get('chunk') != name
                or result.get('plan_sha256') != envelope['sha256'] or result.get('registration') != p['registration']
                or result.get('stage') != stage or result.get('pool_index') != pool or result.get('arm') != arm):
            raise ValueError('Development receipt identity differs')
        rows = result['outcome']['rows']
        if len(rows) != p['registration']['phase_steps'] + 1:
            raise ValueError('Incomplete development timeline')
        equities[arm] += result['outcome']['equity']
        counts[arm].append(sum(row['event'] is not None for row in rows[:-1]))
        provenance[name] = signature(result)
    return {'plan_sha256': envelope['sha256'], 'selected': development_choice(equities, counts),
            'development_equity': equities, 'development_observations': counts,
            'cash_equity': 1000, 'hosting_allocation_usd': HOSTING_ALLOCATION,
            'development_chunk_sha256': provenance, 'test_simulated_before_selection': False}


def run_chunk(envelope, memory_root, news_archive, data, output, *, stage, pool_index, arm,
              development=None, seconds=480):
    p = validate(envelope); r = p['registration']; name = chunk_name(stage, pool_index, arm)
    if r['study'] != '11':raise ValueError('Online chunk runner requires study 11')
    if type(seconds) not in (int, float) or not 1 <= seconds <= 480:raise ValueError('Bound each chunk to at most 480 seconds')
    if stage == 'test':
        selection = select_development(envelope, development or {})
    elif development is not None:
        raise ValueError('Development cannot import test or selection state')
    else:selection = None
    root = Path(output)
    if root.exists():raise ValueError('Refuse to overwrite or resume a neural chunk')
    news_check = audit_news(envelope, news_archive)
    root.mkdir(parents=True); atomic_json(root/'protocol.json', envelope)
    if selection is not None:
        atomic_json(root/'development.json', development); atomic_json(root/'selection.json', selection)
    shutil.copyfile(news_archive, root/'news.db')
    started = time.monotonic(); deadline = started + seconds
    lab = TraceLab(data); b = lab.brain
    imported = imported_memory(b, p, memory_root); pristine = learned_state(b)
    np.savez_compressed(root/'pristine-memory.npz', **pristine)
    np.savez_compressed(root/'initial-dynamics.npz', **dynamic_state(b))
    np.savez_compressed(root/'neuron-ids.npz', neuron_ids=b.ids)
    np.savez_compressed(root/'full-weight-reference.npz', weight=b.weight)
    circuit = b.circuit
    np.savez_compressed(root/'circuit.npz', **{k: circuit[k] for k in ('edges', 'pre', 'dan', 'gain')},
                        post=b.post[circuit['edges']], baseline=b.baseline_plastic)
    (root/'imported').mkdir(); shutil.copyfile(Path(memory_root)/'audit.json', root/'imported/audit.json')
    for i in range(2):shutil.copyfile(Path(memory_root)/f'pool{i}-memory.npz', root/f'imported/pool{i}-memory.npz')
    key = r['cohort'][pool_index]; declared = ARMS[arm]
    state = imported[key] if declared['memory'] == 'paper_trained' else pristine
    restore_learned(b, state)
    outcome = phase(lab, [Tick(**t) for t in p['series'][key]], r[stage+'_start'], r['phase_steps'],
                    {**declared, 'view': 'original'}, declared['learning'], deadline,
                    trace_output=root/'trace', activity_output=root/'boundaries',
                    news=SealedNews(p['news_features']), record_end_state=True)
    if outcome['initial_memory_sha256'] != memory_signature(state):raise ValueError('Chunk imported different memory')
    np.savez_compressed(root/'final-memory.npz', **learned_state(b))
    np.savez_compressed(root/'final-dynamics.npz', **dynamic_state(b))
    if (root/'trace/view.json').exists():
        view = json.loads((root/'trace/view.json').read_text())
        view['report'].update(evaluation_phase=stage, memory_origin=declared['memory'],
            plan_sha256=envelope['sha256'], study='11', pool=key, chunk=name,
            training_exposure=r['source_memories'][key] if declared['memory']=='paper_trained' else None)
        atomic_json(root/'trace/report.json', view['report']); atomic_json(root/'trace/view.json', view)
    result = {'status': 'paper_online_chunk_completed', 'chunk': name, 'stage': stage,
        'pool_index': pool_index, 'pool': key, 'arm': arm, 'plan_sha256': envelope['sha256'],
        'registration': r, 'selection': selection, 'outcome': outcome, 'native_build': b.build,
        'graph': {'neurons': b.n, 'edges': len(b.post), 'plastic_edges': len(circuit['edges'])},
        'news_audit': news_check, 'seconds': time.monotonic()-started,
        'code_sha256': {file: digest(Path(__file__).with_name(file)) for file in SOURCE_FILES},
        'artifact_sha256': {str(file.relative_to(root)): digest(file) for file in sorted(root.rglob('*')) if file.is_file()},
        'interpretation': 'One isolated paper phase. No account or memory flows from development into test, and no live policy is promoted.'}
    atomic_json(root/'summary.json', result)
    print(json.dumps({'event': 'paper_online_chunk_completed', 'chunk': name,
        'equity': outcome['equity'], 'observations': sum(row['event'] is not None for row in outcome['rows']),
        'seconds': result['seconds']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'memory', 'news', 'fly-data', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--stage', choices=('development', 'test'), required=True)
    parser.add_argument('--pool', type=int, choices=(0, 1), required=True)
    parser.add_argument('--arm', choices=tuple(ARMS), required=True)
    parser.add_argument('--development', type=Path, help='All eight completed development summaries, keyed by chunk ID')
    args = parser.parse_args()
    run_chunk(json.loads(args.plan.read_text()), args.memory, args.news, args.fly_data, args.out,
        stage=args.stage, pool_index=args.pool, arm=args.arm,
        development=json.loads(args.development.read_text()) if args.development else None)


if __name__ == '__main__':main()
