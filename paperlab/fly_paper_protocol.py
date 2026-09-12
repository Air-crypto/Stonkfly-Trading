"""Fixed prospective protocol for comparing actual paper-trained memory."""
from dataclasses import asdict
import math

from .multi import DEX_COSTS

ARMS={
    'pristine_frozen':{'memory':'pristine','activity_reset':'carry','eta':.001},
    'trained_frozen':{'memory':'paper_trained','activity_reset':'carry','eta':.001},
    'pristine_input_reset':{'memory':'pristine','activity_reset':'conductance','eta':.001},
    'trained_input_reset':{'memory':'paper_trained','activity_reset':'conductance','eta':.001},
}
NEWS='Use the unchanged market-frame adapter and 50-dimensional news features reconstructed at each selected quote timestamp. Include only revisions whose published, seen and encoded clocks are no later than that timestamp. Seal the feature vectors and news snapshot hash before model evaluation. All four arms receive identical images; never fetch headlines during simulation.'
INFERENCE='Import only the pinned plastic weights/u/w for paper_trained arms; pristine arms use native initial memory. Each development and test phase starts separately with fresh dynamics and the declared memory. All inference is frozen, with no reinforcement or extra current. Carry arms preserve activity between eligible observations. Conductance arms restore only native synaptic-input state g immediately before every eligible observation after the first. Missing slots do not advance or reset the brain. Preserve all other fields and the neural clock at each partial reset.'
SELECTION='Choose the higher development equity of trained_frozen and trained_input_reset; ties within 1e-9 prefer trained_frozen. Select it only if it strictly exceeds cash and both pristine controls by more than 1e-9. Persist selection before test simulation. Report every test arm without reselection or automatic policy deployment.'
MEMORY_FIELDS=('memory_sha256','memory_file_sha256','checkpoint_sha256','last_slot','observations','positive_rewards','negative_rewards')


def validate_registration(registration,audit):
    r=registration
    activation=r.get('study')=='10'
    online=r.get('study')=='11'
    if r.get('schema')!=(3 if online else 2 if activation else 1) or r.get('kind')!='paper_checkpoint_comparison' or r.get('study') not in ('09','10','11'):
        raise ValueError('Unknown paper checkpoint comparison')
    from .fly_activation_protocol import ARMS as activated, INFERENCE as activation_inference, SELECTION as activation_selection, RATIONALE
    arms,inference,selection=(activated,activation_inference,activation_selection) if activation else (ARMS,INFERENCE,SELECTION)
    steps=3
    if online:
        from .fly_online_protocol import ARMS as arms, INFERENCE as inference, SELECTION as selection, RATIONALE, PHASE_STEPS
        steps=PHASE_STEPS
    if (activation or online) and (r.get('rationale')!=RATIONALE or any(not isinstance(r.get(k),str) or len(r[k])!=64 or any(c not in '0123456789abcdef' for c in r[k]) for k in ('mechanism_report_sha256','mechanism_audit_sha256'))):
        raise ValueError('Missing pinned activation mechanism evidence')
    if r.get('arms')!=arms or r.get('inference_protocol')!=inference or r.get('news_protocol')!=NEWS or r.get('selection_rule')!=selection:
        raise ValueError('Registered inference, news or selection protocol differs')
    if r.get('costs')!=asdict(DEX_COSTS) or r.get('decision_seconds')!=300 or type(r.get('phase_steps')) is not int or r['phase_steps']!=steps:
        raise ValueError('Registered costs or cadence differ')
    start=r['development_start']
    if not all(isinstance(r.get(k),(float,int)) and math.isfinite(r[k]) for k in ('recorded_at','development_start','test_start','end','training_cutoff','checkpoint_capture_at')):
        raise ValueError('Invalid registered timestamps')
    if start%300 or r['test_start']!=start+steps*300 or r['end']!=start+2*steps*300 or not 0<r['training_cutoff']<r['checkpoint_capture_at']<=r['recorded_at']<start:
        raise ValueError('Capture and register before the separate future evaluation phases')
    if len(r['cohort'])!=2 or len(set(r['cohort']))!=2 or set(r['cohort'])!=set(audit['pools']) or set(r['source_memories'])!=set(r['cohort']):
        raise ValueError('Checkpoint cohort differs')
    if r['checkpoint_capture_at']!=audit['captured_at']:
        raise ValueError('Checkpoint capture time differs')
    for key in ('parent_plan_sha256','parent_report_sha256','training_audit_sha256'):
        value=r.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing pinned protocol provenance')
    for pool in r['cohort']:
        source=audit['pools'][pool]
        if r['source_memories'][pool]!={k:source[k] for k in MEMORY_FIELDS}:
            raise ValueError('Pinned training memory or exposure differs')
        if source['positive_rewards']<1 or source['negative_rewards']<1 or source['observations']<2:
            raise ValueError('Require observed positive and negative trading feedback')
    if r['training_cutoff']!=max((p['last_slot']+1)*300 for p in audit['pools'].values()):
        raise ValueError('Training cutoff differs from the captured ledger')
    return r


def development_choice(equities):
    from .fly_activation_protocol import ARMS as activated
    activation=set(equities)==set(activated)
    if set(equities) not in (set(ARMS),set(activated)) or any(type(v) not in (int,float) or not math.isfinite(v) for v in equities.values()):
        raise ValueError('Expected finite development equity for all four arms')
    alternative='trained_stimulated' if activation else 'trained_input_reset'
    pristine='pristine_stimulated' if activation else 'pristine_input_reset'
    candidate=alternative if equities[alternative]>equities['trained_frozen']+1e-9 else 'trained_frozen'
    return candidate if equities[candidate]>max(1000,equities['pristine_frozen'],equities[pristine])+1e-9 else None
