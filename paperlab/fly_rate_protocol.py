"""Study 12 design functions. Importing this module registers no market window."""
from dataclasses import asdict
import math

from .fly_paper_protocol import MEMORY_FIELDS, NEWS
from .fly_rate_checkpoint import POLICY as COHORT_POLICY
from .multi import DEX_COSTS

STUDY = '12'
CANDIDATE = 'trained_online_low_eta'
ARMS = {
    'pristine_frozen': {'memory':'pristine','activity_reset':'carry','eta':.001,'learning':False},
    'trained_frozen': {'memory':'paper_trained','activity_reset':'carry','eta':.001,'learning':False},
    'trained_online_carry': {'memory':'paper_trained','activity_reset':'carry','eta':.001,'learning':True},
    CANDIDATE: {'memory':'paper_trained','activity_reset':'carry','eta':.0001,'learning':True},
}
PHASE_STEPS = 24
MIN_OBSERVATIONS = 18
HOSTING_ALLOCATION = 40 * PHASE_STEPS * 300 / (30 * 86400)
HYPOTHESIS = ('Reducing the online learning rate from 0.001 to 0.0001 while retaining both '
    'KC and DAN histories improves fresh cost-aware paper outcomes relative to the original '
    'rate, frozen trained memory and pristine frozen memory. Smaller updates alone are not success.')
INFERENCE = ('Use the pinned current-cohort checkpoint export for all paper_trained arms and '
    'native initial memory for pristine_frozen. Start each pool, arm and phase with fresh '
    'dynamics, its declared weights/u/w, a new $250 account and no pending order. Keep $500 '
    'idle per arm so total starting capital is $1000. Development accounts and learned memory '
    'do not flow into test. Retain both KC and DAN histories between eligible observations. '
    'Keep the full graph, original image adapter, 500 ms observation, fixed decoder and '
    '200 ms dopamine pulse at current 20. Frozen arms receive no learning or reinforcement. '
    'Online arms receive their own equity change after execution of the preceding decision, '
    'zero across unpriced gaps: above $0.01 gives reward, below -$0.01 gives aversive, otherwise '
    'none. Learning stays enabled without an injected pulse. Only the candidate uses eta '
    '0.0001; the other arms use 0.001. No extra current or partial state reset is applied. '
    'Missing, repeated and terminal quotes do not advance the brain. Sealing, reconstruction '
    'and simulation run in fresh Python processes with PYTHONHASHSEED=0 before startup.')
SELECTION = ('The sole candidate is trained_online_low_eta. Require at least 18 of 24 eligible '
    'observations in each development pool and identical coverage across all four arms. '
    'Select only when candidate aggregate development equity strictly exceeds all three '
    'controls and $1000 plus $40/month hosting prorated to two hours on a 30-day month, '
    'by more than 1e-9. Equity includes the declared execution costs. Persist selection after '
    'all eight development chunks and before any test chunk. Run and report all eight test '
    'chunks regardless of selection, including turnover, fees, missing marks and coverage. '
    'Insufficient test coverage is inconclusive. Never reselect from test or automatically '
    'promote a policy. Two-hour results are not monthly return estimates.')


def chunk_order():
    return [(stage,pool,arm) for stage in ('development','test') for pool in range(2) for arm in ARMS]


def chunk_name(stage,pool,arm):
    if type(pool) is not int or (stage,pool,arm) not in chunk_order():
        raise ValueError('Unknown rate comparison condition')
    return f'{stage}-pool{pool}-{arm}'


def development_choice(equities,observations):
    if set(equities)!=set(ARMS) or any(type(v) not in (int,float) or not math.isfinite(v) for v in equities.values()):
        raise ValueError('Finite development equity is required for every arm')
    if set(observations)!=set(ARMS) or any(len(v)!=2 or any(type(n) is not int or not 0<=n<=PHASE_STEPS for n in v) for v in observations.values()):
        raise ValueError('Expected per-pool development observation counts')
    coverage=observations['pristine_frozen']
    if any(v!=coverage for v in observations.values()):raise ValueError('Different arm coverage')
    if min(coverage)<MIN_OBSERVATIONS:return None
    threshold=max(1000+HOSTING_ALLOCATION,*(equities[k] for k in ARMS if k!=CANDIDATE))
    return CANDIDATE if equities[CANDIDATE]>threshold+1e-9 else None


def validate_registration(r,audit):
    """Validate a later timestamped registration against its exported memories."""
    fixed={'schema':4,'kind':'paper_checkpoint_comparison','study':STUDY,'arms':ARMS,
           'phase_steps':PHASE_STEPS,'decision_seconds':300,'costs':asdict(DEX_COSTS),
           'hypothesis':HYPOTHESIS,'inference_protocol':INFERENCE,'selection_rule':SELECTION,
           'news_protocol':NEWS,'cohort_policy':COHORT_POLICY,'python_hash_seed':'0'}
    if any(r.get(k)!=v for k,v in fixed.items()):raise ValueError('Registered rate comparison design differs')
    for key in ('parent_report_sha256','mechanism_audit_sha256','training_audit_sha256',
                'capture_result_sha256','cohort_selection_sha256'):
        value=r.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing pinned rate comparison provenance')
    clocks=('recorded_at','development_start','test_start','end','training_cutoff','checkpoint_capture_at','parent_end')
    if any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) for k in clocks):
        raise ValueError('Invalid registration clock')
    start=r['development_start']
    if (start%300 or r['test_start']!=start+7200 or r['end']!=start+14400
            or not 0<r['training_cutoff']<r['checkpoint_capture_at']<=r['recorded_at']<start
            or not 0<r['parent_end']<start):
        raise ValueError('Register and capture before independent future phases')
    cohort=r.get('cohort',[])
    if (len(cohort)!=2 or len(set(cohort))!=2 or any(not k.startswith('solana:') for k in cohort)
            or set(cohort)!=set(audit['pools']) or set(r.get('source_memories',{}))!=set(cohort)
            or r['checkpoint_capture_at']!=audit['captured_at']):
        raise ValueError('Registered cohort or checkpoint capture differs')
    for key in cohort:
        source=audit['pools'][key]
        if (r['source_memories'][key]!={k:source[k] for k in MEMORY_FIELDS}
                or source['positive_rewards']<1 or source['negative_rewards']<1 or source['observations']<2):
            raise ValueError('Pinned trained memory or reward exposure differs')
    if r['training_cutoff']!=max((p['last_slot']+1)*300 for p in audit['pools'].values()):
        raise ValueError('Training cutoff differs from exported ledger')
    return r
