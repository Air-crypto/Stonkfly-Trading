"""Fixed study 11 design; creating this module does not register a market window."""
import math

ARMS = {
    'pristine_frozen': {'memory': 'pristine', 'activity_reset': 'carry', 'eta': .001, 'learning': False},
    'trained_frozen': {'memory': 'paper_trained', 'activity_reset': 'carry', 'eta': .001, 'learning': False},
    'trained_online_carry': {'memory': 'paper_trained', 'activity_reset': 'carry', 'eta': .001, 'learning': True},
    'trained_online_reset_rates': {'memory': 'paper_trained', 'activity_reset': 'reset_rates', 'eta': .001, 'learning': True},
}
PHASE_STEPS = 24
MIN_OBSERVATIONS = 18
HOSTING_ALLOCATION = 40 * (PHASE_STEPS * 300) / (30 * 86400)
INFERENCE = ('Import the same pinned plastic weights/u/w for every paper_trained arm; pristine uses native initial memory. '
    'Start each pool, arm and phase with fresh dynamics and the declared memory, a fresh $250 account, and no pending order. '
    'Do not carry development memory or accounts into test. Frozen arms receive no reinforcement or learning. '
    'Online arms use their own equity change after execution of the preceding decision as feedback, zero across unpriced gaps; '
    'map changes above $0.01 to reward, below -$0.01 to aversive, otherwise none. Keep learning enabled even without a pulse. '
    'Use the unchanged 500 ms observation and 200 ms dopamine pulse at current 20. '
    'The reset_rates arm clears only rate_kc and rate_dan immediately before each eligible observation after the first. '
    'Carry all other dynamic state, weights/u/w and the neural clock across that boundary. Missing or repeated quotes do not '
    'advance or reset the brain. Keep the full graph, original image adapter, eta 0.001 and fixed decoder unchanged.')
SELECTION = ('The sole candidate is trained_online_reset_rates. Require at least 18 of 24 eligible observations per pool '
    'in development and identical coverage across arms. Select the candidate only if its aggregate development equity '
    'strictly exceeds all three controls and $1000 plus $40/month hosting prorated to two hours on a 30-day month, '
    'by more than 1e-9. Persist selection after all eight development chunks and before any test chunk. '
    'Evaluate every test arm regardless of selection. Report test coverage and costs; never reselect or automatically deploy a policy.')
RATIONALE = ('The audited trace-reset assay reproduced three carry controls and all first observations. Clearing only '
    'KC/DAN rate traces changed subsequent online updates and one recorded action, while frozen counts stayed identical. '
    'Test that single intervention over fresh two-hour development and two-hour test intervals; no claim of improved returns '
    'follows from the earlier three-image mechanism assay.')


def chunk_order():
    """Each chunk is a complete phase for one pool/arm, never a neural resume."""
    return [(stage, pool, arm) for stage in ('development', 'test') for pool in range(2) for arm in ARMS]


def chunk_name(stage, pool, arm):
    if type(pool) is not int or (stage, pool, arm) not in chunk_order():
        raise ValueError('Unknown online comparison chunk')
    return f'{stage}-pool{pool}-{arm}'


def development_choice(equities, observations):
    if set(equities) != set(ARMS) or any(type(v) not in (int, float) or not math.isfinite(v) for v in equities.values()):
        raise ValueError('Require finite development equity for every registered arm')
    if set(observations) != set(ARMS) or any(len(v) != 2 or any(type(n) is not int or not 0 <= n <= PHASE_STEPS for n in v) for v in observations.values()):
        raise ValueError('Require per-pool development observation counts')
    coverage = observations['pristine_frozen']
    if any(v != coverage for v in observations.values()):
        raise ValueError('Arms observed different development coverage')
    candidate = 'trained_online_reset_rates'
    if min(coverage) < MIN_OBSERVATIONS:
        return None
    threshold = max(1000 + HOSTING_ALLOCATION, *(equities[k] for k in ARMS if k != candidate))
    return candidate if equities[candidate] > threshold + 1e-9 else None
