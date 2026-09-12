import copy

import pytest

from paperlab.fly_paper_figure import CHECKS, evidence
from paperlab.fly_paper_protocol import ARMS, development_choice


def report_fixture():
    # Explicit display fixture. It is not published as market or neural evidence.
    r = {'kind': 'paper_checkpoint_comparison', 'arms': ARMS, 'cohort': ['one', 'two'],
         'costs': {'capital': 250}, 'development_start': 900, 'test_start': 1800,
         'phase_steps': 3, 'decision_seconds': 300}
    totals = {a: {'development': 1000 + i, 'test': 1005 - i} for i, a in enumerate(ARMS)}
    diagnostics = {}
    for key in r['cohort']:
        diagnostics[key] = {}
        for arm in ARMS:
            diagnostics[key][arm] = {}
            for phase in ('development', 'test'):
                diagnostics[key][arm][phase] = {
                    'fills': 0, 'fees': 0, 'decisions': [
                        {'decision_ts': r[phase + '_start'] + 300 * i,
                         'equity': 250 if i < 3 else (totals[arm][phase] - 500) / 2,
                         'available': i != 1, 'terminal': i == 3,
                         'neural': {} if i in (0, 2) else None,
                         'fill': {'status': 'hold'}} for i in range(4)]}
    development = {a: totals[a]['development'] for a in ARMS}
    return {'status': 'paper_checkpoint_study_completed', 'registration': r,
            'initial_capital': 1000, 'costs': r['costs'], 'plan_sha256': 'a' * 64,
            'total_equity': totals, 'phase_diagnostics': diagnostics,
            'verification': {k: True for k in CHECKS},
            'selection': {'selected': development_choice(development),
                          'development_equity': development,
                          'test_simulated_before_selection': False,
                          'plan_sha256': 'a' * 64}}


def test_matched_effects_use_same_reset_scope_and_keep_test_out_of_selection():
    report = report_fixture();panels, effects = evidence(report)
    assert effects == {'development': {'carry': 1, 'reset': 1}, 'test': {'carry': -1, 'reset': -1}}
    assert report['selection']['selected'] == 'trained_input_reset'
    assert panels['trained_frozen', 'test']['observed'] == 4
    assert panels['trained_frozen', 'test']['unavailable'] == [False, True, False, False]
    # A test winner must not replace the pre-test choice.
    report['selection']['selected'] = 'trained_frozen'
    with pytest.raises(ValueError, match='selection'):evidence(report)


@pytest.mark.parametrize('field', ['audit', 'equity', 'fees', 'time', 'phase', 'nan', 'summary_nan', 'fee_nan'])
def test_plot_rejects_missing_audit_or_misrepresented_ledger(field):
    report = copy.deepcopy(report_fixture())
    lane = report['phase_diagnostics']['one']['trained_frozen']['test']
    if field == 'audit':report['verification']['all_full_count_decoders_verified'] = False
    if field == 'equity':lane['decisions'][-1]['equity'] += 1
    if field == 'fees':lane['fees'] = 1
    if field == 'time':lane['decisions'][1]['decision_ts'] += 1
    if field == 'phase':report['total_equity']['trained_frozen']['training'] = 1000
    if field == 'nan':lane['decisions'][1]['equity'] = float('nan')
    if field == 'summary_nan':report['total_equity']['trained_frozen']['test'] = float('nan')
    if field == 'fee_nan':lane['fees'] = float('nan')
    with pytest.raises(ValueError):evidence(report)
