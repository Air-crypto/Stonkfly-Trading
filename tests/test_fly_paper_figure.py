import copy

import pytest

from paperlab.fly_paper_figure import CHECKS, evidence
from paperlab.fly_paper_protocol import ARMS, development_choice


def report_fixture(study='09'):
    # Explicit display fixture. It is not published as market or neural evidence.
    from paperlab.fly_activation_protocol import ARMS as activated
    arms = copy.deepcopy(activated if study == '10' else ARMS)
    r = {'study': study, 'kind': 'paper_checkpoint_comparison', 'arms': arms, 'cohort': ['one', 'two'],
         'costs': {'capital': 250}, 'development_start': 900, 'test_start': 1800,
         'phase_steps': 3, 'decision_seconds': 300,
         'source_memories': {k: {'observations': 10, 'positive_rewards': 4, 'negative_rewards': 4}
                             for k in ('one', 'two')}}
    totals = {a: {'development': 1000 + i, 'test': 1005 - i} for i, a in enumerate(arms)}
    diagnostics = {}
    for key in r['cohort']:
        diagnostics[key] = {}
        for arm in arms:
            diagnostics[key][arm] = {}
            for phase in ('development', 'test'):
                diagnostics[key][arm][phase] = {
                    'fills': 0, 'fees': 0, 'decisions': [
                        {'decision_ts': r[phase + '_start'] + 300 * i,
                         'equity': 250 if i < 3 else (totals[arm][phase] - 500) / 2,
                         'available': i != 1, 'terminal': i == 3,
                         'neural': ({'stimulation': {'current': arms[arm]['recipient_current'],
                                     'target_ids': ['10704', '11402'], 'duration_ms': 500,
                                     'artifact_sha256': 'd'*64}} if study == '10' else {}) if i in (0, 2) else None,
                         'fill': {'status': 'hold'}} for i in range(4)]}
    development = {a: totals[a]['development'] for a in arms}
    checks = CHECKS + (('prospective_cloud_registration_verified', 'native_recipient_current_verified') if study == '10' else ())
    return {'status': 'paper_checkpoint_study_completed', 'registration': r,
            'initial_capital': 1000, 'costs': r['costs'], 'plan_sha256': 'a' * 64,
            'total_equity': totals, 'phase_diagnostics': diagnostics,
            'verification': {k: True for k in checks},
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


def test_activation_figure_compares_memory_at_matched_current():
    panels, effects = evidence(report_fixture('10'))
    assert effects == {'development': {'current_0': 1, 'current_10': 1},
                       'test': {'current_0': -1, 'current_10': -1}}
    assert panels['trained_stimulated', 'test']['observed'] == 4


@pytest.mark.parametrize('field', ['prospective_cloud_registration_verified', 'native_recipient_current_verified',
                                   'current', 'target', 'duration', 'artifact', 'condition'])
def test_activation_figure_refuses_missing_evidence_or_mislabeled_intervention(field):
    report = report_fixture('10')
    stimulus = report['phase_diagnostics']['one']['trained_stimulated']['test']['decisions'][0]['neural']['stimulation']
    if field.endswith('_verified'):report['verification'][field] = False
    if field == 'current':stimulus['current'] = 0
    if field == 'target':stimulus['target_ids'] = ['10704']
    if field == 'duration':stimulus['duration_ms'] = 200
    if field == 'artifact':stimulus['artifact_sha256'] = ''
    if field == 'condition':report['registration']['arms']['trained_stimulated']['recipient_current'] = 5
    with pytest.raises(ValueError):evidence(report)
