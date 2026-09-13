"""Eta-explicit form of the preserved full-edge rule reconstruction.

Only the registered .001 control and .0001 candidate are accepted. The original
credit auditor remains unchanged for all preceding experiments.
"""
import numpy as np
from .fly_credit_audit import close, split_drive, INTERPRETATION


def reconstruct(arrays, state, circuit, baseline, rule, learning, selection, *, eta):
    """Verify every bin/edge and return a bounded display plus full-edge totals.

    state is the independently verified initial memory or preceding observation's
    reconstructed end state. Rate traces begin at zero for each arm, then carry between its images.
    """
    if eta not in (.001, .0001): raise ValueError('Unregistered learning rate')
    a = arrays; edges = circuit['edges']; pre = circuit['pre']; dan = circuit['dan']
    count = len(edges); n = len(a['neuron_ids'])
    for key, expected in (('plastic_edges', edges), ('plastic_pre', pre), ('plastic_post', circuit['post'])):
        if not np.array_equal(a[key], expected):
            raise ValueError('Recorded circuit identity differs: ' + key)
    counts = a['counts']
    if counts.shape != (50, n) or counts.dtype.kind not in 'iu' or np.any(counts < 0):
        raise ValueError('Invalid whole-network counts')
    for key, width in (('kc', count), ('dan', len(dan)), ('u', count), ('w', count), ('weights', count)):
        if a[key].shape != (50, width) or not np.isfinite(a[key]).all():
            raise ValueError('Invalid full recorded array: ' + key)
    if not np.array_equal(a['initial_weights'], state['weights']):
        raise ValueError('Initial weights differ from preceding state')
    boundary_kc = state['kc'].copy(); boundary_dan = state['dan'].copy()
    values = {k: [] for k in ('earlier', 'current', 'total', 'weight_step')}
    full = {k: [] for k in ('earlier_l1', 'current_l1', 'total_l1', 'opposed_edges')}
    errors = {k: 0. for k in ('kc', 'dan', 'u', 'w')}
    clipped = 0
    for i, row in enumerate(counts):
        K = row[pre] / .01; D = row[dan] / .01  # This assay used the default zero DAN baseline.
        earlier, current, total = split_drive(state['kc'], state['dan'], boundary_kc,
            boundary_dan, K, D, circuit['gain'], i*.01, eta, rule.PARAMETERS)
        if not learning:
            earlier = np.zeros(count); current = np.zeros(count); total = np.zeros(count)
        before = state['weights'].copy()
        rule.advance(state['kc'], state['dan'], state['u'], state['w'], K, D,
                     circuit['gain'], .01, eta, learning=learning, frozen=not learning)
        if learning:
            state['weights'] = (baseline*(1+state['w'])).astype(before.dtype)
        for key in errors:
            errors[key] = max(errors[key], close(state[key], a[key][i], key))
        if not np.array_equal(state['weights'], a['weights'][i]):
            raise ValueError('Recorded float32 weights do not reproduce exactly')
        lo = rule.PARAMETERS['minimum_fraction'] - 1; hi = rule.PARAMETERS['maximum_fraction'] - 1
        clipped += int(np.count_nonzero((state['u'] <= lo) | (state['u'] >= hi)
                                        | (state['w'] <= lo) | (state['w'] >= hi)))
        for key, value in (('earlier', earlier), ('current', current), ('total', total),
                           ('weight_step', state['weights'].astype(float)-before)):
            values[key].append(value[selection].tolist())
        for key, value in (('earlier_l1', earlier), ('current_l1', current), ('total_l1', total)):
            full[key].append(float(np.abs(value).sum()))
        full['opposed_edges'].append(int(np.count_nonzero(earlier*current < -1e-20)))
    values.update(schema=1, eta=eta, plastic_selection=selection.tolist(), times_ms=a['ms'].tolist(),
                  learning=learning, units='u per second', weight_step_units='synaptic weight per bin',
                  boundary='start of this 500 ms image', interpretation=INTERPRETATION)
    summary = {'integrated_absolute_drive': {k.removesuffix('_l1'): .01*sum(v)
                for k, v in full.items() if k.endswith('_l1')},
               'opposed_edge_bins': sum(full['opposed_edges']), 'saturated_edge_bins': clipped,
               'maximum_reconstruction_error': errors, 'all_weights_exact': True,
               'bins': full}
    return values, summary

