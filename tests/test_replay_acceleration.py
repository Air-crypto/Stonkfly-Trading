from pathlib import Path
import copy
import functools
import hashlib
import importlib.util
import json
import numpy as np
import pytest
from replay_readout import ReplayReadout
from paperlab.solana_paper import Readout,FEATURES
from paperlab.core import atomic_json,digest


def transition(i=1):return dict(x=np.full(FEATURES,i/100,dtype=np.float32),action=i%2)


@pytest.mark.parametrize('ratio',[1,4])
def test_exact_optimizer_ratio_and_reward_is_not_multiplied(ratio):
    h=ReplayReadout(ratio=ratio,batch_size=4,capacity=8,target_every=5)
    metrics=[h.update(transition(i),transition(i+1)['x'],float(i),5,terminal=i==4) for i in range(5)]
    assert h.updates==5 and h.optimizer_steps==5*ratio
    assert sum(m['reward_usd'] for m in metrics)==10
    assert all(m['gradient_steps']==ratio and len(m['optimizer_metrics'])==ratio for m in metrics)
    assert h.target_syncs==ratio
    assert not any(p.requires_grad for p in h.target.parameters())
    assert metrics[-1]['discount']==0 and metrics[-1]['target']==pytest.approx(4/25)


def test_target_network_stays_fixed_until_sync_and_masks_actions():
    h=ReplayReadout(target_every=100)
    import torch
    with torch.no_grad():
        for p in h.target.parameters():p.zero_()
        h.target[-1].bias.copy_(torch.tensor([2.,999.]))
    t=h.transition(transition(),transition()['x'],0,5,(0,),False)
    assert float(h.targets([t])[0])==pytest.approx(1.9)
    before=[p.clone() for p in h.target.parameters()]
    h.update(transition(),transition()['x'],1,5)
    assert all(torch.equal(a,b) for a,b in zip(before,h.target.parameters()))


def test_resume_preserves_replay_rng_target_optimizer_and_exploration(tmp_path):
    h=ReplayReadout();h.update(transition(),transition()['x'],2,5);h.save(tmp_path/'h.pt')
    restored=ReplayReadout();restored.restore(tmp_path/'h.pt')
    assert h.choose(transition()['x'])==restored.choose(transition()['x'])
    a=h.update(transition(2),transition(3)['x'],-5,7,terminal=True)
    b=restored.update(transition(2),transition(3)['x'],-5,7,terminal=True)
    assert a==b
    np.testing.assert_array_equal(h.values(transition()['x']),restored.values(transition()['x']))
    with pytest.raises(ValueError,match='configuration'):ReplayReadout(ratio=1).restore(tmp_path/'h.pt')


def test_old_checkpoint_initializes_target_and_is_not_modified(tmp_path):
    baseline=Readout();baseline.update(transition(),transition()['x'],1,5);baseline.save(tmp_path/'old.pt')
    before=digest(tmp_path/'old.pt');h=ReplayReadout();h.restore(tmp_path/'old.pt')
    assert h.optimizer_steps==h.updates==1 and not h.buffer
    np.testing.assert_array_equal(h.values(transition()['x']),baseline.values(transition()['x']))
    h.update(transition(),transition()['x'],2,5)
    assert digest(tmp_path/'old.pt')==before


def test_buffer_bounded_and_development_loss_does_not_train():
    import torch
    h=ReplayReadout(capacity=4,batch_size=4)
    for i in range(7):h.update(transition(i),transition(i+1)['x'],1,5)
    assert len(h.buffer)==4
    before=[p.clone() for p in h.model.parameters()];steps=h.optimizer_steps
    assert np.isfinite(h.td_loss(list(h.buffer)))
    assert h.optimizer_steps==steps and all(torch.equal(a,b) for a,b in zip(before,h.model.parameters()))


def test_replay_rejects_invalid_inputs_before_buffer_or_optimizer_changes():
    h=ReplayReadout();bad=np.zeros(FEATURES);bad[0]=np.nan
    with pytest.raises(ValueError):h.update(transition(),bad,0,5)
    with pytest.raises(ValueError):h.update(transition(),transition()['x'],0,-1)
    with pytest.raises(ValueError):h.update(transition(),transition()['x'],0,5,allowed_actions=())
    assert h.updates==h.optimizer_steps==0 and not h.buffer


def test_faster_loop_reuses_real_account_audit_and_extracts_exact_rewards(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('baseline_test_fixture',Path(__file__).with_name('test_solana_online.py'))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from accelerated_online import run
    from acceleration_study import extract_transitions
    monkeypatch.setattr(fixture,'run',functools.partial(run,interval_seconds=2.5,head_factory=ReplayReadout))
    fixture.test_cloud_entry_logic_rotates_launches_and_restores_account(tmp_path,monkeypatch,True,False,True,True)
    result=json.loads((tmp_path/'run/completed.json').read_text())
    rows=[json.loads(s) for s in (tmp_path/'run/decisions.jsonl').read_text().splitlines()]
    opening=json.loads((tmp_path/'run/opening.json').read_text())
    t=extract_transitions(opening,rows)
    assert result['new_optimizer_steps']==4*len(t)==4*result['new_readout_updates']
    assert sum(x['reward_usd'] for x in t)==pytest.approx(result['equity_stress_usd']-1000)
    assert result['neural_observations']>80


def test_data_selection_excludes_all_protected_tapes_and_post_cutoff(tmp_path):
    from acceleration_study import select_data
    for i in range(7):
        root=tmp_path/'solana-live'/f'solana-online-{i}'
        atomic_json(root/'completed.json',dict(status='completed',all_observed=True,started=i*10,ended=i*10+5,
                                              account_audit=dict(reward_reconciliation_verified=True)))
        for name in ['opening.json','decisions.jsonl']:(root/name).write_text('{}')
    atomic_json(tmp_path/'checkpoint-eval/evaluation-old/plan.json',dict(tape=dict(id='solana-online-1')))
    atomic_json(tmp_path/'checkpoint-eval/evaluation-active/plan.json',dict(tape=None,cutoff=55))
    atomic_json(tmp_path/'checkpoint-eval/control.json',dict(batch='evaluation-active'))
    selection=select_data(tmp_path,100)
    assert [e['id'] for e in selection['training']]==['solana-online-0','solana-online-2','solana-online-3']
    assert selection['development']['id']=='solana-online-4'
    assert all(e['ended']<selection['cutoff'] for e in selection['training'])


def test_baseline_source_is_pinned_and_sealed_fingerprint_unchanged():
    from paperlab.checkpoint_eval import source_fingerprint
    source=Path('paperlab/solana_online.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() in Path('accelerated_online.py').read_text().splitlines()[2]
    sealed=json.loads(Path('reports/all-pump-evaluation-plan-20260914.json').read_text())
    assert source_fingerprint('.')==sealed['source_hashes']


def test_synthetic_native_benchmark_feed_accepts_runner_clock(tmp_path):
    from acceleration_study import SyntheticFeed
    feed=SyntheticFeed(tmp_path)
    snapshots,now,health=feed.snapshot_at(lambda:feed.base+7.5)
    assert now==feed.base+7.5 and health['synthetic']
    assert health['last_message']==now
    assert len(snapshots)==24
    assert all(s['trades'][0]['received']==now for s in snapshots.values())


def test_actual_synthetic_feed_runs_entire_accelerated_entrypoint(tmp_path):
    spec=importlib.util.spec_from_file_location('native_benchmark_fixture',Path(__file__).with_name('test_solana_online.py'))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from acceleration_study import SyntheticFeed
    from accelerated_online import run
    from paperlab.solana_online_audit import audit
    clock=fixture.Clock();parent=fixture.make_parent(tmp_path)
    saved=json.loads((parent/'online-state.json').read_text())
    (parent/'native.npz').write_bytes(b'fake immutable native weights')
    saved['native_checkpoint']=str(parent/'native.npz');atomic_json(parent/'online-state.json',saved)
    result=run(tmp_path/'run','unused',parent,seconds=180,fly_factory=fixture.FakeFly,
        feed_factory=lambda root:SyntheticFeed(root,clock),clock=clock,sleep=clock.sleep,
        fx_fetch=lambda:100.,quote_fx_fetch=lambda:1.,account_mode='fresh_training_episode',
        all_observed=True,interval_seconds=2.5,head_factory=ReplayReadout)
    rows=[json.loads(s) for s in (tmp_path/'run/decisions.jsonl').read_text().splitlines()]
    verified=audit(json.loads((tmp_path/'run/opening.json').read_text()),rows)
    assert result['status']=='completed' and result['neural_observations']>20
    assert result['new_optimizer_steps']==4*verified['new_head_updates']>0
    assert verified['reward_reconciliation_verified']


def test_native_benchmark_refuses_local_fly_execution():
    from acceleration_study import native_benchmark
    with pytest.raises(RuntimeError,match='must run in Modal'):
        native_benchmark(None,None,None,None)
