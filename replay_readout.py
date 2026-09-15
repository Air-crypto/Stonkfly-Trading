"""Experimental DQN replay head; never imported by the sealed baseline worker."""
from collections import deque
import copy
import math
import numpy as np
from paperlab.solana_paper import Readout, FEATURES

PROTOCOL='dqn_replay_target_v1'


class ReplayReadout(Readout):
    def __init__(self,seed=13,*,ratio=4,capacity=10000,batch_size=32,target_every=100):
        super().__init__(seed)
        if ratio not in (1,4) or not 1<=batch_size<=capacity<=100000 or target_every<1:
            raise ValueError('Invalid replay configuration')
        self.ratio=ratio;self.capacity=capacity;self.batch_size=batch_size;self.target_every=target_every
        self.target=copy.deepcopy(self.model).eval()
        for p in self.target.parameters():p.requires_grad_(False)
        self.buffer=deque(maxlen=capacity);self.replay_rng=np.random.default_rng(seed+1000)
        self.optimizer_steps=0;self.target_syncs=0

    def transition(self,previous,x,reward,elapsed,allowed_actions,terminal):
        a=np.asarray(previous['x'],dtype=np.float32);b=np.asarray(x,dtype=np.float32)
        if a.shape!=(FEATURES,) or b.shape!=(FEATURES,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid replay features')
        if previous['action'] not in (0,1) or not allowed_actions or any(v not in (0,1) for v in allowed_actions):
            raise ValueError('Invalid action mask')
        if not math.isfinite(reward) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid reward timing')
        return dict(x=a.tolist(),action=int(previous['action']),next_x=b.tolist(),reward_usd=float(reward),
                    reward_scaled=float(np.clip(reward/25,-1,1)),discount=0. if terminal else .95**(elapsed/5),
                    allowed=list(allowed_actions),terminal=bool(terminal),elapsed=float(elapsed))

    def targets(self,transitions,target_model=None):
        torch=self.torch
        with torch.no_grad():
            q=(self.target if target_model is None else target_model)(torch.tensor([t['next_x'] for t in transitions],dtype=torch.float32))
            mask=torch.tensor([[a in t['allowed'] for a in (0,1)] for t in transitions],dtype=torch.bool)
            q=q.masked_fill(~mask,-torch.inf).max(dim=1).values
            return torch.tensor([t['reward_scaled'] for t in transitions])+torch.tensor([t['discount'] for t in transitions])*q

    def td_loss(self,transitions,target_model=None):
        torch=self.torch
        if not transitions:return None
        with torch.no_grad():
            q=self.model(torch.tensor([t['x'] for t in transitions],dtype=torch.float32))
            actions=torch.tensor([t['action'] for t in transitions])
            loss=torch.nn.functional.smooth_l1_loss(q.gather(1,actions[:,None]).squeeze(1),self.targets(transitions,target_model))
            return float(loss)

    def update(self,previous,x,reward,elapsed,allowed_actions=(0,1),*,terminal=False):
        torch=self.torch;t=self.transition(previous,x,reward,elapsed,allowed_actions,terminal)
        diagnostic_target=float(self.targets([t])[0]);estimate=float(self.values(t['x'])[t['action']])
        self.buffer.append(t);before=torch.cat([p.detach().flatten() for p in self.model.parameters()]).clone();metrics=[]
        for _ in range(self.ratio):
            indices=self.replay_rng.choice(len(self.buffer),size=min(self.batch_size,len(self.buffer)),replace=False)
            batch=[self.buffer[int(i)] for i in indices]
            q=self.model(torch.tensor([s['x'] for s in batch],dtype=torch.float32))
            actions=torch.tensor([s['action'] for s in batch])
            loss=torch.nn.functional.smooth_l1_loss(q.gather(1,actions[:,None]).squeeze(1),self.targets(batch))
            self.optimizer.zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
            if not math.isfinite(float(loss.detach())) or not math.isfinite(float(norm)):raise ValueError('Nonfinite replay gradient')
            self.optimizer.step();self.optimizer_steps+=1
            sync=self.optimizer_steps%self.target_every==0
            if sync:self.target.load_state_dict(self.model.state_dict());self.target_syncs+=1
            metrics.append(dict(step=self.optimizer_steps,loss=float(loss.detach()),gradient_l2_before_clip=float(norm),
                                batch_size=len(batch),target_synced=sync))
        self.updates+=1
        after=torch.cat([p.detach().flatten() for p in self.model.parameters()])
        return dict(algorithm=PROTOCOL,updates=self.updates,optimizer_steps=self.optimizer_steps,gradient_steps=len(metrics),
                    loss=float(np.mean([m['loss'] for m in metrics])),gradient_l2_before_clip=max(m['gradient_l2_before_clip'] for m in metrics),
                    weight_delta_l2=float(torch.linalg.vector_norm(after-before)),reward_usd=float(reward),reward_scaled=t['reward_scaled'],
                    discount=t['discount'],target=diagnostic_target,td_error=diagnostic_target-estimate,
                    transition_seconds=float(elapsed),terminal=bool(terminal),replay_size=len(self.buffer),optimizer_metrics=metrics)

    def save(self,path):
        self.torch.save(dict(model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),updates=self.updates,
            rng=self.rng.bit_generator.state,replay_protocol=PROTOCOL,target=self.target.state_dict(),buffer=list(self.buffer),
            replay_rng=self.replay_rng.bit_generator.state,optimizer_steps=self.optimizer_steps,target_syncs=self.target_syncs,
            ratio=self.ratio,capacity=self.capacity,batch_size=self.batch_size,target_every=self.target_every),path)

    def restore(self,path):
        saved=self.torch.load(path,map_location='cpu',weights_only=False)
        super().restore(path)
        if saved.get('replay_protocol'):
            if saved['replay_protocol']!=PROTOCOL or any(saved[k]!=getattr(self,k) for k in ('ratio','capacity','batch_size','target_every')):
                raise ValueError('Replay checkpoint configuration differs')
            self.target.load_state_dict(saved['target']);self.buffer=deque(saved['buffer'],maxlen=self.capacity)
            self.replay_rng.bit_generator.state=saved['replay_rng'];self.optimizer_steps=saved['optimizer_steps'];self.target_syncs=saved['target_syncs']
        else:
            self.target.load_state_dict(self.model.state_dict());self.buffer.clear();self.optimizer_steps=self.updates;self.target_syncs=0
