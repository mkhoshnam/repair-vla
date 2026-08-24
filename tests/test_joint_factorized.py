"""Offline regression tests for the joint-factorized method.

No LIBERO model, rendering, or GPU is required. Run from the repository root:
    python tests/test_joint_factorized.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn(); PASS.append(name); print(f"  ok    {name}")
    except Exception as e:
        FAIL.append((name, repr(e))); print(f"  FAIL  {name}: {e}")


def make_module(obs_dim=169, token_dim=28, K=4):
    from rl.joint_factorized_encoder import JointFactorizedCapabilityModule
    return JointFactorizedCapabilityModule(
        obs_dim=obs_dim, act_dim=6, token_dim=token_dim, context_len=K,
        temporal_hidden=32, cap_dim=16, z_dim=24,
        transformer_layers=1, transformer_heads=4, transformer_ffn=32,
        device="cpu",
    )


def fake_obs(B=3, obs_dim=169):
    x = torch.randn(B, obs_dim)
    # Last 42 are joint-major Jacobian columns. Keep finite/nondegenerate.
    J = torch.randn(B, 7, 6) * 0.2
    x[:, -42:] = J.reshape(B, 42)
    return x


def t_shapes_and_finiteness():
    m = make_module()
    ctx = torch.randn(3, 4, 7 * 28)
    mask = torch.tensor([[0,0,1,1],[0,1,1,1],[1,1,1,1]], dtype=torch.float32)
    obs = fake_obs(3)
    z, cap = m.encode(ctx, mask, obs, return_joint=True)
    assert z.shape == (3,24)
    assert cap.shape == (3,7,16)
    assert torch.isfinite(z).all() and torch.isfinite(cap).all()


def t_shared_joint_permutation_equivariance():
    """No learned joint id: permuting physical joint tokens+J permutes caps."""
    torch.manual_seed(2)
    m = make_module(); m.eval()
    ctx = torch.randn(2,4,7,28)
    mask = torch.ones(2,4)
    obs = fake_obs(2)
    perm = torch.tensor([6,4,2,0,1,3,5])

    z1, c1 = m.encode(ctx.reshape(2,4,-1), mask, obs, return_joint=True)

    ctx2 = ctx[:,:,perm,:]
    obs2 = obs.clone()
    q = obs[:,8:15][:,perm]; qd = obs[:,15:22][:,perm]
    J = obs[:,-42:].reshape(2,7,6)[:,perm,:]
    obs2[:,8:15] = q; obs2[:,15:22] = qd; obs2[:,-42:] = J.reshape(2,42)
    z2, c2 = m.encode(ctx2.reshape(2,4,-1), mask, obs2, return_joint=True)

    # Global pooled representation is invariant to a pure relabeling when the
    # physical geometry travels with each token; per-joint outputs equivary.
    assert torch.allclose(z1, z2, atol=2e-5), (z1-z2).abs().max()
    assert torch.allclose(c1[:,perm,:], c2, atol=2e-5), (c1[:,perm,:]-c2).abs().max()


def t_empty_history_is_valid():
    m = make_module(); m.eval()
    ctx = torch.zeros(2,4,7*28); mask = torch.zeros(2,4); obs = fake_obs(2)
    z = m.encode(ctx, mask, obs)
    assert z.shape == (2,24) and torch.isfinite(z).all()


def t_query_parser_jacobian_orientation():
    m = make_module()
    obs = torch.zeros(1,169)
    Jjoint = torch.arange(42, dtype=torch.float32).reshape(1,7,6)
    obs[:,-42:] = Jjoint.reshape(1,42)
    q, J = m.query_from_raw_obs(obs)
    assert q.shape == (1,7,14) and J.shape == (1,6,7)
    assert torch.equal(J, Jjoint.transpose(1,2))


def t_representation_update_runs():
    torch.manual_seed(3)
    m = make_module()
    B,K=16,4
    ctx=torch.randn(B,K,7*28); mask=torch.ones(B,K)
    raw=fake_obs(B); obsn=torch.randn(B,169); act=torch.randn(B,6).tanh()
    y=torch.randn(B,13)*0.01
    m.update_context_stats(ctx.numpy().reshape(-1,28))
    m.update_target_stats(y.numpy())
    out=m.update(ctx,mask,obsn,raw,act,y)
    for k,v in out.items(): assert np.isfinite(v), (k,v)
    assert out["cap/L_total"] >= 0


def t_serialization_exact():
    torch.manual_seed(4)
    m=make_module(); m.eval()
    ctx=torch.randn(2,4,7*28); mask=torch.ones(2,4); obs=fake_obs(2)
    m.update_context_stats(ctx.numpy().reshape(-1,28))
    z1=m.encode(ctx,mask,obs).detach()
    d=m.checkpoint_state()
    m2=make_module(); m2.load_checkpoint_state(d, load_optimizer=False); m2.eval()
    z2=m2.encode(ctx,mask,obs).detach()
    assert torch.allclose(z1,z2,atol=1e-7)


def t_actor_zero_for_any_capability():
    from rl.factorized_sac import FactorizedSACAgent
    a=FactorizedSACAgent(169,24,6,device="cpu",hidden=32)
    o=torch.randn(20,169); z=torch.randn(20,24)*20
    u,_=a.actor(o,z,deterministic=True,with_logp=False)
    assert float(u.abs().max()) < 1e-7


def t_film_is_initially_neutral():
    from rl.factorized_sac import FiLMBlock
    f=FiLMBlock(24,32)
    h=torch.randn(5,32); z1=torch.randn(5,24); z2=torch.randn(5,24)
    assert torch.allclose(f(h,z1),h,atol=1e-7)
    assert torch.allclose(f(h,z2),h,atol=1e-7)


def t_factorized_sac_update_runs():
    from rl.factorized_sac import FactorizedSACAgent
    a=FactorizedSACAgent(20,8,3,device="cpu",hidden=32)
    B=32
    batch={
        "obs":torch.randn(B,20), "next_obs":torch.randn(B,20),
        "act":torch.randn(B,3).tanh(), "rew":torch.zeros(B),
        "term":torch.zeros(B), "disc":torch.full((B,),0.99),
    }
    z=torch.randn(B,8); nz=torch.randn(B,8)
    out=a.update(batch,z,nz)
    assert all(np.isfinite(v) for v in out.values())


def t_replay_exposes_raw_observations():
    from rl.sac import NStepReplayBuffer, RunningNorm
    b=NStepReplayBuffer(5,2,capacity=20,n_step=1,gamma=.99)
    o=np.array([1,2,3,4,5],dtype=np.float32)
    for _ in range(5): b.add(o,np.zeros(2,np.float32),0,o,False,True)
    rn=RunningNorm(5); rn.update(np.stack([o*0,o*2]))
    x=b.sample(3,torch.device("cpu"),obs_norm=rn)
    assert "raw_obs" in x and "raw_next_obs" in x
    assert torch.allclose(x["raw_obs"],torch.as_tensor(np.tile(o,(3,1))))
    assert not torch.allclose(x["obs"],x["raw_obs"])


def t_heldout_joint_variants_are_same_joint():
    from faults.multi_fault import joint_indices
    assert joint_indices(["2"]) == {2}
    assert joint_indices(["2:off=0.2","2:damp=50"]) == {2}


def t_left_padding_helper():
    from rl.joint_factorized_encoder import build_left_padded_history
    h=[np.full(14,i,dtype=np.float32) for i in [1,2,3]]
    c,m=build_left_padded_history(h,5,14)
    assert m.tolist()==[0,0,1,1,1]
    assert np.all(c[-3:,0]==[1,2,3])


def main():
    tests=[
        ("encoder shapes finite",t_shapes_and_finiteness),
        ("shared joint permutation equivariance",t_shared_joint_permutation_equivariance),
        ("empty history valid",t_empty_history_is_valid),
        ("Jacobian parser orientation",t_query_parser_jacobian_orientation),
        ("representation update",t_representation_update_runs),
        ("serialization exact",t_serialization_exact),
        ("zero residual for any capability",t_actor_zero_for_any_capability),
        ("FiLM neutral initialization",t_film_is_initially_neutral),
        ("factorized SAC update",t_factorized_sac_update_runs),
        ("replay raw observations",t_replay_exposes_raw_observations),
        ("whole held-out joint variants",t_heldout_joint_variants_are_same_joint),
        ("left padding helper",t_left_padding_helper),
    ]
    for n,f in tests: check(n,f)
    print(f"\n{len(PASS)}/{len(tests)} passed")
    if FAIL:
        for n,e in FAIL: print(f"FAIL {n}: {e}")
        raise SystemExit(1)


if __name__=="__main__": main()
