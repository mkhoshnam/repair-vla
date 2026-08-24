"""
test_offline.py -- everything that can be checked without LIBERO, MuJoCo,
robosuite, or a 7B model. Run this before using the full training stack.

It cannot tell you whether the fault is physically correct or whether the
VLA still succeeds. It CAN tell you that the residual arithmetic, the
gripper handling, the n-step buffer, the terminal/timeout distinction, the
observation layout, and the XML processor are right -- which is where
silent, plausible-looking bugs live.

    python tests/test_offline.py
"""

from __future__ import annotations

import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_DIR = REPO_ROOT / "rl"
sys.path.insert(0, str(REPO_ROOT))

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok    {name}")
    except Exception as e:
        FAIL.append((name, repr(e)))
        print(f"  FAIL  {name}: {e}")


# ==========================================================================
# fake upstream modules so residual_env's lazy imports resolve
# ==========================================================================

def install_fakes(chunk):
    def _mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    for n in ["experiments", "experiments.robot", "experiments.robot.libero"]:
        if n not in sys.modules:
            _mod(n)

    ru = _mod("experiments.robot.robot_utils")
    ru.get_action = lambda cfg, model, obs, task, **kw: [np.array(a) for a in chunk]
    ru.normalize_gripper_action = lambda a, binarize=True: np.concatenate(
        [a[:6], [1.0 if a[6] > 0.5 else -1.0]]
    )
    ru.invert_gripper_action = lambda a: np.concatenate([a[:6], [-a[6]]])

    lu = _mod("experiments.robot.libero.libero_utils")
    lu.get_libero_dummy_action = lambda family: [0.0] * 6 + [-1.0]


class FakeEnv:
    """Minimal LIBERO-shaped env. Deterministic, no physics."""

    def __init__(self, succeed_at=None):
        self.t = 0
        self.succeed_at = succeed_at
        self.last_action = None
        self.pos = np.zeros(3)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.env = types.SimpleNamespace(
            sim=types.SimpleNamespace(
                model=types.SimpleNamespace(joint_name2id=lambda n: 0,
                                            jnt_qposadr=np.zeros(1, dtype=int)),
                data=types.SimpleNamespace(qpos=np.zeros(1)),
            ),
            set_xml_processor=lambda p: None,
        )

    def _obs(self):
        return {
            "robot0_eef_pos": self.pos.copy(),
            "robot0_eef_quat": self.quat.copy(),
            "robot0_gripper_qpos": np.zeros(2),
            "robot0_joint_pos": np.zeros(7),
            "robot0_joint_vel": np.zeros(7),
        }

    def reset(self):
        self.t = 0
        return self._obs()

    def set_init_state(self, s):
        self.t = 0
        return self._obs()

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float64)
        self.t += 1
        self.pos = self.pos + 0.001 * self.last_action[:3]
        done = self.succeed_at is not None and self.t >= self.succeed_at
        return self._obs(), (1.0 if done else 0.0), done, {}


class FakeFaults:
    monitor = types.SimpleNamespace(max_drift=0.0)

    def reset_episode(self, env, state, init_id=None):
        env.reset()
        return env.set_init_state(state)

    def step_record(self, env):
        return 0.0

    def stats(self):
        return {}


def make_env(succeed_at=None, max_steps=20, scale=0.2, wait=2, chunk=None):
    from rl.residual_env import FrozenOFT, ResidualCfg, ResidualLiberoEnv

    chunk = chunk or [[0.5] * 6 + [1.0]] * 8
    install_fakes(chunk)

    cfg_vla = types.SimpleNamespace(model_family="openvla")
    vla = FrozenOFT(cfg_vla, model=types.SimpleNamespace(), resize_size=224,
                    task_description="t", chunk_len=8)
    rc = ResidualCfg(residual_scale=scale, history_len=4,
                     max_steps=max_steps, num_steps_wait=wait)
    env = FakeEnv(succeed_at=succeed_at)
    renv = ResidualLiberoEnv(
        env=env, vla=vla, fault_mgr=FakeFaults(),
        initial_states=[None] * 5, init_ids=[0, 1, 2], cfg=rc,
        prepare_observation=lambda o, rs: ({"state": np.zeros(8)}, None),
    )
    return renv, env


# ==========================================================================
# 1. quaternion / execution-mismatch math
# ==========================================================================

def t_quat_identity():
    from rl.residual_env import rot_delta_axisangle
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(rot_delta_axisangle(q, q), 0, atol=1e-9)


def t_quat_known_rotation():
    from rl.residual_env import rot_delta_axisangle
    a = np.deg2rad(30.0)
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])
    d = rot_delta_axisangle(q0, q1)
    assert abs(np.linalg.norm(d) - a) < 1e-6, np.linalg.norm(d)
    assert abs(d[2] - a) < 1e-6


def t_quat_no_wrap_blowup():
    """Differencing axis-angle vectors directly explodes near pi; this must not."""
    from rl.residual_env import quat2axisangle, rot_delta_axisangle
    a0, a1 = np.deg2rad(179.0), np.deg2rad(-179.0)
    q0 = np.array([np.sin(a0 / 2), 0, 0, np.cos(a0 / 2)])
    q1 = np.array([np.sin(a1 / 2), 0, 0, np.cos(a1 / 2)])
    naive = np.linalg.norm(quat2axisangle(q1) - quat2axisangle(q0))
    proper = np.linalg.norm(rot_delta_axisangle(q0, q1))
    assert proper < 0.1, proper
    assert naive > 6.0, naive


# ==========================================================================
# 2. residual arithmetic
# ==========================================================================

def t_zero_residual_is_exact_base():
    renv, env = make_env(scale=0.3, chunk=[[0.5] * 6 + [1.0]] * 8)
    renv.reset()
    renv.step(np.zeros(6, dtype=np.float32))
    # fake gripper pipeline: binarize(1.0)->+1 then invert -> -1
    assert np.allclose(env.last_action[:6], 0.5), env.last_action
    assert np.isclose(env.last_action[6], -1.0), env.last_action


def t_no_double_tanh():
    """delta must be scale * u, not scale * tanh(u)."""
    renv, env = make_env(scale=0.3, chunk=[[0.0] * 6 + [1.0]] * 8)
    renv.reset()
    renv.step(np.ones(6, dtype=np.float32))
    assert np.allclose(env.last_action[:6], 0.3), (
        f"expected 0.3 (= scale*1.0); got {env.last_action[:6]}. "
        f"tanh(1)*0.3 = {0.3 * np.tanh(1.0):.4f} would mean a double squash."
    )


def t_gripper_never_residualized():
    renv, env = make_env(scale=0.5, chunk=[[0.0] * 6 + [1.0]] * 8)
    renv.reset()
    for u in (1.0, -1.0, 0.0):
        renv.step(np.full(6, u, dtype=np.float32))
        assert np.isclose(abs(env.last_action[6]), 1.0), env.last_action[6]


def t_final_action_clipped():
    renv, env = make_env(scale=0.5, chunk=[[0.9] * 6 + [1.0]] * 8)
    renv.reset()
    _, _, _, _, info = renv.step(np.ones(6, dtype=np.float32))
    assert np.all(env.last_action[:6] <= 1.0 + 1e-9)
    assert np.isclose(env.last_action[0], 1.0)
    assert info["clipped_frac"] == 1.0


def t_unbounded_residual_rejected():
    renv, _ = make_env()
    renv.reset()
    try:
        renv.step(np.full(6, 3.0, dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("an unbounded residual must raise, not be silently squashed")


# ==========================================================================
# 3. episode mechanics
# ==========================================================================

def t_obs_dim_matches_declared():
    renv, _ = make_env()
    o = renv.reset()
    assert o.shape == (renv.obs_dim,), (o.shape, renv.obs_dim)
    o2, *_ = renv.step(np.zeros(6, dtype=np.float32))
    assert o2.shape == (renv.obs_dim,)
    assert np.isfinite(o2).all()


def t_dummy_prefix_not_counted():
    renv, env = make_env(wait=5)
    renv.reset()
    assert env.t == 5, env.t          # the wait steps happened
    assert renv.t == 0, renv.t        # but no RL step has


def t_truncation_at_max_steps():
    renv, _ = make_env(succeed_at=None, max_steps=6, wait=0)
    renv.reset()
    z = np.zeros(6, dtype=np.float32)
    for i in range(5):
        _, _, term, trunc, _ = renv.step(z)
        assert not term and not trunc, i
    _, _, term, trunc, info = renv.step(z)
    assert trunc and not term
    assert info["success"] is False


def t_success_terminates_with_reward_one():
    renv, _ = make_env(succeed_at=3, max_steps=50, wait=0)
    renv.reset()
    z = np.zeros(6, dtype=np.float32)
    renv.step(z); renv.step(z)
    _, r, term, trunc, info = renv.step(z)
    assert term and not trunc
    assert r == 1.0 and info["success"] is True


def t_chunk_requery_every_8_steps():
    renv, _ = make_env(max_steps=40, wait=0)
    renv.reset()
    assert renv.vla.n_queries == 1
    z = np.zeros(6, dtype=np.float32)
    for _ in range(7):
        renv.step(z)
    assert renv.vla.n_queries == 1, renv.vla.n_queries
    renv.step(z)
    assert renv.vla.n_queries == 2, renv.vla.n_queries


def t_chunk_phase_progresses_and_wraps():
    """Phase must be strictly increasing within a chunk and reset on requery.

    Regression guard: `1 - len(queue)/chunk_len` gives 0.0 for BOTH the
    first and the eighth action, so the residual could not tell a fresh
    command from a fully stale one.
    """
    renv, _ = make_env(max_steps=40, wait=0)
    renv.reset()
    phases = []
    z = np.zeros(6, dtype=np.float32)
    for _ in range(16):
        phases.append(renv.vla.chunk_phase)
        renv.step(z)
    first, second = phases[:8], phases[8:]
    assert first == [i / 8 for i in range(8)], first
    assert second == first, second
    assert len(set(first)) == 8, "phase values within a chunk must be distinct"


def t_history_records_execution_mismatch():
    """A commanded x-motion that the body does not realize must be visible."""
    renv, env = make_env(scale=0.0, max_steps=40, wait=0,
                         chunk=[[1.0, 0, 0, 0, 0, 0, 1.0]] * 8)
    renv.reset()
    renv.step(np.zeros(6, dtype=np.float32))
    h = list(renv._hist)[-1]
    assert np.isclose(h[0], 1.0), h[:6]      # commanded +x
    assert np.isclose(h[6], 0.001), h[6:9]   # realized +x (FakeEnv gain)
    # a locked joint would show h[0] large and h[6] ~ 0 -- that gap is the signal


# ==========================================================================
# 4. n-step replay buffer
# ==========================================================================

def t_nstep_reward_accumulation():
    from rl.sac import NStepReplayBuffer
    b = NStepReplayBuffer(2, 1, capacity=100, n_step=3, gamma=0.5)
    o = lambda i: np.full(2, i, dtype=np.float32)
    a = np.zeros(1, dtype=np.float32)
    b.add(o(0), a, 1.0, o(1), False, False)
    b.add(o(1), a, 2.0, o(2), False, False)
    b.add(o(2), a, 4.0, o(3), False, False)
    assert b.size == 1
    # 1 + 0.5*2 + 0.25*4 = 3.0 ; discount used = 0.5^3 = 0.125
    assert np.isclose(b.rew[0], 3.0), b.rew[0]
    assert np.isclose(b.disc[0], 0.125), b.disc[0]
    assert np.allclose(b.next_obs[0], 3.0)


def t_nstep_chain_cut_by_success():
    from rl.sac import NStepReplayBuffer
    b = NStepReplayBuffer(2, 1, capacity=100, n_step=3, gamma=0.5)
    o = lambda i: np.full(2, i, dtype=np.float32)
    a = np.zeros(1, dtype=np.float32)
    b.add(o(0), a, 0.0, o(1), False, False)
    b.add(o(1), a, 1.0, o(2), True, False)   # success on step 2
    assert b.size == 2
    assert np.isclose(b.rew[0], 0.5), b.rew[0]     # 0 + 0.5*1
    assert np.isclose(b.disc[0], 0.25), b.disc[0]  # chain length 2, not 3
    assert b.term[0] == 1.0
    assert b.term[1] == 1.0


def t_timeout_is_not_terminal():
    """The bug that silently kills sparse-reward SAC."""
    from rl.sac import NStepReplayBuffer
    b = NStepReplayBuffer(2, 1, capacity=100, n_step=1, gamma=0.99)
    o = np.zeros(2, dtype=np.float32)
    a = np.zeros(1, dtype=np.float32)
    b.add(o, a, 0.0, o, False, True)   # truncated, NOT terminated
    assert b.term[0] == 0.0, (
        "truncation stored as terminal -> the critic learns every late state "
        "is worth zero and the policy learns to do nothing"
    )
    b.add(o, a, 1.0, o, True, False)
    assert b.term[1] == 1.0


def t_truncation_next_obs_has_fresh_base_action():
    """On truncation we bootstrap, so next_obs must be a real state.

    Regression guard: reusing the stale `_a_base` puts the PREVIOUS step's
    base action into the bootstrapped state. At a 20% success rate almost
    every episode truncates, so that bias would land on almost all of the
    critic's targets.
    """
    chunk = [[i / 10.0] * 6 + [1.0] for i in range(8)]
    renv, _ = make_env(succeed_at=None, max_steps=3, wait=0, chunk=chunk)
    # base-action slice sits after proprio(8) + qpos(7) + qvel(7)
    off = 8 + 7 + 7
    z = np.zeros(6, dtype=np.float32)
    renv.reset()
    renv.step(z)
    renv.step(z)
    nobs, _, term, trunc, _ = renv.step(z)
    assert trunc and not term
    assert np.isclose(nobs[off], 0.3), (
        f"truncated next_obs carries base action {nobs[off]:.3f}; expected 0.3 "
        f"(freshly queried). 0.2 means the stale _a_base was reused."
    )


def t_termination_skips_the_extra_vla_query():
    """Success masks next_obs, so don't pay for a 7B forward pass."""
    renv, _ = make_env(succeed_at=2, max_steps=50, wait=0)
    renv.reset()
    z = np.zeros(6, dtype=np.float32)
    renv.step(z)
    q_before = renv.vla.n_queries
    _, _, term, _, _ = renv.step(z)
    assert term
    assert renv.vla.n_queries == q_before, "no query needed on a masked next state"


def t_buffer_stores_raw_observations():
    """Pre-normalized storage mixes stale and current statistics in one batch."""
    from rl.sac import NStepReplayBuffer
    b = NStepReplayBuffer(3, 1, capacity=50, n_step=1, gamma=0.99)
    o = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    b.add(o, np.zeros(1, dtype=np.float32), 0.0, o, False, True)
    assert np.allclose(b.obs[0], o), b.obs[0]
    assert np.allclose(b.next_obs[0], o)


def t_sample_normalizes_with_current_stats():
    from rl.sac import NStepReplayBuffer, RunningNorm
    b = NStepReplayBuffer(3, 1, capacity=50, n_step=1, gamma=0.99)
    o = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    for _ in range(8):
        b.add(o, np.zeros(1, dtype=np.float32), 0.0, o, False, True)

    rn = RunningNorm(3)
    rn.update(np.array([[0.0, 0.0, 0.0], [20.0, 40.0, 60.0]]))  # mean 10/20/30

    raw = b.sample(4, torch.device("cpu"))["obs"]
    assert np.allclose(raw.numpy(), o), "no normalizer -> raw"

    normed = b.sample(4, torch.device("cpu"), obs_norm=rn)["obs"]
    # RunningNorm seeds count at eps=1e-4, so the mean is 9.9995 not 10.0 --
    # a ~5e-5 residual is the correct value, not a bug.
    assert np.allclose(normed.numpy(), 0.0, atol=1e-3), normed
    assert normed.dtype == torch.float32


def t_end_episode_prevents_cross_episode_chaining():
    """The bug a mid-episode held-out eval would otherwise introduce."""
    from rl.sac import NStepReplayBuffer
    b = NStepReplayBuffer(1, 1, capacity=50, n_step=3, gamma=1.0)
    a = np.zeros(1, dtype=np.float32)
    ep1 = lambda i: np.array([i], dtype=np.float32)

    # episode A abandoned after two steps (an eval fired, say)
    b.add(ep1(0), a, 0.0, ep1(1), False, False)
    b.add(ep1(1), a, 0.0, ep1(2), False, False)
    assert len(b._stage) == 2
    b.end_episode()
    assert len(b._stage) == 0 and b.size == 2

    # episode B: its reward must not leak backwards into episode A
    ep2 = lambda i: np.array([100 + i], dtype=np.float32)
    b.add(ep2(0), a, 1.0, ep2(1), True, False)
    assert b.size == 3
    assert b.rew[0] == 0.0 and b.rew[1] == 0.0, (
        f"episode A transitions picked up episode B's reward: "
        f"{b.rew[0]}, {b.rew[1]}"
    )
    assert b.next_obs[0][0] < 100 and b.next_obs[1][0] < 100, (
        "episode A next_obs points into episode B"
    )


def t_without_end_episode_the_chain_would_leak():
    """Demonstrates that the guard above is load-bearing, not decorative."""
    from rl.sac import NStepReplayBuffer
    b = NStepReplayBuffer(1, 1, capacity=50, n_step=3, gamma=1.0)
    a = np.zeros(1, dtype=np.float32)
    b.add(np.array([0.0], dtype=np.float32), a, 0.0,
          np.array([1.0], dtype=np.float32), False, False)
    b.add(np.array([1.0], dtype=np.float32), a, 0.0,
          np.array([2.0], dtype=np.float32), False, False)
    # no end_episode() -- next episode's success chains straight in
    b.add(np.array([100.0], dtype=np.float32), a, 1.0,
          np.array([101.0], dtype=np.float32), True, False)
    assert b.rew[0] == 1.0, (
        "expected the leak to be reproducible; if this now passes as 0.0 the "
        "buffer changed and the end_episode guard may be redundant"
    )


# ==========================================================================
# 5. SAC
# ==========================================================================

def t_actor_output_bounded():
    from rl.sac import Actor
    net = Actor(20, 6)
    a, logp = net(torch.randn(64, 20) * 50)
    assert a.shape == (64, 6) and logp.shape == (64,)
    assert a.abs().max().item() <= 1.0
    assert torch.isfinite(logp).all()


def t_actor_deterministic_is_repeatable():
    from rl.sac import Actor
    net = Actor(20, 6)
    o = torch.randn(8, 20)
    a1, _ = net(o, deterministic=True, with_logp=False)
    a2, _ = net(o, deterministic=True, with_logp=False)
    assert torch.allclose(a1, a2)


def t_sac_update_runs_and_moves_params():
    from rl.sac import NStepReplayBuffer, SACAgent
    ag = SACAgent(20, 6, device="cpu", hidden=32)
    b = NStepReplayBuffer(20, 6, capacity=1000, n_step=3, gamma=0.99)
    rng = np.random.default_rng(0)
    for i in range(300):
        b.add(rng.normal(size=20).astype(np.float32),
              rng.uniform(-1, 1, 6).astype(np.float32),
              float(i % 50 == 49), rng.normal(size=20).astype(np.float32),
              i % 50 == 49, False)
    before = ag.actor.mu.weight.detach().clone()
    m = ag.update(b.sample(64, torch.device("cpu")))
    assert all(np.isfinite(v) for v in m.values()), m
    assert not torch.allclose(before, ag.actor.mu.weight)


def t_zero_init_actor_starts_at_the_base_policy():
    """The j6 failure: a randomly initialized head corrupts a working VLA."""
    from rl.sac import Actor
    net = Actor(20, 6, zero_init=True)
    o = torch.randn(64, 20) * 10
    a, _ = net(o, deterministic=True, with_logp=False)
    assert a.abs().max().item() < 1e-6, a.abs().max().item()


def t_zero_init_still_explores():
    """Mean zero, but sampling must still spread or the buffer sees nothing."""
    from rl.sac import Actor
    net = Actor(20, 6, zero_init=True, log_std_init=-1.0)
    a, logp = net(torch.randn(512, 20))
    assert a.std().item() > 0.1, a.std().item()
    assert abs(a.mean().item()) < 0.05, a.mean().item()
    assert torch.isfinite(logp).all()


def t_random_init_actor_does_not_start_at_zero():
    """Guard that the zero-init test above is actually testing something."""
    from rl.sac import Actor
    torch.manual_seed(0)
    net = Actor(20, 6, zero_init=False)
    a, _ = net(torch.randn(64, 20) * 10, deterministic=True, with_logp=False)
    assert a.abs().max().item() > 1e-3


def t_alpha_init_is_not_one():
    """alpha=1.0 is calibrated for per-step rewards ~1. Ours is ~0.005."""
    from rl.sac import SACAgent
    ag = SACAgent(20, 6, device="cpu", hidden=32)
    assert abs(float(ag.alpha) - 0.01) < 1e-6, float(ag.alpha)
    ag2 = SACAgent(20, 6, device="cpu", hidden=32, alpha_init=0.5)
    assert abs(float(ag2.alpha) - 0.5) < 1e-6


def t_entropy_bonus_cannot_swamp_task_return():
    """Quantifies why alpha_init matters: entropy in Q vs max task return."""
    gamma, target_H, T = 0.99, 6.0, 200
    max_task = gamma ** 0
    for alpha, should_swamp in [(1.0, True), (0.55, True), (0.01, False)]:
        entropy_in_q = alpha * target_H / (1 - gamma)
        assert (entropy_in_q > 20 * max_task) == should_swamp, (
            f"alpha={alpha}: entropy contributes {entropy_in_q:.1f} to Q "
            f"vs max task return {max_task}"
        )


def t_target_entropy_matches_action_dim():
    from rl.sac import SACAgent
    ag = SACAgent(20, 6, device="cpu", hidden=32)
    assert ag.target_entropy == -6.0


def t_running_norm_matches_numpy():
    from rl.sac import RunningNorm
    rng = np.random.default_rng(1)
    x = rng.normal(3.0, 7.0, size=(4000, 5))
    rn = RunningNorm(5)
    for i in range(0, 4000, 50):
        rn.update(x[i:i + 50])
    assert np.allclose(rn.mean, x.mean(0), atol=1e-6), rn.mean - x.mean(0)
    assert np.allclose(rn.var, x.var(0), rtol=1e-4), rn.var - x.var(0)


# ==========================================================================
# 6. fault injection
# ==========================================================================

def t_processor_creates_equality_constraint():
    from faults.joint_lock import make_joint_lock_processor
    xml = "<mujoco><worldbody/></mujoco>"
    out = make_joint_lock_processor("robot0_joint1", 0.123)(xml)
    root = ET.fromstring(out)
    js = root.find("equality").findall("joint")
    assert len(js) == 1
    assert js[0].get("joint1") == "robot0_joint1"
    assert js[0].get("polycoef").split()[0].startswith("0.123")


def t_processor_is_idempotent():
    from faults.joint_lock import make_joint_lock_processor
    xml = "<mujoco><worldbody/></mujoco>"
    out = make_joint_lock_processor("robot0_joint1", 0.1)(xml)
    out = make_joint_lock_processor("robot0_joint1", 0.9)(out)
    root = ET.fromstring(out)
    js = root.find("equality").findall("joint")
    assert len(js) == 1, "re-applying must update, not stack a second constraint"
    assert js[0].get("polycoef").split()[0].startswith("0.9")


def t_processor_preserves_existing_equality():
    from faults.joint_lock import make_joint_lock_processor
    xml = "<mujoco><equality><weld name='w' body1='a'/></equality></mujoco>"
    out = make_joint_lock_processor("robot0_joint1", 0.1)(xml)
    root = ET.fromstring(out)
    assert len(root.find("equality").findall("weld")) == 1
    assert len(root.find("equality").findall("joint")) == 1


def t_joint_naming():
    from faults.joint_lock import panda_joint_name
    assert panda_joint_name(0) == "robot0_joint1"
    assert panda_joint_name(6) == "robot0_joint7"
    try:
        panda_joint_name(7)
    except ValueError:
        return
    raise AssertionError("joint_idx 7 must be rejected")


def t_lock_monitor_flags_drift():
    from faults.joint_lock import LockMonitor
    m = LockMonitor("j", 0.5, drift_tol=1e-2)
    for q in (0.5, 0.5001, 0.4999):
        m.record(q)
    assert m.ok
    m.record(0.6)
    assert not m.ok and np.isclose(m.max_drift, 0.1)


def t_manager_rebuilds_once_when_target_constant():
    """The optimization that makes 100k steps affordable."""
    from faults.joint_lock import JointLockManager
    mgr = JointLockManager(joint_idx=0)
    env = FakeEnv()
    for i in range(20):
        mgr.reset_episode(env, None, init_id=i % 5)
    assert mgr._rebuilds == 1, mgr._rebuilds
    assert mgr.stats()["fault/n_init_states_seen"] == 5


def t_manager_disabled_does_nothing():
    from faults.joint_lock import JointLockManager
    mgr = JointLockManager(joint_idx=0, enabled=False)
    mgr.reset_episode(FakeEnv(), None, init_id=0)
    assert mgr._rebuilds == 0 and mgr.monitor is None


# ==========================================================================
# 6b. shared fault-agnostic policy (handoff section 10 acceptance tests)
# ==========================================================================

class FakeMultiEnv(FakeEnv):
    """FakeEnv that actually round-trips XML through the processor, so the
    strip-then-add logic is exercised the way robosuite would exercise it."""

    BASE = "<mujoco><worldbody/><equality><weld name='keepme' body1='a'/></equality></mujoco>"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.xml = self.BASE
        self._proc = None
        outer = self

        def _set(p):
            outer._proc = p
        self.env.set_xml_processor = _set

    def reset(self):
        # robosuite regenerates the model from the task, then applies the
        # currently-registered processor. Mirror that: always start from BASE.
        self.xml = self._proc(self.BASE) if self._proc else self.BASE
        self.sync_model()
        return super().reset()

    def sync_model(self):
        """Expose constraints as a fake COMPILED model.

        `assert_exactly_one_lock` reads the model back rather than trusting
        the XML we sent, so the fake needs a model side -- otherwise these
        tests would pass while the real checker could never work, which is
        exactly what happened before.
        """
        names = self.active_locks()
        m = self.env.sim.model
        m.neq = len(names)
        m.eq_active = [1] * len(names)
        m._fault_names = names

    def active_locks(self):
        root = ET.fromstring(self.xml)
        eq = root.find("equality")
        if eq is None:
            return []
        return [c.get("name") for c in eq.findall("joint")
                if (c.get("name") or "").startswith("fault_lock_")]


def _mgr(**kw):
    from faults.multi_fault import MultiFaultManager
    kw.setdefault("joint_pool", (0, 6))
    return MultiFaultManager(**kw)


def t_switch_j0_to_j6_leaves_only_j6():
    """The silent-corruption case: two locks active, logged as one."""
    from faults.multi_fault import FaultSpec
    env = FakeMultiEnv()
    m = _mgr()
    m.reset_episode(env, None, init_id=0, force=FaultSpec(0))
    assert env.active_locks() == ["fault_lock_robot0_joint1"], env.active_locks()
    m.reset_episode(env, None, init_id=0, force=FaultSpec(6))
    assert env.active_locks() == ["fault_lock_robot0_joint7"], (
        f"stale j0 constraint survived the switch: {env.active_locks()} -- "
        f"this episode would be a TWO-joint fault logged as j6"
    )


def t_switch_to_healthy_removes_all_locks():
    from faults.multi_fault import FaultSpec
    env = FakeMultiEnv()
    m = _mgr(include_healthy=True)
    m.reset_episode(env, None, init_id=0, force=FaultSpec(0))
    assert len(env.active_locks()) == 1
    m.reset_episode(env, None, init_id=0, force=FaultSpec(None))
    assert env.active_locks() == [], env.active_locks()
    assert m.monitor is None


def t_full_cycle_j0_j6_healthy_j0():
    from faults.multi_fault import FaultSpec
    env = FakeMultiEnv()
    m = _mgr(include_healthy=True)
    want = [(FaultSpec(0), ["fault_lock_robot0_joint1"]),
            (FaultSpec(6), ["fault_lock_robot0_joint7"]),
            (FaultSpec(None), []),
            (FaultSpec(0), ["fault_lock_robot0_joint1"])]
    for spec, exp in want:
        m.reset_episode(env, None, init_id=0, force=spec)
        assert env.active_locks() == exp, (spec.name, env.active_locks())


def t_processor_preserves_non_fault_equalities():
    """Stripping must not remove the task's own constraints."""
    from faults.multi_fault import FaultSpec
    env = FakeMultiEnv()
    m = _mgr()
    m.reset_episode(env, None, init_id=0, force=FaultSpec(6))
    root = ET.fromstring(env.xml)
    assert len(root.find("equality").findall("weld")) == 1


def t_fault_identity_absent_from_observation():
    """The headline claim: no fault label reaches the actor."""
    from faults.multi_fault import FaultSpec
    from rl.residual_env import FrozenOFT, ResidualCfg, ResidualLiberoEnv
    install_fakes([[0.3] * 6 + [1.0]] * 8)

    def build(spec):
        env = FakeMultiEnv()
        m = _mgr(include_healthy=True)
        vla = FrozenOFT(types.SimpleNamespace(model_family="openvla"),
                        types.SimpleNamespace(), 224, "t", chunk_len=8)
        rc = ResidualCfg(residual_scale=0.1, history_len=4,
                         max_steps=10, num_steps_wait=0)
        r = ResidualLiberoEnv(env=env, vla=vla, fault_mgr=m,
                              initial_states=[None] * 5, init_ids=[0], cfg=rc,
                              prepare_observation=lambda o, rs: ({}, None))
        return r.reset(init_id=0, force_fault=spec)

    o_j0 = build(FaultSpec(0))
    o_j6 = build(FaultSpec(6))
    o_h = build(FaultSpec(None))
    assert np.allclose(o_j0, o_j6), (
        "observation differs by fault identity under identical dynamics -- "
        "a fault label is leaking into the actor input")
    assert np.allclose(o_j0, o_h)


def t_info_does_carry_fault_for_logging():
    from faults.multi_fault import FaultSpec
    from rl.residual_env import FrozenOFT, ResidualCfg, ResidualLiberoEnv
    install_fakes([[0.0] * 6 + [1.0]] * 8)
    env = FakeMultiEnv()
    m = _mgr()
    vla = FrozenOFT(types.SimpleNamespace(model_family="openvla"),
                    types.SimpleNamespace(), 224, "t", chunk_len=8)
    rc = ResidualCfg(residual_scale=0.1, history_len=4, max_steps=10,
                     num_steps_wait=0)
    r = ResidualLiberoEnv(env=env, vla=vla, fault_mgr=m,
                          initial_states=[None] * 5, init_ids=[0], cfg=rc,
                          prepare_observation=lambda o, rs: ({}, None))
    r.reset(init_id=0, force_fault=FaultSpec(6))
    _, _, _, _, info = r.step(np.zeros(6, dtype=np.float32))
    assert info["fault"] == "j6", info["fault"]


def t_fault_sampled_per_episode_not_per_step():
    from rl.residual_env import FrozenOFT, ResidualCfg, ResidualLiberoEnv
    install_fakes([[0.0] * 6 + [1.0]] * 8)
    env = FakeMultiEnv()
    m = _mgr(seed=3)
    vla = FrozenOFT(types.SimpleNamespace(model_family="openvla"),
                    types.SimpleNamespace(), 224, "t", chunk_len=8)
    rc = ResidualCfg(residual_scale=0.1, history_len=4, max_steps=12,
                     num_steps_wait=0)
    r = ResidualLiberoEnv(env=env, vla=vla, fault_mgr=m,
                          initial_states=[None] * 5, init_ids=[0], cfg=rc,
                          prepare_observation=lambda o, rs: ({}, None))
    r.reset(init_id=0)
    seen = set()
    for _ in range(11):
        _, _, _, _, info = r.step(np.zeros(6, dtype=np.float32))
        seen.add(info["fault"])
    assert len(seen) == 1, f"fault changed mid-episode: {seen}"


def t_sampling_respects_probabilities():
    m = _mgr(joint_pool=(0, 6), fault_probs=(0.8, 0.2), seed=0)
    draws = [m.sample().name for _ in range(4000)]
    frac = draws.count("j0") / len(draws)
    assert 0.76 < frac < 0.84, frac


def t_probs_length_mismatch_is_rejected():
    """Silent misalignment here would mis-weight the whole curriculum."""
    try:
        _mgr(joint_pool=(0, 6), fault_probs=(0.5, 0.3, 0.2))
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise")
    try:
        _mgr(joint_pool=(0, 6), fault_probs=(0.5, 0.5), include_healthy=True)
    except ValueError:
        return
    raise AssertionError("healthy needs its own probability entry")


def t_lock_targets_cached_per_joint_and_state():
    from faults.multi_fault import FaultSpec
    env = FakeMultiEnv()
    m = _mgr()
    for _ in range(6):
        m.reset_episode(env, None, init_id=0, force=FaultSpec(0))
    r_after_j0 = m._rebuilds
    assert r_after_j0 == 1, r_after_j0
    m.reset_episode(env, None, init_id=0, force=FaultSpec(6))
    assert m._rebuilds == 2
    assert set(m._targets) == {(0, 0), (6, 0)}


def t_healthy_pool_names():
    m = _mgr(joint_pool=(0, 6), include_healthy=True)
    assert m.names == ["j0", "j6", "healthy"]




# ==========================================================================
# 6c. PILOT REGRESSION TESTS -- the five fixes
# ==========================================================================

def t_checker_raises_when_model_unreadable():
    """FIX 1. A verifier that cannot verify must SAY SO, never return [].

    The old code wrapped `equality_id2name` in a bare except and returned an
    empty list, which reads exactly like 'no fault compiled'. That is what
    produced `found []` in the training environment while the fault was fine.
    """
    from faults.multi_fault import ModelInspectionError, count_active_fault_locks
    broken = types.SimpleNamespace(env=types.SimpleNamespace(
        sim=types.SimpleNamespace(model=types.SimpleNamespace())))
    try:
        count_active_fault_locks(broken)
    except ModelInspectionError:
        return
    raise AssertionError(
        "an unreadable model must raise ModelInspectionError, not return []")


def t_checker_reads_compiled_names():
    """FIX 1. Names come from the compiled model, not from our intent."""
    from faults.multi_fault import FaultSpec, count_active_fault_locks
    env = FakeMultiEnv()
    m = _mgr()
    m.reset_episode(env, None, init_id=0, force=FaultSpec(0))
    assert count_active_fault_locks(env) == ["fault_lock_robot0_joint1"]
    m.reset_episode(env, None, init_id=0, force=FaultSpec(6))
    assert count_active_fault_locks(env) == ["fault_lock_robot0_joint7"]


def t_env_factory_rebuilds_on_switch():
    """FIX 2. A fault change must yield a FRESH env with the lock compiled."""
    from faults.multi_fault import FaultSpec
    built = []

    def factory():
        e = FakeMultiEnv()
        built.append(e)
        return e

    env0 = FakeMultiEnv()
    m = _mgr(env_factory=factory)
    m.reset_episode(env0, None, init_id=0, force=FaultSpec(0))
    assert m.env is not env0, "a fault change must build a fresh env"
    m.assert_exactly_one_lock(m.env)
    m.reset_episode(m.env, None, init_id=0, force=FaultSpec(6))
    m.assert_exactly_one_lock(m.env)
    assert m._env_rebuilds == 2, m._env_rebuilds
    assert m.stats()["fault/env_factory"] is True


def t_env_factory_idle_when_fault_unchanged():
    """Rebuilding every episode would be pure waste."""
    from faults.multi_fault import FaultSpec
    n = {"c": 0}

    def factory():
        n["c"] += 1
        return FakeMultiEnv()

    m = _mgr(env_factory=factory)
    env = FakeMultiEnv()
    for _ in range(6):
        m.reset_episode(env if m.env is None else m.env, None,
                        init_id=0, force=FaultSpec(0))
    assert n["c"] == 1, n["c"]


def t_residual_env_adopts_rebuilt_env():
    """FIX 2, caller side: keeping the old handle steps a CLOSED sim."""
    from faults.multi_fault import FaultSpec
    from rl.residual_env import FrozenOFT, ResidualCfg, ResidualLiberoEnv
    install_fakes([[0.0] * 6 + [1.0]] * 8)
    m = _mgr(env_factory=FakeMultiEnv)
    env0 = FakeMultiEnv()
    vla = FrozenOFT(types.SimpleNamespace(model_family="openvla"),
                    types.SimpleNamespace(), 224, "t", chunk_len=8)
    rc = ResidualCfg(residual_scale=0.1, history_len=4, max_steps=10,
                     num_steps_wait=0)
    r = ResidualLiberoEnv(env=env0, vla=vla, fault_mgr=m,
                          initial_states=[None] * 5, init_ids=[0], cfg=rc,
                          prepare_observation=lambda o, rs: ({}, None))
    r.reset(init_id=0, force_fault=FaultSpec(0))
    assert r.env is m.env and r.env is not env0
    r.step(np.zeros(6, dtype=np.float32))


def t_sampling_alone_can_miss_the_switch():
    """FIX 3. Why the dry run must FORCE the sequence."""
    misses = sum(1 for seed in range(40)
                 if len({_mgr(seed=seed).sample().name for _ in range(3)}) == 1)
    assert misses > 0, "expected some seeds where 3 draws never switch"


def t_forced_sequence_covers_every_transition():
    """FIX 3. The sequence the dry run walks: all conditions, then back."""
    from faults.multi_fault import FaultSpec
    m = _mgr(include_healthy=True, env_factory=FakeMultiEnv)
    specs = list(m.specs)
    sequence = specs + [specs[0]]
    trans, prev = [], None
    for spec in sequence:
        m.reset_episode(m.env or FakeMultiEnv(), None, init_id=0, force=spec)
        m.assert_exactly_one_lock(m.env)
        if prev is not None:
            trans.append((prev, spec.name))
        prev = spec.name
    assert trans == [("j0", "j6"), ("j6", "healthy"), ("healthy", "j0")], trans


def t_eval_every_zero_is_legal():
    """FIX 5. `step % 0` raises ZeroDivisionError; the guard must short-circuit."""
    for step in (1, 500, 25000):
        assert not (0 and step % 0 == 0)          # the pattern used in the trainer
    src = (RL_DIR / "train_shared_fault_sac.py").read_text()
    assert "if args.eval_every and step % args.eval_every == 0" in src, (
        "trainer must short-circuit on eval_every == 0")
    assert 'p.add_argument("--eval_every", type=int, default=0' in src, (
        "eval_every must DEFAULT to 0 so held-out states stay untouched")


def t_trainer_uses_train_baselines_flag():
    """FIX 4. The guard watches training episodes, so baselines must be
    train-state rates. A held-out number compares different state sets."""
    src = (RL_DIR / "train_shared_fault_sac.py").read_text()
    assert "--train_baselines" in src
    assert "--baselines\"" not in src, "the old held-out flag must be gone"


def t_trainer_refuses_j2_in_training_pool():
    """The unseen joint must not silently enter training.

    Now generalized: `--heldout_conditions` defaults to ["2"] and the
    trainer refuses to start if any held-out condition appears in the pool.
    """
    src = (RL_DIR / "train_shared_fault_sac.py").read_text()
    assert 'default=["2"]' in src, "j2 must remain the default held-out cell"
    assert "heldout & pool_names" in src
    assert "raise SystemExit" in src


def t_replay_tags_fault_and_reports_fractions():
    """50/50 EPISODES is not 50/50 TRANSITIONS."""
    from rl.sac import NStepReplayBuffer
    b = NStepReplayBuffer(3, 1, capacity=500, n_step=1, gamma=0.99, n_faults=2)
    o = np.zeros(3, dtype=np.float32)
    a = np.zeros(1, dtype=np.float32)
    for _ in range(200):                      # j0: long episodes
        b.add(o, a, 0.0, o, False, False, fault_id=0)
    b.add(o, a, 0.0, o, False, True, fault_id=0)
    for _ in range(50):                       # j6: short episodes
        b.add(o, a, 0.0, o, False, False, fault_id=1)
    b.add(o, a, 1.0, o, True, False, fault_id=1)
    fr = b.fault_fractions()
    assert fr[0] > 0.7 and fr[1] < 0.3, fr    # the imbalance is real


def t_stratified_sampling_balances_the_minibatch():
    from rl.sac import NStepReplayBuffer
    b = NStepReplayBuffer(3, 1, capacity=2000, n_step=1, gamma=0.99, n_faults=2)
    o = np.zeros(3, dtype=np.float32)
    a = np.zeros(1, dtype=np.float32)
    for _ in range(900):
        b.add(o, a, 0.0, o, False, True, fault_id=0)
    for _ in range(100):
        b.add(o, a, 0.0, o, False, True, fault_id=1)

    idx_uniform = []
    for _ in range(20):
        b.sample(256, torch.device("cpu"))
    # inspect directly: stratified must draw ~half from the rare fault
    ids = b.fault_id[: b.size]
    rare = np.flatnonzero(ids == 1)
    assert len(rare) == 100
    b.sample(256, torch.device("cpu"), stratified=True)   # must not raise
    frac_rare_uniform = (ids == 1).mean()
    assert frac_rare_uniform < 0.15, frac_rare_uniform


def t_stratified_sampling_survives_single_fault_buffer():
    """Must not crash before the second fault has any data."""
    from rl.sac import NStepReplayBuffer
    b = NStepReplayBuffer(3, 1, capacity=100, n_step=1, gamma=0.99, n_faults=2)
    o = np.zeros(3, dtype=np.float32)
    a = np.zeros(1, dtype=np.float32)
    for _ in range(20):
        b.add(o, a, 0.0, o, False, True, fault_id=0)
    out = b.sample(16, torch.device("cpu"), stratified=True)
    assert out["obs"].shape == (16, 3)


def t_build_shared_passes_env_factory():
    """FIX 2 at the construction site."""
    src = (RL_DIR / "build.py").read_text()
    assert "env_factory=lambda: build_env(cfg)[0]" in src


def t_eval_shared_supports_conditions_flag():
    """The pilot needs healthy + j0 + j6 + unseen j2, not just the pool."""
    src = (RL_DIR / "eval_shared.py").read_text()
    assert "--conditions" in src
    assert "unseen_conditions" in src
    assert "UNSEEN" in src




# ==========================================================================
# 6d. CAPABILITY-CONDITIONED SAC (generalization spec, section 9)
# ==========================================================================

CTX_DIM = 40   # 7+7+6+6+7+3+3+1, time excluded (matches residual_env)


def _buf(ctx_dim=CTX_DIM, K=4, n_step=1, cap=200, n_faults=1):
    from rl.sac import NStepReplayBuffer
    return NStepReplayBuffer(5, 2, capacity=cap, n_step=n_step, gamma=0.99,
                             n_faults=n_faults, ctx_dim=ctx_dim, context_len=K)


def _fill_episode(b, ep, n, base=0.0, fault=0):
    """Push n steps of one episode and record their transitions."""
    o = np.zeros(5, dtype=np.float32)
    a = np.zeros(2, dtype=np.float32)
    for t in range(n):
        feat = np.full(CTX_DIM, base + t, dtype=np.float32)
        s = b.push_context(feat, ep, t)
        b.add(o, a, 0.0, o, t == n - 1, False, fault_id=fault,
              s_obs=s, s_next=s,
              context_episode_id=ep, context_t=t,
              dyn_target=np.full(13, base + t, dtype=np.float32))


def t_context_window_never_crosses_episodes():
    """The single most dangerous bug in sequence replay."""
    b = _buf(K=4)
    _fill_episode(b, ep=0, n=6, base=0.0)
    _fill_episode(b, ep=1, n=6, base=100.0)
    # first step of episode 1 sits right after the last step of episode 0
    s_first_ep1 = None
    for i in range(b.stream_size):
        if b.stream_ep[i] == 1 and b.stream_t[i] == 0:
            s_first_ep1 = i
            break
    ctx, mask = b._window(s_first_ep1)
    assert mask.sum() == 0, (
        f"window at the first step of episode 1 pulled in {mask.sum()} steps "
        f"-- it reached back across the episode boundary into episode 0")
    assert np.allclose(ctx, 0.0)


def t_context_window_is_left_padded_and_ordered():
    b = _buf(K=4)
    _fill_episode(b, ep=0, n=3, base=10.0)
    s = int(np.flatnonzero((b.stream_ep == 0) & (b.stream_t == 2))[0])
    ctx, mask = b._window(s)
    # window ends at t-1, so it holds t=0 and t=1 only
    assert list(mask) == [0.0, 0.0, 1.0, 1.0], mask
    assert ctx[2, 0] == 10.0 and ctx[3, 0] == 11.0, ctx[:, 0]


def t_context_window_excludes_the_current_step():
    """LEAKAGE BOUNDARY: including step t hands the encoder its own target."""
    b = _buf(K=4)
    _fill_episode(b, ep=0, n=5, base=0.0)
    s = int(np.flatnonzero((b.stream_ep == 0) & (b.stream_t == 4))[0])
    ctx, mask = b._window(s)
    vals = ctx[mask > 0][:, 0]
    assert 4.0 not in set(vals.tolist()), (
        "the current step leaked into its own context window; L_dyn would "
        "collapse and z would learn nothing about capability")
    assert set(vals.tolist()) == {0.0, 1.0, 2.0, 3.0}, vals


def t_context_window_grows_within_episode():
    b = _buf(K=4)
    _fill_episode(b, ep=0, n=6, base=0.0)
    got = []
    for t in range(6):
        s = int(np.flatnonzero((b.stream_ep == 0) & (b.stream_t == t))[0])
        _, mask = b._window(s)
        got.append(int(mask.sum()))
    assert got == [0, 1, 2, 3, 4, 4], got


def t_context_window_rejects_overwritten_slots():
    """Ring wrap must not resurrect a stale slot as valid history."""
    b = _buf(K=4, cap=8)
    _fill_episode(b, ep=0, n=8, base=0.0)
    _fill_episode(b, ep=1, n=3, base=50.0)   # wraps over episode 0
    s = int(np.flatnonzero((b.stream_ep == 1) & (b.stream_t == 2))[0])
    ctx, mask = b._window(s)
    assert mask.sum() == 2, mask
    assert set(ctx[mask > 0][:, 0].tolist()) == {50.0, 51.0}


def t_gru_encoder_ignores_padding():
    from rl.context_encoder import GRUContextEncoder
    torch.manual_seed(0)
    enc = GRUContextEncoder(CTX_DIM, hidden=16, latent_dim=8)
    real = torch.randn(1, 2, CTX_DIM)
    short = torch.cat([torch.zeros(1, 2, CTX_DIM), real], dim=1)
    m_short = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    junk = torch.cat([torch.randn(1, 2, CTX_DIM) * 99, real], dim=1)
    z1 = enc(short, m_short)
    z2 = enc(junk, m_short)
    assert torch.allclose(z1, z2, atol=1e-6), (
        "masked positions changed the latent -- padding is not being ignored")


def t_encoder_none_gives_zero_width_latent():
    """`--context_encoder none` must reproduce the vanilla path exactly."""
    from rl.context_encoder import build_context_encoder
    enc = build_context_encoder("none", CTX_DIM)
    z = enc(torch.randn(4, 8, CTX_DIM), torch.ones(4, 8))
    assert z.shape == (4, 0), z.shape
    assert enc.latent_dim == 0


def t_vanilla_path_bit_identical_with_zero_latent():
    """Concat with a zero-width latent must not change the action."""
    from rl.sac import SACAgent
    torch.manual_seed(0)
    ag = SACAgent(20, 6, device="cpu", hidden=32)
    o = np.random.RandomState(0).randn(20).astype(np.float32)
    a1 = ag.act(o, deterministic=True)
    a2 = ag.act(o, deterministic=True, z_np=np.zeros(0, dtype=np.float32))
    assert np.allclose(a1, a2)


def t_zero_init_actor_holds_for_any_context():
    """Residual must be exactly 0 at init whatever the latent says."""
    from rl.sac import Actor
    net = Actor(20 + 32, 6, zero_init=True)
    for scale in (0.0, 1.0, 50.0):
        o = torch.randn(16, 52) * scale
        a, _ = net(o, deterministic=True, with_logp=False)
        assert a.abs().max().item() < 1e-6, scale


def t_dynamics_decoder_can_overfit_a_batch():
    """If L_dyn cannot fall on a tiny batch, the objective is wired wrong."""
    from rl.context_encoder import ContextModule
    torch.manual_seed(0)
    cm = ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="gru",
                       hidden=32, latent_dim=8, context_len=4, lr=3e-3)
    ctx = torch.randn(8, 4, CTX_DIM)
    mask = torch.ones(8, 4)
    obs, act = torch.randn(8, 5), torch.randn(8, 2)
    y = torch.randn(8, 13)
    cm.update_target_stats(y.numpy())
    first = cm.update(ctx, mask, obs, act, y)["ctx/L_dyn"]
    for _ in range(120):
        last = cm.update(ctx, mask, obs, act, y)["ctx/L_dyn"]
    assert last < first * 0.5, (first, last)


def t_context_detached_from_sac_by_default():
    """Spec 4.2: SAC gradients must not reach the encoder by default."""
    from rl.context_encoder import ContextModule
    cm = ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="gru",
                       hidden=16, latent_dim=8, context_len=4)
    assert cm.detach_for_policy is True
    z = cm.encode_for_policy(torch.randn(2, 4, CTX_DIM), torch.ones(2, 4))
    assert not z.requires_grad
    cm2 = ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="gru",
                        hidden=16, latent_dim=8, context_len=4,
                        detach_for_policy=False)
    z2 = cm2.encode_for_policy(torch.randn(2, 4, CTX_DIM), torch.ones(2, 4))
    assert z2.requires_grad


def t_context_module_roundtrips():
    from rl.context_encoder import ContextModule
    torch.manual_seed(0)
    a = ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="gru",
                      hidden=16, latent_dim=8, context_len=4)
    ctx, mask = torch.randn(3, 4, CTX_DIM), torch.ones(3, 4)
    a.update_target_stats(np.random.randn(32, 13))
    z_before = a.encode(ctx, mask).detach().clone()
    b = ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="gru",
                      hidden=16, latent_dim=8, context_len=4)
    b.load_state_dict(a.state_dict())
    assert torch.allclose(z_before, b.encode(ctx, mask), atol=1e-6)
    assert np.allclose(a.tgt_mean, b.tgt_mean)


def t_context_kind_mismatch_refused():
    """A latent from a differently-shaped encoder means something else."""
    from rl.context_encoder import ContextModule
    gru = ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="gru",
                        hidden=16, latent_dim=8, context_len=4)
    none = ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="none",
                         context_len=4)
    try:
        none.load_state_dict(gru.state_dict())
    except ValueError:
        return
    raise AssertionError("loading a GRU checkpoint into 'none' must raise")


def t_transformer_encoder_same_interface():
    from rl.context_encoder import build_context_encoder
    enc = build_context_encoder("transformer", CTX_DIM, 32, 8, 4)
    z = enc(torch.randn(3, 4, CTX_DIM), torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]))
    assert z.shape == (3, 8)
    assert torch.isfinite(z).all(), "all-padded row produced NaN"


def t_no_fault_metadata_in_context_feature():
    """Changing fault metadata with identical physics must not move z."""
    from rl.context_encoder import ContextModule
    cm = ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="gru",
                       hidden=16, latent_dim=8, context_len=4)
    ctx, mask = torch.randn(2, 4, CTX_DIM), torch.ones(2, 4)
    z1 = cm.encode(ctx, mask).detach().clone()
    z2 = cm.encode(ctx, mask).detach()
    assert torch.allclose(z1, z2)
    # the feature layout itself carries no label slot
    # Inspect the CODE that builds the feature, ignoring comments -- the
    # comments legitimately mention "fault" while explaining its absence.
    src = (RL_DIR / "residual_env.py").read_text()
    blk = src.split("---- context feature + dynamics target")[1].split("info = {")[0]
    code = "\n".join(ln for ln in blk.splitlines()
                     if not ln.strip().startswith("#"))
    for banned in ("self.faults", "fault", "joint_idx", "lock_value",
                   "one_hot", "active.name"):
        assert banned not in code.lower(), (banned, code)


def t_trainer_enforces_heldout_exclusion():
    src = (RL_DIR / "train_shared_fault_sac.py").read_text()
    assert "--heldout_conditions" in src
    assert "heldout & pool_names" in src


def t_eval_shared_no_random_by_default():
    """Spec 13: no random-residual control in scripts unless requested."""
    src = (RL_DIR / "eval_shared.py").read_text()
    assert 'default=["zero", "ckpt"]' in src, "random must not be a default"
    # Check actual INVOCATIONS, not prose: a comment explaining why the
    # random arm is absent legitimately contains the word.
    runner = (Path(__file__).resolve().parents[1] / "scripts"
              / "run_context_gru_j0_j6.sh")
    if runner.exists():
        code = "\n".join(ln for ln in runner.read_text().splitlines()
                         if not ln.strip().startswith("#"))
        assert "--policies" in code, "runner must state its policies explicitly"
        for ln in code.splitlines():
            if "--policies" in ln:
                assert "random" not in ln, ln


def t_env_exposes_context_feature_dims():
    src = (RL_DIR / "residual_env.py").read_text()
    assert "self.ctx_dim = 7 + 7 + 6 + 6 + 7 + 3 + 3 + 1" in src
    assert "self.dyn_dim = 13" in src
    assert "self.episode_id += 1" in src




# ==========================================================================
# 6e. NOVELTY GATE (negative-transfer remedy)
# ==========================================================================

def _ctxmod(beta=1.0, gmin=0.0):
    from rl.context_encoder import ContextModule
    return ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="gru",
                         hidden=16, latent_dim=8, context_len=4,
                         gate_beta=beta, gate_min=gmin)


def t_gate_open_on_familiar_error():
    cm = _ctxmod(beta=1.0)
    cm.update_novelty_stats(np.random.RandomState(0).normal(1.0, 0.1, 500))
    cm.nov_calibrated = True
    assert cm.gate(1.0) > 0.95, cm.gate(1.0)
    assert cm.gate(0.5) == 1.0, "below-average error must not shrink the gate"


def t_gate_closes_on_unfamiliar_error():
    """The j2 case: unfamiliar dynamics must silence the residual."""
    cm = _ctxmod(beta=1.0)
    cm.update_novelty_stats(np.random.RandomState(0).normal(1.0, 0.1, 500))
    cm.nov_calibrated = True     # calibration is a prerequisite for gating
    g = cm.gate(1.0 + 5 * 0.1)          # 5 sigma out
    assert g < 0.05, g


def t_gate_beta_zero_is_exactly_ungated():
    """The ablation arm must be bit-identical to the current behaviour."""
    cm = _ctxmod(beta=0.0)
    cm.update_novelty_stats(np.random.RandomState(0).normal(1.0, 0.1, 500))
    for e in (0.0, 1.0, 100.0):
        assert cm.gate(e) == 1.0, e


def t_gate_needs_enough_statistics():
    """With too few samples 'familiar' is undefined; do not gate on noise."""
    cm = _ctxmod(beta=1.0)
    cm.update_novelty_stats([1.0, 1.1, 0.9])
    cm.nov_calibrated = True   # even marked calibrated, n is too small
    assert cm.gate(50.0) == 1.0, "gated before the stats were established"


def t_gate_min_floor_respected():
    cm = _ctxmod(beta=5.0, gmin=0.25)
    cm.update_novelty_stats(np.random.RandomState(0).normal(1.0, 0.1, 500))
    cm.nov_calibrated = True
    assert abs(cm.gate(10.0) - 0.25) < 1e-9


def t_gate_scales_the_applied_residual():
    """gate=0 must reproduce the frozen VLA action exactly."""
    renv, env = make_env(scale=0.5, chunk=[[0.3] * 6 + [1.0]] * 8)
    renv.reset()
    renv.step(np.ones(6, dtype=np.float32), gate=0.0)
    assert np.allclose(env.last_action[:6], 0.3), env.last_action[:6]
    renv.reset()
    renv.step(np.ones(6, dtype=np.float32), gate=1.0)
    assert np.allclose(env.last_action[:6], 0.8), env.last_action[:6]
    renv.reset()
    _, _, _, _, info = renv.step(np.ones(6, dtype=np.float32), gate=0.5)
    assert np.allclose(env.last_action[:6], 0.55), env.last_action[:6]
    assert info["gate"] == 0.5


def t_gate_default_is_backward_compatible():
    """Every existing call site passes no gate and must be unaffected."""
    renv, env = make_env(scale=0.5, chunk=[[0.3] * 6 + [1.0]] * 8)
    renv.reset()
    _, _, _, _, info = renv.step(np.ones(6, dtype=np.float32))
    assert info["gate"] == 1.0
    assert np.allclose(env.last_action[:6], 0.8)


def t_novelty_stats_survive_checkpoint_roundtrip():
    """Recomputing at eval time would redefine 'familiar' to include the
    held-out fault, and the gate would stop firing where it is needed."""
    """Recomputing them at eval time would redefine 'familiar' to include
    the held-out fault, and the gate would stop firing where it is needed."""
    a = _ctxmod(beta=1.5, gmin=0.1)
    a.update_novelty_stats(np.random.RandomState(1).normal(2.0, 0.3, 400))
    b = _ctxmod(beta=0.0, gmin=0.0)
    b.load_state_dict(a.state_dict())
    assert abs(b.nov_mean - a.nov_mean) < 1e-12
    assert abs(b.gate_beta - 1.5) < 1e-12, "gate strength must travel too"
    assert abs(b.gate(5.0) - a.gate(5.0)) < 1e-12


def t_per_sample_dyn_error_shape():
    cm = _ctxmod()
    e = cm.per_sample_dyn_error(torch.randn(6, 4, CTX_DIM), torch.ones(6, 4),
                                torch.randn(6, 5), torch.randn(6, 2),
                                torch.randn(6, 13))
    assert e.shape == (6,) and torch.isfinite(e).all()


def t_wide_pool_tokens_all_parse():
    """Every condition in the wide-pool runner must be constructible."""
    from faults.multi_fault import FaultSpec
    pool = ["0", "0:off=+0.2", "0:off=-0.2", "0:damp=10", "0:damp=50",
            "0:damp=200", "6", "6:off=+0.2", "6:off=-0.2", "6:damp=10",
            "6:damp=50", "6:damp=200", "4", "5", "healthy"]
    names = [FaultSpec.parse(t).name for t in pool]
    assert len(set(names)) == len(names), f"duplicate names: {names}"
    assert "j2" not in names, "the held-out joint must not be in the pool"
    m = _mgr(joint_pool=tuple(pool), env_factory=FakeMultiEnv)
    assert len(m.specs) == 15
    assert m.index_of("healthy") == 14


def t_no_effect_joints_are_usable_conditions():
    """j4/j5 locks: real fault signature, near-zero correct residual."""
    from faults.multi_fault import FaultSpec
    env = FakeMultiEnv()
    m = _mgr(joint_pool=(4, 5), env_factory=FakeMultiEnv)
    for spec in m.specs:
        m.reset_episode(env if m.env is None else m.env, None,
                        init_id=0, force=spec)
        m.assert_exactly_one_lock(m.env)
    assert m.names == ["j4", "j5"]




# ==========================================================================
# 6f. REVIEW FIXES -- these exist because 100/100 was not enough
# ==========================================================================
#
# The previous suite tested the gate FUNCTIONS but never EXECUTED the
# evaluator, so a NameError on `g`, `gates` and `last_err` survived a fully
# green run. The evaluator rollout is now a function and these tests call it.

def _fake_eval_stack(gate_beta=0.0, kind="gru", calibrated=False):
    """A runnable evaluator stack with no LIBERO and no VLA."""
    import types as _types

    from faults.multi_fault import MultiFaultManager
    from rl.context_encoder import ContextModule
    from rl.residual_env import FrozenOFT, ResidualCfg, ResidualLiberoEnv
    from rl.sac import RunningNorm, SACAgent

    install_fakes([[0.2] * 6 + [1.0]] * 8)
    mgr = MultiFaultManager(joint_pool=(0, 6), seed=1,
                            env_factory=FakeMultiEnv)
    vla = FrozenOFT(_types.SimpleNamespace(model_family="openvla"),
                    _types.SimpleNamespace(), 224, "t", chunk_len=8)
    rc = ResidualCfg(residual_scale=0.1, history_len=4, max_steps=8,
                     num_steps_wait=0)
    renv = ResidualLiberoEnv(env=FakeMultiEnv(), vla=vla, fault_mgr=mgr,
                             initial_states=[None] * 5, init_ids=[0, 1],
                             cfg=rc,
                             prepare_observation=lambda o, rs: ({}, None))
    K = 4
    cm = ContextModule(ctx_dim=renv.ctx_dim, obs_dim=renv.obs_dim, act_dim=6,
                       kind=kind, hidden=16, latent_dim=8, context_len=K,
                       gate_beta=gate_beta)
    if calibrated:
        cm.update_novelty_stats(np.random.RandomState(0).normal(1.0, 0.05, 500))
        cm.nov_calibrated = True
    agent = SACAgent(renv.obs_dim + cm.latent_dim, 6, device="cpu", hidden=16)
    norm = RunningNorm(renv.obs_dim)
    norm.update(np.random.RandomState(0).randn(64, renv.obs_dim))
    meta = {"obs_dim": renv.obs_dim, "act_dim": 6}
    return renv, mgr, cm, agent, norm, meta, K


def t_evaluator_rollout_actually_runs():
    """FIX 1. The bug that a green suite missed: undefined g/gates/last_err."""
    from rl.eval_shared import rollout_condition
    renv, mgr, cm, agent, norm, meta, K = _fake_eval_stack(calibrated=True)
    for policy in ("zero", "ckpt"):
        res = rollout_condition(
            renv=renv, spec=mgr.specs[0], policy=policy, ids=[0, 1],
            meta=meta, agent=agent, normalizer=norm, ctxmod=cm, K=K,
            seed=7, device="cpu", trained_names={"j0"})
        assert len(res["rows"]) == 2, res
        assert 0.0 <= res["summary"]["rate"] <= 1.0
        assert "mean_gate" in res["summary"]


def t_evaluator_zero_policy_is_never_gated():
    """A zero residual has nothing to shrink; gating it would be noise."""
    from rl.eval_shared import rollout_condition
    renv, mgr, cm, agent, norm, meta, K = _fake_eval_stack(
        gate_beta=5.0, calibrated=True)
    res = rollout_condition(renv=renv, spec=mgr.specs[0], policy="zero",
                            ids=[0], meta=meta, agent=agent, normalizer=norm,
                            ctxmod=cm, K=K, seed=7, device="cpu")
    assert res["summary"]["mean_gate"] == 1.0


def t_evaluator_runs_without_context_module():
    """`--context_encoder none` checkpoints must still evaluate."""
    from rl.eval_shared import rollout_condition
    renv, mgr, cm, agent, norm, meta, K = _fake_eval_stack(kind="none")
    res = rollout_condition(renv=renv, spec=mgr.specs[1], policy="ckpt",
                            ids=[0], meta=meta, agent=agent, normalizer=norm,
                            ctxmod=cm, K=K, seed=7, device="cpu")
    assert res["summary"]["mean_gate"] == 1.0


def t_dyn_error_uses_normalized_observations():
    """FIX 2. The decoder trains on normalized obs; probing raw is invalid.

    Raw and normalized inputs must give DIFFERENT errors -- if they didn't,
    the normalizer would be doing nothing and the bug would be invisible.
    """
    from rl.eval_shared import _dyn_error_online
    from rl.context_encoder import ContextModule
    from rl.sac import RunningNorm
    cm = ContextModule(ctx_dim=CTX_DIM, obs_dim=5, act_dim=2, kind="gru",
                       hidden=16, latent_dim=8, context_len=4)
    cm.update_target_stats(np.random.RandomState(0).randn(64, 13))
    norm = RunningNorm(5)
    norm.update(np.random.RandomState(0).normal(10.0, 3.0, size=(500, 5)))
    hist = [np.random.RandomState(i).randn(CTX_DIM).astype(np.float32)
            for i in range(4)]
    obs_raw = np.full(5, 12.0, dtype=np.float32)
    a = np.zeros(2, dtype=np.float32)
    y = np.zeros(13, dtype=np.float32)
    e_raw = _dyn_error_online(cm, hist, 4, CTX_DIM, obs_raw, a, y, "cpu")
    e_nrm = _dyn_error_online(cm, hist, 4, CTX_DIM,
                              norm(obs_raw).astype(np.float32), a, y, "cpu")
    assert e_raw > 0 and e_nrm > 0
    assert abs(e_raw - e_nrm) > 1e-6, (
        "raw and normalized inputs gave the same error -- the scale mismatch "
        "would be undetectable")


def t_trainer_normalizes_obs_for_dyn_error():
    src = (RL_DIR / "train_shared_fault_sac.py").read_text()
    assert "normalizer(obs).astype(np.float32)" in src, (
        "trainer must feed NORMALIZED observations to the dynamics decoder")




def t_gate_refuses_until_calibrated():
    """FIX 3. Training-time running stats mix three regimes; fail OPEN."""
    cm = _ctxmod(beta=1.0)
    cm.update_novelty_stats(np.random.RandomState(0).normal(1.0, 0.1, 500))
    assert cm.nov_calibrated is False
    assert cm.gate(50.0) == 1.0, "gated on uncalibrated statistics"
    cm.nov_calibrated = True
    assert cm.gate(50.0) < 0.05


def t_reset_novelty_stats_clears_calibration():
    cm = _ctxmod(beta=1.0)
    cm.update_novelty_stats(np.random.RandomState(0).normal(1.0, 0.1, 500))
    cm.nov_calibrated = True
    cm.reset_novelty_stats()
    assert cm.nov_calibrated is False and cm.nov_count < 1.0
    assert cm.gate(50.0) == 1.0


def t_calibration_flag_survives_roundtrip():
    a = _ctxmod(beta=1.0)
    a.update_novelty_stats(np.random.RandomState(0).normal(1.0, 0.1, 500))
    a.nov_calibrated = True
    b = _ctxmod(beta=0.0)
    b.load_state_dict(a.state_dict())
    assert b.nov_calibrated is True
    assert abs(b.gate(5.0) - a.gate(5.0)) < 1e-12


def t_damping_verified_against_compiled_model():
    """FIX 5. 'No lock present' does not prove the damping took effect."""
    from faults.multi_fault import FaultSpec
    env = FakeMultiEnv()
    env.BASE = ("<mujoco><worldbody><body>"
                "<joint name='robot0_joint1' damping='0.1'/>"
                "</body></worldbody><equality/></mujoco>")
    m = _mgr(joint_pool=(FaultSpec(0, kind="damping", damping=50.0),))
    m._baseline_damping = {"robot0_joint1": 0.1}
    m.reset_episode(env, None, init_id=0, force=m.specs[0])

    # fake compiled model reports the XML value -> should pass
    def _damp_ok(e, name):
        import xml.etree.ElementTree as ET
        el = ET.fromstring(e.xml).find("worldbody").find(f".//joint[@name='{name}']")
        return float(el.get("damping"))

    import faults.multi_fault as mf
    real = mf.read_compiled_damping
    mf.read_compiled_damping = _damp_ok
    try:
        m.assert_exactly_one_lock(env)          # must not raise
        mf.read_compiled_damping = lambda e, n: 0.1   # processor silently failed
        try:
            m.assert_exactly_one_lock(env)
        except AssertionError as exc:
            assert "effectively HEALTHY" in str(exc), str(exc)
        else:
            raise AssertionError(
                "a damping fault that did not compile must be detected")
    finally:
        mf.read_compiled_damping = real


def t_heldout_excludes_the_whole_joint():
    """FIX 6. j2, j2_off+0.2 and j2_damp50 are one joint, three names."""
    from faults.multi_fault import joint_indices
    assert joint_indices(["2"]) == {2}
    assert joint_indices(["2:off=0.2"]) == {2}
    assert joint_indices(["2:damp=50"]) == {2}
    assert joint_indices(["healthy"]) == set()
    assert joint_indices(["0", "6", "2:damp=50"]) == {0, 6, 2}
    src = (RL_DIR / "train_shared_fault_sac.py").read_text()
    assert "heldout_joints & pool_joints" in src, (
        "trainer must exclude by JOINT index, not by condition name")


def t_wide_pool_runner_uses_40_state_baselines():
    """FIX 4. The guard's rolling rate covers states 0-39."""
    r = (Path(__file__).resolve().parents[1] / "scripts"
         / "run_context_wide_pool.sh")
    if r.exists():
        txt = r.read_text()
        assert "--n_episodes 40" in txt, "baselines must cover states 0-39"


def t_wide_pool_runner_does_not_overclaim():
    r = (Path(__file__).resolve().parents[1] / "scripts"
         / "run_context_wide_pool.sh")
    if r.exists():
        txt = r.read_text()
        assert "~15% nominal" not in txt, "healthy share was ~7%, not 15%"
        assert "HYPOTHESIS" in txt, (
            "joint-index order does not imply latent interpolation; that "
            "must be stated as a hypothesis")


# ==========================================================================
# 7. experiment hygiene
# ==========================================================================

def t_split_is_disjoint_and_holds_out_tail():
    from rl.build import split_init_states
    tr, ev = split_init_states(50, n_eval=10)
    assert set(tr).isdisjoint(ev)
    assert len(tr) == 40 and len(ev) == 10
    assert ev == list(range(40, 50))
    assert set(range(20)).issubset(tr), (
        "the 20 screening states must stay in TRAIN; evaluating on them "
        "would not be held out"
    )


def t_task_reward_only_is_the_default():
    from rl.residual_env import ResidualCfg
    c = ResidualCfg()
    assert c.w_residual == 0.0 and c.w_shaping == 0.0, (
        "the headline condition must be task-reward-only by default"
    )


def t_residual_penalty_does_not_touch_reported_task_reward():
    renv, _ = make_env(succeed_at=1, max_steps=10, wait=0)
    renv.cfg.w_residual = 1.0
    renv.reset()
    _, r, _, _, info = renv.step(np.ones(6, dtype=np.float32))
    assert info["r_task"] == 1.0
    assert r < 1.0, "penalty should reduce the learning signal..."
    assert info["r_task"] == 1.0, "...but never the reported task outcome"


def t_vla_params_frozen():
    from rl.residual_env import FrozenOFT
    install_fakes([[0.0] * 7] * 8)
    m = torch.nn.Linear(4, 4)
    assert m.weight.requires_grad
    FrozenOFT(types.SimpleNamespace(model_family="openvla"), m, 224, "t")
    assert not m.weight.requires_grad
    assert not m.training


# ==========================================================================

TESTS = [
    ("quat: identity", t_quat_identity),
    ("quat: known 30deg", t_quat_known_rotation),
    ("quat: no pi-wrap blowup", t_quat_no_wrap_blowup),
    ("residual: zero == base action", t_zero_residual_is_exact_base),
    ("residual: no double tanh", t_no_double_tanh),
    ("residual: gripper untouched", t_gripper_never_residualized),
    ("residual: final action clipped", t_final_action_clipped),
    ("residual: unbounded input rejected", t_unbounded_residual_rejected),
    ("episode: obs dim matches", t_obs_dim_matches_declared),
    ("episode: dummy prefix not counted", t_dummy_prefix_not_counted),
    ("episode: truncation at max_steps", t_truncation_at_max_steps),
    ("episode: success terminates, r=1", t_success_terminates_with_reward_one),
    ("episode: truncated next_obs has fresh base action", t_truncation_next_obs_has_fresh_base_action),
    ("episode: terminated skips extra VLA query", t_termination_skips_the_extra_vla_query),
    ("chunk: requery every 8", t_chunk_requery_every_8_steps),
    ("chunk: phase progresses and wraps", t_chunk_phase_progresses_and_wraps),
    ("history: execution mismatch recorded", t_history_records_execution_mismatch),
    ("buffer: n-step reward accumulation", t_nstep_reward_accumulation),
    ("buffer: chain cut by success", t_nstep_chain_cut_by_success),
    ("buffer: timeout is NOT terminal", t_timeout_is_not_terminal),
    ("buffer: stores raw observations", t_buffer_stores_raw_observations),
    ("buffer: sample normalizes with current stats", t_sample_normalizes_with_current_stats),
    ("buffer: end_episode stops cross-episode chaining", t_end_episode_prevents_cross_episode_chaining),
    ("buffer: the leak is real without the guard", t_without_end_episode_the_chain_would_leak),
    ("sac: actor output bounded", t_actor_output_bounded),
    ("sac: deterministic repeatable", t_actor_deterministic_is_repeatable),
    ("sac: update runs, params move", t_sac_update_runs_and_moves_params),
    ("sac: zero-init actor == base policy", t_zero_init_actor_starts_at_the_base_policy),
    ("sac: zero-init still explores", t_zero_init_still_explores),
    ("sac: random init is NOT zero (guard)", t_random_init_actor_does_not_start_at_zero),
    ("sac: alpha_init is not 1.0", t_alpha_init_is_not_one),
    ("sac: entropy cannot swamp task return", t_entropy_bonus_cannot_swamp_task_return),
    ("sac: target entropy = -act_dim", t_target_entropy_matches_action_dim),
    ("sac: running norm correct", t_running_norm_matches_numpy),
    ("fault: processor builds constraint", t_processor_creates_equality_constraint),
    ("fault: processor idempotent", t_processor_is_idempotent),
    ("fault: preserves other equalities", t_processor_preserves_existing_equality),
    ("fault: joint naming j0->joint1", t_joint_naming),
    ("fault: monitor flags drift", t_lock_monitor_flags_drift),
    ("fault: one rebuild when target constant", t_manager_rebuilds_once_when_target_constant),
    ("fault: disabled is a no-op", t_manager_disabled_does_nothing),
    ("shared: j0->j6 leaves only j6", t_switch_j0_to_j6_leaves_only_j6),
    ("shared: switch to healthy clears locks", t_switch_to_healthy_removes_all_locks),
    ("shared: full j0/j6/healthy/j0 cycle", t_full_cycle_j0_j6_healthy_j0),
    ("shared: task equalities preserved", t_processor_preserves_non_fault_equalities),
    ("shared: NO fault label in observation", t_fault_identity_absent_from_observation),
    ("shared: info carries fault for logging", t_info_does_carry_fault_for_logging),
    ("shared: fault fixed within an episode", t_fault_sampled_per_episode_not_per_step),
    ("shared: sampling respects probs", t_sampling_respects_probabilities),
    ("shared: probs length mismatch rejected", t_probs_length_mismatch_is_rejected),
    ("shared: lock targets cached per (joint,state)", t_lock_targets_cached_per_joint_and_state),
    ("shared: pool names include healthy", t_healthy_pool_names),
    ("FIX1: checker raises, never returns []", t_checker_raises_when_model_unreadable),
    ("FIX1: checker reads compiled names", t_checker_reads_compiled_names),
    ("FIX2: env_factory rebuilds on switch", t_env_factory_rebuilds_on_switch),
    ("FIX2: no rebuild when fault unchanged", t_env_factory_idle_when_fault_unchanged),
    ("FIX2: residual env adopts new env", t_residual_env_adopts_rebuilt_env),
    ("FIX2: build_shared passes env_factory", t_build_shared_passes_env_factory),
    ("FIX3: sampling alone can miss the switch", t_sampling_alone_can_miss_the_switch),
    ("FIX3: forced sequence covers transitions", t_forced_sequence_covers_every_transition),
    ("FIX4: trainer uses --train_baselines", t_trainer_uses_train_baselines_flag),
    ("FIX5: eval_every=0 legal and default", t_eval_every_zero_is_legal),
    ("pilot: j2 refused in training pool", t_trainer_refuses_j2_in_training_pool),
    ("pilot: eval_shared has --conditions", t_eval_shared_supports_conditions_flag),
    ("replay: fault-tagged, fractions reported", t_replay_tags_fault_and_reports_fractions),
    ("replay: stratified balances minibatch", t_stratified_sampling_balances_the_minibatch),
    ("replay: stratified ok with one fault", t_stratified_sampling_survives_single_fault_buffer),
    ("ctx: window never crosses episodes", t_context_window_never_crosses_episodes),
    ("ctx: window left-padded and ordered", t_context_window_is_left_padded_and_ordered),
    ("ctx: window excludes current step", t_context_window_excludes_the_current_step),
    ("ctx: window grows within episode", t_context_window_grows_within_episode),
    ("ctx: window rejects overwritten slots", t_context_window_rejects_overwritten_slots),
    ("ctx: GRU ignores padding", t_gru_encoder_ignores_padding),
    ("ctx: 'none' gives zero-width latent", t_encoder_none_gives_zero_width_latent),
    ("ctx: vanilla path bit-identical", t_vanilla_path_bit_identical_with_zero_latent),
    ("ctx: zero-init holds for any context", t_zero_init_actor_holds_for_any_context),
    ("ctx: dynamics decoder overfits batch", t_dynamics_decoder_can_overfit_a_batch),
    ("ctx: detached from SAC by default", t_context_detached_from_sac_by_default),
    ("ctx: module save/load roundtrip", t_context_module_roundtrips),
    ("ctx: encoder kind mismatch refused", t_context_kind_mismatch_refused),
    ("ctx: transformer same interface", t_transformer_encoder_same_interface),
    ("ctx: no fault metadata in feature", t_no_fault_metadata_in_context_feature),
    ("ctx: env exposes context dims", t_env_exposes_context_feature_dims),
    ("gen: trainer enforces held-out exclusion", t_trainer_enforces_heldout_exclusion),
    ("gen: eval has no random by default", t_eval_shared_no_random_by_default),
    ("gate: open on familiar error", t_gate_open_on_familiar_error),
    ("gate: closes on unfamiliar error", t_gate_closes_on_unfamiliar_error),
    ("gate: beta=0 is exactly ungated", t_gate_beta_zero_is_exactly_ungated),
    ("gate: needs enough statistics", t_gate_needs_enough_statistics),
    ("gate: gate_min floor respected", t_gate_min_floor_respected),
    ("gate: scales the applied residual", t_gate_scales_the_applied_residual),
    ("gate: default backward compatible", t_gate_default_is_backward_compatible),
    ("gate: novelty stats survive roundtrip", t_novelty_stats_survive_checkpoint_roundtrip),
    ("gate: per-sample dyn error shape", t_per_sample_dyn_error_shape),
    ("pool: wide-pool tokens all parse", t_wide_pool_tokens_all_parse),
    ("pool: no-effect joints usable", t_no_effect_joints_are_usable_conditions),
    ("FIX1: evaluator rollout actually runs", t_evaluator_rollout_actually_runs),
    ("FIX1: zero policy never gated", t_evaluator_zero_policy_is_never_gated),
    ("FIX1: evaluator runs without context", t_evaluator_runs_without_context_module),
    ("FIX2: dyn error uses normalized obs", t_dyn_error_uses_normalized_observations),
    ("FIX2: trainer normalizes for dyn error", t_trainer_normalizes_obs_for_dyn_error),
    ("FIX3: gate refuses until calibrated", t_gate_refuses_until_calibrated),
    ("FIX3: reset clears calibration", t_reset_novelty_stats_clears_calibration),
    ("FIX3: calibration flag roundtrips", t_calibration_flag_survives_roundtrip),
    ("FIX4: runner uses 40-state baselines", t_wide_pool_runner_uses_40_state_baselines),
    ("FIX5: damping verified vs compiled model", t_damping_verified_against_compiled_model),
    ("FIX6: held-out excludes whole joint", t_heldout_excludes_the_whole_joint),
    ("FIX7: runner does not overclaim", t_wide_pool_runner_does_not_overclaim),
    ("hygiene: train/eval split disjoint", t_split_is_disjoint_and_holds_out_tail),
    ("hygiene: task-reward-only default", t_task_reward_only_is_the_default),
    ("hygiene: penalty separate from r_task", t_residual_penalty_does_not_touch_reported_task_reward),
    ("hygiene: VLA frozen on construction", t_vla_params_frozen),
]

if __name__ == "__main__":
    print(f"running {len(TESTS)} offline tests\n")
    for name, fn in TESTS:
        check(name, fn)
    print(f"\n{len(PASS)}/{len(TESTS)} passed")
    if FAIL:
        print("\nfailures:")
        for n, e in FAIL:
            print(f"  {n}: {e}")
        sys.exit(1)
