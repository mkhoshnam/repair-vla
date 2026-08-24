"""
sac.py -- actor, twin critic, and an n-step replay buffer.

Why SAC and not PPO: one env step costs an OSC solve, a MuJoCo step, two
256x256 renders, and (every 8th step) a 7B-parameter forward pass. PPO
throws every transition away after one update and needs 1e6-1e7 steps. SAC
reuses the buffer and gets there in ~1e5-3e5. At the throughput this loop
can reach, that is the difference between one overnight run and one month.

Why n-step returns (default n = 3): the reward is sparse and terminal, and
episodes run to 220 steps. With 1-step backups and gamma = 0.99 the success
signal has to propagate through 200+ Bellman updates before it reaches the
early-episode states where the compensating motion would have to start.
n-step divides that latency by n at the cost of a small off-policy bias,
which is the right trade here.

The one thing that is easy to get wrong and fatal: TIMEOUT IS NOT TERMINAL.
Hitting `max_steps` says nothing about the value of the state, so the target
must still bootstrap there. Only task success is a real terminal. Storing
`done = terminated or truncated` -- the default in most tutorial code --
teaches the critic that every state 220 steps in is worth zero, and the
policy learns to do nothing. `Transition.terminated` is the success flag
alone; `truncated` is carried separately and used only to cut n-step chains.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh-squashed diagonal Gaussian. Output is bounded to [-1, 1].

    ZERO-INITIALIZED OUTPUT HEAD. This is the defining property of a residual
    policy and it was missing. With a randomly initialized `mu`, step 0 of
    training emits an arbitrary constant correction on top of the VLA -- on
    j6 that was already 57% of maximum residual magnitude before a single
    gradient step. The learner then had to discover "do nothing" from sparse
    reward, starting from a policy that was actively corrupting a base
    controller that was right 52.5% of the time.

    With `zero_init`, the deterministic policy at step 0 is exactly delta = 0,
    i.e. the frozen VLA. RL can then only move away from the baseline if the
    task reward pays for it. On a fault like j0 where the base policy is
    broken, this costs nothing -- the residual grows immediately because
    anything is better than 20%. On a fault like j6 where the base policy is
    mostly right, it is the difference between improving and destroying.

    `log_std_init` sets the initial exploration width. The mean starts at
    zero but sampling still explores, so the buffer still sees varied
    residuals during warm-up.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 zero_init: bool = True, log_std_init: float = -1.0):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden])
        self.act = nn.ReLU()
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.act_dim = act_dim

        if zero_init:
            nn.init.zeros_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
            nn.init.zeros_(self.log_std.weight)
            nn.init.constant_(self.log_std.bias, log_std_init)

    def forward(self, obs, deterministic: bool = False, with_logp: bool = True):
        h = self.act(self.net(obs))
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            u = mu
            logp = None
        else:
            dist = torch.distributions.Normal(mu, std)
            u = dist.rsample()
            if with_logp:
                logp = dist.log_prob(u).sum(-1)
                # tanh change-of-variables, numerically stable form
                logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
            else:
                logp = None
        return torch.tanh(u), logp


class Critic(nn.Module):
    """Twin Q. Clipped double-Q is what keeps sparse-reward SAC from
    exploding: with 0.99^220 discounting, a single overestimated Q on a
    rarely visited state propagates fast."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, hidden, hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class RunningNorm:
    """Welford normalizer for observations.

    Not optional here: the observation concatenates joint velocities (O(1)),
    end-effector deltas (O(1e-3)) and normalized action values (O(1)). Raw,
    the network sees the delta features as noise around zero -- and those
    deltas are precisely the fault signal.
    """

    def __init__(self, dim: int, eps: float = 1e-4):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.atleast_2d(x).astype(np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class NStepReplayBuffer:
    """Flat buffer fed through an n-step staging queue.

    `terminated` means task success. `truncated` means the horizon ran out;
    it ends the n-step chain but the stored transition still bootstraps.

    OBSERVATIONS ARE STORED RAW. Normalization happens in `sample()` against
    the *current* statistics. Storing pre-normalized observations looks
    equivalent and is not: `RunningNorm` keeps moving, so a transition added
    at step 1k was normalized against a different mean and variance than one
    added at step 90k, and a single minibatch then mixes representations that
    disagree about what "zero" means. The critic sees a non-stationary input
    distribution that has nothing to do with the policy changing, and the
    drift is largest early on, exactly when the run is most fragile.
    """

    def __init__(self, obs_dim, act_dim, capacity=400_000, n_step=3, gamma=0.99,
                 n_faults: int = 1, ctx_dim: int = 0, context_len: int = 16):
        # ---- CONTEXT STREAM ------------------------------------------------
        # A parallel ring of per-step context features, one entry per ENV
        # STEP, tagged with episode id and within-episode timestep. Windows
        # are rebuilt from it at sample time.
        #
        # Why a separate stream rather than storing a window per transition:
        # with n-step returns the transition at index i has its next_obs k
        # steps later, so the target critic needs a window ending at a
        # DIFFERENT step than the one the online critic needs. Materializing
        # both windows per transition would cost K x ctx_dim x 2 floats per
        # slot; the stream costs one row per step and both windows are
        # slices of it.
        #
        # A window is valid only where episode id matches AND the timestep
        # decreases by exactly one. That single check rejects episode
        # boundaries, fault switches (which end an episode), env rebuilds,
        # and ring-buffer overwrites at once -- a stale slot cannot satisfy
        # it, because its timestep will not line up.
        self.ctx_dim = int(ctx_dim)
        self.context_len = int(context_len)
        self.n_step, self.gamma = int(n_step), float(gamma)

        # Context stream gets ahead of n-step replay by a few steps.
        # Give it headroom so live replay transitions keep their history.
        self.stream_capacity = capacity + self.context_len + self.n_step + 8

        self.stream_ctx = np.zeros(
            (self.stream_capacity, max(1, self.ctx_dim)),
            dtype=np.float32,
        )
        self.stream_ep = np.full(
            self.stream_capacity, -1, dtype=np.int64
        )
        self.stream_t = np.full(
            self.stream_capacity, -1, dtype=np.int64
        )
        self.stream_ptr = 0
        self.stream_size = 0

        # Actual WRITTEN context anchors.
        self.s_obs = np.zeros(capacity, dtype=np.int64)
        self.s_next = np.zeros(capacity, dtype=np.int64)

        # Expected metadata prevents an overwritten stream slot from
        # masquerading as valid history for an old replay transition.
        self.s_obs_ep = np.full(capacity, -1, dtype=np.int64)
        self.s_obs_t = np.full(capacity, -1, dtype=np.int64)
        self.s_next_ep = np.full(capacity, -1, dtype=np.int64)
        self.s_next_t = np.full(capacity, -1, dtype=np.int64)
        # realized motion at the transition's own step -- the L_dyn target
        self.dyn_target = np.zeros((capacity, 13), dtype=np.float32)

        # Fault id per transition. Bookkeeping and STRATIFIED SAMPLING only --
        # it is never concatenated into the observation. Needed because
        # 50/50 EPISODE exposure does not give 50/50 TRANSITIONS: a failing
        # j0 episode runs the full 220 steps while a successful j6 episode
        # may end in 60, so j0 can contribute several times the data and
        # quietly dominate every minibatch.
        self.fault_id = np.zeros(capacity, dtype=np.int64)
        self.n_faults = int(n_faults)
        self._stage_fault = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.term = np.zeros(capacity, dtype=np.float32)
        self.disc = np.zeros(capacity, dtype=np.float32)  # gamma^k actually used
        self.ptr, self.size, self.capacity = 0, 0, capacity

        self._stage: list = []
        self._stage_s_obs = 0
        self._stage_s_next = 0
        self._stage_s_obs_ep = -1
        self._stage_s_obs_t = -1
        self._stage_s_next_ep = -1
        self._stage_s_next_t = -1
        self._stage_dyn = np.zeros(13, dtype=np.float32)

    # ---------------- context stream -----------------------------------

    def push_context(self, ctx_feat, episode_id: int, t_in_ep: int) -> int:
        """Append one already-executed env step and return its stream slot."""
        i = self.stream_ptr

        if self.ctx_dim:
            self.stream_ctx[i] = ctx_feat

        self.stream_ep[i] = int(episode_id)
        self.stream_t[i] = int(t_in_ep)

        self.stream_ptr = (self.stream_ptr + 1) % self.stream_capacity
        self.stream_size = min(
            self.stream_size + 1, self.stream_capacity
        )
        return i

    def _anchor_valid(self, s_idx: int, exp_ep: int, exp_t: int) -> bool:
        """Check that this physical stream slot still contains this step."""
        if self.ctx_dim == 0 or exp_ep < 0 or exp_t < 0:
            return False

        s_idx = int(s_idx) % self.stream_capacity

        return (
            self.stream_ep[s_idx] == int(exp_ep)
            and self.stream_t[s_idx] == int(exp_t)
        )

    def _window_before(self, s_idx: int, exp_ep: int, exp_t: int):
        """Context for obs_t: excludes the outcome produced during step t."""
        K, C = self.context_len, self.ctx_dim

        ctx = np.zeros(
            (K, max(1, C)),
            dtype=np.float32,
        )
        mask = np.zeros(K, dtype=np.float32)

        if C == 0 or not self._anchor_valid(s_idx, exp_ep, exp_t):
            return ctx, mask

        for j in range(1, K + 1):
            idx = (int(s_idx) - j) % self.stream_capacity

            if (
                self.stream_ep[idx] != exp_ep
                or self.stream_t[idx] != exp_t - j
            ):
                break

            ctx[K - j] = self.stream_ctx[idx]
            mask[K - j] = 1.0

        return ctx, mask

    def _window_through(self, s_idx: int, exp_ep: int, exp_t: int):
        """Context for next_obs: includes the final executed step."""
        K, C = self.context_len, self.ctx_dim

        ctx = np.zeros(
            (K, max(1, C)),
            dtype=np.float32,
        )
        mask = np.zeros(K, dtype=np.float32)

        if C == 0 or not self._anchor_valid(s_idx, exp_ep, exp_t):
            return ctx, mask

        for j in range(K):
            idx = (int(s_idx) - j) % self.stream_capacity

            if (
                self.stream_ep[idx] != exp_ep
                or self.stream_t[idx] != exp_t - j
            ):
                break

            ctx[K - 1 - j] = self.stream_ctx[idx]
            mask[K - 1 - j] = 1.0

        return ctx, mask

    def _window(self, s_idx: int):
        """Backward-compatible helper: obs semantics, exclusive."""
        s_idx = int(s_idx) % self.stream_capacity

        return self._window_before(
            s_idx,
            int(self.stream_ep[s_idx]),
            int(self.stream_t[s_idx]),
        )

    def _windows(self, idx_array, which: str):
        K = self.context_len
        C = max(1, self.ctx_dim)

        ctx = np.zeros(
            (len(idx_array), K, C),
            dtype=np.float32,
        )
        mask = np.zeros(
            (len(idx_array), K),
            dtype=np.float32,
        )

        if which == "obs":
            for n, i in enumerate(idx_array):
                ii = int(i)

                ctx[n], mask[n] = self._window_before(
                    int(self.s_obs[ii]),
                    int(self.s_obs_ep[ii]),
                    int(self.s_obs_t[ii]),
                )

        elif which == "next":
            for n, i in enumerate(idx_array):
                ii = int(i)

                ctx[n], mask[n] = self._window_through(
                    int(self.s_next[ii]),
                    int(self.s_next_ep[ii]),
                    int(self.s_next_t[ii]),
                )

        else:
            raise ValueError(
                f"which must be 'obs' or 'next', got {which!r}"
            )

        return ctx, mask

    # -------------------------------------------------------------------

    def _push(self, o, a, r, no, term, disc):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = o, a, r
        self.next_obs[i], self.term[i], self.disc[i] = no, term, disc
        self.fault_id[i] = self._stage_fault
        self.s_obs[i] = self._stage_s_obs
        self.s_next[i] = self._stage_s_next
        self.s_obs_ep[i] = self._stage_s_obs_ep
        self.s_obs_t[i] = self._stage_s_obs_t
        self.s_next_ep[i] = self._stage_s_next_ep
        self.s_next_t[i] = self._stage_s_next_t
        self.dyn_target[i] = self._stage_dyn
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _flush_one(self):
        """Collapse oldest staged transition into one n-step transition."""
        first = self._stage[0]

        o, a = first[0], first[1]

        # Context anchor for original obs.
        self._stage_s_obs = first[5]
        self._stage_s_obs_ep = first[6]
        self._stage_s_obs_t = first[7]

        self._stage_dyn = first[8]

        r = 0.0
        disc = 1.0
        term = 0.0
        no = first[3]

        # next-state anchor begins at first executed step and advances
        # to the final executed step in the n-step chain.
        self._stage_s_next = first[5]
        self._stage_s_next_ep = first[6]
        self._stage_s_next_t = first[7]

        for st in self._stage:
            r += disc * st[2]
            disc *= self.gamma

            no = st[3]
            term = st[4]

            self._stage_s_next = st[5]
            self._stage_s_next_ep = st[6]
            self._stage_s_next_t = st[7]

            if st[4] > 0.5:
                break

        self._push(o, a, r, no, term, disc)
        self._stage.pop(0)

    def end_episode(self):
        """Flush the staging queue. Call on ANY episode boundary.

        `add()` already does this when it sees terminated/truncated. This
        exists for the case where an episode is abandoned without a final
        transition -- most importantly when a periodic held-out evaluation
        interrupts training. Without it, the leftover staged transitions of
        the abandoned episode get n-step-chained to the first transitions of
        the next one, fabricating rewards and next-states across an episode
        boundary that never existed.
        """
        while self._stage:
            self._flush_one()

    def add(
        self,
        obs,
        act,
        rew,
        next_obs,
        terminated,
        truncated,
        fault_id=0,
        s_obs=0,
        s_next=None,
        dyn_target=None,
        context_episode_id=-1,
        context_t=-1,
    ):
        self._stage_fault = int(fault_id)

        # Both anchors refer to THIS already-written physical context step.
        # next_obs gets inclusive semantics later.
        if s_next is None:
            s_next = s_obs

        if self.ctx_dim and int(s_next) != int(s_obs):
            raise ValueError(
                "Do not pass a future s_next sentinel. "
                "Use the current written context index for both anchors."
            )

        self._stage.append(
            (
                obs,
                act,
                float(rew),
                next_obs,
                float(terminated),
                int(s_obs),
                int(context_episode_id),
                int(context_t),
                np.zeros(13, dtype=np.float32)
                if dyn_target is None
                else np.asarray(dyn_target, dtype=np.float32),
            )
        )

        if len(self._stage) >= self.n_step:
            self._flush_one()

        if terminated or truncated:
            self.end_episode()

    def fault_fractions(self) -> dict:
        """Share of stored transitions per fault id. Log this every interval."""
        if self.size == 0:
            return {}
        ids = self.fault_id[: self.size]
        return {int(k): float((ids == k).mean()) for k in range(self.n_faults)}

    def sample_at(self, idx, device, obs_norm=None):
        """Assemble a replay batch from explicit transition indices.

        This keeps all context-window/raw-observation semantics identical to
        `sample()` while allowing higher-level samplers to choose strata.
        """
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError("sample_at requires at least one index")
        if np.any(idx < 0) or np.any(idx >= self.size):
            raise IndexError("sample_at index outside populated replay range")

        o, no = self.obs[idx], self.next_obs[idx]
        if obs_norm is not None:
            o = obs_norm(o).astype(np.float32)
            no = obs_norm(no).astype(np.float32)
        t = lambda x: torch.as_tensor(x, device=device)
        batch = {
            "obs": t(o), "act": t(self.act[idx]), "rew": t(self.rew[idx]),
            "next_obs": t(no), "term": t(self.term[idx]),
            "disc": t(self.disc[idx]), "idx": idx,
            # Raw copies are needed by the joint-factorized kinematic
            # auxiliary losses to recover q/qdot/J in physical units. The
            # legacy SAC path ignores these keys, so this is backward-safe.
            "raw_obs": t(self.obs[idx]),
            "raw_next_obs": t(self.next_obs[idx]),
        }
        if self.ctx_dim:
            c0, m0 = self._windows(idx, "obs")
            c1, m1 = self._windows(idx, "next")
            batch["ctx"], batch["mask"] = t(c0), t(m0)
            batch["next_ctx"], batch["next_mask"] = t(c1), t(m1)
            batch["dyn_target"] = t(self.dyn_target[idx])
        return batch

    def sample(self, batch_size, device, obs_norm=None, stratified=False):
        """obs_norm: a RunningNorm applied at sample time.

        stratified: draw an equal share from each fault present in the
        buffer instead of uniformly. Without it the longer-episode fault
        dominates the minibatch and the shared policy silently specializes
        toward it while the per-episode exposure looks balanced.
        """
        if stratified and self.size > 0 and self.n_faults > 1:
            ids = self.fault_id[: self.size]
            pools = [np.flatnonzero(ids == k) for k in range(self.n_faults)]
            pools = [p for p in pools if len(p) > 0]
            if len(pools) > 1:
                per = batch_size // len(pools)
                take = [np.random.choice(p, size=per, replace=True) for p in pools]
                rest = batch_size - per * len(pools)
                if rest:
                    take.append(np.random.randint(0, self.size, size=rest))
                idx = np.concatenate(take)
            else:
                idx = np.random.randint(0, self.size, size=batch_size)
        else:
            idx = np.random.randint(0, self.size, size=batch_size)
        return self.sample_at(idx, device, obs_norm=obs_norm)


class SACAgent:
    def __init__(self, obs_dim, act_dim, device="cuda:0", hidden=256, lr=3e-4,
                 gamma=0.99, tau=0.005, target_entropy=None,
                 alpha_init=0.01, zero_init_actor=True, log_std_init=-1.0):
        self.device = torch.device(device)
        self.gamma, self.tau = gamma, tau
        # Upper bound on the discounted task return: reward 1.0 delivered at
        # best immediately. Any Q far above this is entropy, not task value.
        self._max_task_return = 1.0

        self.actor = Actor(obs_dim, act_dim, hidden,
                           zero_init=zero_init_actor,
                           log_std_init=log_std_init).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_targ = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)

        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = (
            -float(act_dim) if target_entropy is None else float(target_entropy)
        )
        # alpha_init MATTERS ENORMOUSLY HERE, and the usual default of 1.0 is
        # wrong for this reward. SAC's benchmark defaults assume per-step
        # rewards of order 1. Ours is 1.0 exactly once in a ~200-step episode,
        # so the per-step reward is ~0.005. At alpha = 1.0 the entropy bonus
        # contributes alpha * H / (1 - gamma) ~ +400 to Q while the task can
        # pay at most gamma^200 = 0.13. The critic then measures entropy, the
        # actor maximizes entropy, and task success is ~0.5% of the objective.
        # That is not a tuning nuisance -- it is the whole failure.
        self.log_alpha = torch.tensor(
            [float(np.log(alpha_init))], requires_grad=True, device=self.device
        )
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, obs_np, deterministic=False, z_np=None):
        """z_np: the capability latent, concatenated to the observation.

        Conditioning is a CONCAT, not a separate network branch. That is
        what makes `--context_encoder none` (latent_dim = 0) reproduce the
        vanilla action path bit-identically and load the preserved baseline
        checkpoint -- the concat is a no-op on a zero-width latent.
        """
        if z_np is not None and len(z_np) > 0:
            obs_np = np.concatenate([np.asarray(obs_np), np.asarray(z_np)])
        o = torch.as_tensor(obs_np, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logp=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch, z=None, next_z=None):
        """batch: dict from NStepReplayBuffer.sample.

        z / next_z: latents for obs and next_obs. Pass them already detached
        when the encoder is being trained separately (the default), so SAC
        gradients cannot reach the representation and the two failure modes
        stay distinguishable in the logs.
        """
        obs, act = batch["obs"], batch["act"]
        rew, next_obs = batch["rew"], batch["next_obs"]
        term, disc = batch["term"], batch["disc"]
        if z is not None and z.shape[-1] > 0:
            obs = torch.cat([obs, z], dim=-1)
            next_obs = torch.cat([next_obs, next_z], dim=-1)

        with torch.no_grad():
            a2, logp2 = self.actor(next_obs)
            q1t, q2t = self.critic_targ(next_obs, a2)
            qt = torch.min(q1t, q2t) - self.alpha * logp2
            # `disc` is gamma^k for the actual chain length, not gamma^n:
            # chains cut short by success or truncation have k < n.
            target = rew + disc * (1.0 - term) * qt

        q1, q2 = self.critic(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        a, logp = self.actor(obs)
        q1p, q2p = self.critic(obs, a)
        pi_loss = (self.alpha * logp - torch.min(q1p, q2p)).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # SB3 form: gradient taken w.r.t. log_alpha directly rather than
        # exp(log_alpha). Equivalent at the optimum; better conditioned once
        # alpha is small, because the exp() form's gradient shrinks with alpha
        # and recovery from a collapsed alpha becomes glacial.
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.a_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return {
            "loss/critic": q_loss.detach().item(),
            "loss/actor": pi_loss.detach().item(),
            "loss/alpha": alpha_loss.detach().item(),
            "sac/alpha": float(self.alpha),
            "sac/q_mean": q1.detach().mean().item(),
            "sac/target_mean": float(target.mean()),
            "sac/entropy": -logp.detach().mean().item(),
            "sac/q_max_task_return": self._max_task_return,
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_targ": self.critic_targ.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, d):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_targ.load_state_dict(d["critic_targ"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))
