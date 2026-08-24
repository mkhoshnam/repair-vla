"""
jfcrl_v2_heads.py -- capability-driven gate for JFCRL.

Arm A  FiLMActor        unchanged, rl/factorized_sac.py
Arm B  GatedFiLMActor   this file

No task embedding. The gate reads z_cap only.

Why
---
Every regression measured so far sits where the frozen policy was already
strong: healthy 98->92 in the 5-task run, and goal:0 100->60, goal:3 50->20,
10:3 50->0 in the 20-task run. An always-on residual with fixed eta cannot
stand down when the base policy is fine. The gate gives it that option using
information the capability encoder already has, and stays actuator-only, so
it generalises to tasks and joints never seen.

Replay correctness
------------------
The gate is applied INSIDE the actor:  a = g(z) * tanh(u).
The trainer stores exactly the `a` returned by agent.act and passes it to
b.renv.step(a, gate=1.0), so replay holds the action the robot received.
Do NOT also pass a gate to the environment -- that would desynchronise the
stored action from the executed one.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class FiLMBlock(nn.Module):
    """Mirrors rl/factorized_sac.py:27 so this module imports standalone."""

    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.to_gb = nn.Linear(z_dim, 2 * hidden)
        # Match rl.factorized_sac.FiLMBlock exactly: neutral modulation at init.
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, h, z):
        g, b = self.to_gb(z).chunk(2, dim=-1)
        return (1.0 + 0.5 * torch.tanh(g)) * h + 0.5 * torch.tanh(b)


class GatedFiLMActor(nn.Module):
    """FiLMActor with a scalar gate g(z_cap) in [gate_floor, 1].

        a = g(z) * tanh(u),    u ~ N(mu(obs, z), sigma(obs, z))

    Log-density. g depends on z but not on u, so the change of variables adds
    a constant per sample:

        log p(a) = log p(tanh(u)) - act_dim * log g(z)

    Applied so SAC's temperature sees the entropy of the executed action. Note
    the sign: closing the gate raises log p, so the entropy term mildly
    resists gating. At the alphas seen in these runs (~1e-4) this is
    negligible; set logp_jacobian=False if alpha ever becomes large.

    Init. The deterministic mean action is exactly zero at step 0. The gate is
    initialized near one (sigmoid(4) ~= 0.982) so stochastic exploration is
    also close to arm A while still leaving usable gradient to learn to close.
    """

    def __init__(self, obs_dim, z_dim, act_dim, hidden=256, zero_init=True,
                 log_std_init=-1.0, gate_hidden=64, gate_bias_init=4.0,
                 gate_floor=0.0, logp_jacobian=True):
        super().__init__()
        self.trunk = mlp([obs_dim, hidden, hidden])
        self.act = nn.ReLU()
        self.film = FiLMBlock(z_dim, hidden)
        self.post = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

        # Reads z_cap ONLY: not obs, not task. It must justify itself from
        # inferred capability, which is the claim under test.
        self.gate_net = nn.Sequential(
            nn.Linear(z_dim, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, gate_bias_init)

        self.act_dim = act_dim
        self.gate_floor = float(gate_floor)
        self.logp_jacobian = bool(logp_jacobian)

        if zero_init:
            nn.init.zeros_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
            nn.init.zeros_(self.log_std.weight)
            nn.init.constant_(self.log_std.bias, log_std_init)

    def gate(self, z):
        g = torch.sigmoid(self.gate_net(z))
        if self.gate_floor > 0:
            g = self.gate_floor + (1.0 - self.gate_floor) * g
        return g

    def forward(self, obs, z, deterministic=False, with_logp=True,
                return_gate=False):
        h = self.post(self.film(self.act(self.trunk(obs)), z))
        mu = self.mu(h)
        std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX).exp()

        if deterministic:
            u, logp = mu, None
        else:
            dist = torch.distributions.Normal(mu, std)
            u = dist.rsample()
            if with_logp:
                logp = dist.log_prob(u).sum(-1) - (
                    2 * (np.log(2) - u - F.softplus(-2 * u))
                ).sum(-1)
            else:
                logp = None

        g = self.gate(z)
        a = g * torch.tanh(u)
        if logp is not None and self.logp_jacobian:
            logp = logp - self.act_dim * torch.log(g.squeeze(-1) + 1e-8)

        return (a, logp, g) if return_gate else (a, logp)


# ------------------------------------------------------------- reporting ---
@torch.no_grad()
def gate_by_task_condition(actor, z, task_keys, conditions):
    """Mean gate per (task, condition) cell -- the mechanism figure.

    Gate-near-1-on-every-fault is NOT the target. Where the frozen VLA already
    handles the fault (goal:0 / j2 screened 80%), a LOW gate is correct: it
    preserves the base policy and avoids the negative transfer we measured.
    Where compensation is needed (object:6 / j2 screened 0%), a HIGH gate is
    correct. The claim is that the gate tracks *how much help is needed*, and
    that only shows up per cell, not per condition.

    Returns {(task, condition): (mean_gate, n)}.
    """
    g = actor.gate(z).squeeze(-1).cpu().numpy()
    tk, cd = np.asarray(task_keys), np.asarray(conditions)
    out = {}
    for t in np.unique(tk):
        for c in np.unique(cd):
            m = (tk == t) & (cd == c)
            if m.any():
                out[(str(t), str(c))] = (float(g[m].mean()), int(m.sum()))
    return out


def format_gate_table(cells):
    tasks = sorted({t for t, _ in cells})
    conds = sorted({c for _, c in cells})
    w = max(len(t) for t in tasks) + 2
    lines = ["gate by task x condition (1 = full residual authority)",
             " " * w + "".join(f"{c:>10s}" for c in conds)]
    for t in tasks:
        row = f"{t:<{w}}"
        for c in conds:
            v = cells.get((t, c))
            row += f"{v[0]:>10.3f}" if v else f"{'-':>10s}"
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, obs_dim, z_dim, act_dim = 12, 40, 64, 6
    obs, z = torch.randn(B, obs_dim), torch.randn(B, z_dim)

    a = GatedFiLMActor(obs_dim, z_dim, act_dim)
    act, logp, g = a(obs, z, return_gate=True)
    assert act.shape == (B, act_dim) and logp.shape == (B,)
    assert 0.0 <= float(g.min()) and float(g.max()) <= 1.0

    det, _ = a(obs, z, deterministic=True)
    assert torch.allclose(det, torch.zeros_like(det), atol=1e-6), \
        "zero-init must make the initial residual exactly zero"

    # gradient must reach the gate
    a(obs, z)[0].sum().backward()
    assert a.gate_net[-1].weight.grad is not None
    assert a.gate_net[-1].weight.grad.abs().sum() > 0

    # floor is respected
    af = GatedFiLMActor(obs_dim, z_dim, act_dim, gate_floor=0.2,
                        gate_bias_init=-8.0)
    assert float(af.gate(z).min()) >= 0.2 - 1e-6

    cells = gate_by_task_condition(
        a, z,
        ["goal:0"] * 6 + ["object:6"] * 6,
        (["healthy"] * 3 + ["j2"] * 3) * 2,
    )
    print(format_gate_table(cells))
    print("\nall checks passed")
