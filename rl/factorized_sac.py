"""FiLM-conditioned SAC for joint-factorized capability latents.

Keeps the validated SAC semantics: tanh Gaussian actor, twin Q, learned alpha,
Polyak target, timeout bootstrapping supplied by the replay buffer, and zero
initial residual. The only architectural change is HOW z conditions control:
FiLM modulation instead of burying z in a concatenated observation vector.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class FiLMBlock(nn.Module):
    """Near-identity modulation initialized exactly neutral."""

    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.to_gb = nn.Linear(z_dim, 2 * hidden)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, h, z):
        gamma, beta = self.to_gb(z).chunk(2, dim=-1)
        # Bounded modulation prevents an early representation transient from
        # catastrophically rescaling a working base controller.
        gamma = 1.0 + 0.5 * torch.tanh(gamma)
        beta = 0.5 * torch.tanh(beta)
        return gamma * h + beta


class FiLMActor(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, act_dim: int, hidden: int = 256,
                 zero_init: bool = True, log_std_init: float = -1.0):
        super().__init__()
        self.trunk = mlp([obs_dim, hidden, hidden])
        self.act = nn.ReLU()
        self.film = FiLMBlock(z_dim, hidden)
        self.post = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        if zero_init:
            nn.init.zeros_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
            nn.init.zeros_(self.log_std.weight)
            nn.init.constant_(self.log_std.bias, log_std_init)

    def forward(self, obs, z, deterministic=False, with_logp=True):
        h = self.act(self.trunk(obs))
        h = self.post(self.film(h, z))
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            u, logp = mu, None
        else:
            dist = torch.distributions.Normal(mu, std)
            u = dist.rsample()
            if with_logp:
                logp = dist.log_prob(u).sum(-1)
                logp = logp - (
                    2 * (np.log(2) - u - F.softplus(-2 * u))
                ).sum(-1)
            else:
                logp = None
        return torch.tanh(u), logp


class FiLMQ(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = mlp([obs_dim + act_dim, hidden, hidden])
        self.act = nn.ReLU()
        self.film = FiLMBlock(z_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, z, act):
        h = self.act(self.trunk(torch.cat([obs, act], dim=-1)))
        h = self.film(h, z)
        return self.head(h).squeeze(-1)


class TwinFiLMCritic(nn.Module):
    def __init__(self, obs_dim, z_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = FiLMQ(obs_dim, z_dim, act_dim, hidden)
        self.q2 = FiLMQ(obs_dim, z_dim, act_dim, hidden)

    def forward(self, obs, z, act):
        return self.q1(obs, z, act), self.q2(obs, z, act)


class FactorizedSACAgent:
    def __init__(self, obs_dim, z_dim, act_dim, device="cuda:0", hidden=256,
                 lr=3e-4, gamma=0.99, tau=0.005, target_entropy=None,
                 alpha_init=0.01, zero_init_actor=True, log_std_init=-1.0):
        self.device = torch.device(device)
        self.obs_dim, self.z_dim, self.act_dim = obs_dim, z_dim, act_dim
        self.gamma, self.tau = gamma, tau
        self.actor = FiLMActor(obs_dim, z_dim, act_dim, hidden,
                               zero_init_actor, log_std_init).to(self.device)
        self.critic = TwinFiLMCritic(obs_dim, z_dim, act_dim, hidden).to(self.device)
        self.critic_targ = TwinFiLMCritic(obs_dim, z_dim, act_dim, hidden).to(self.device)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)

        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.target_entropy = -float(act_dim) if target_entropy is None else float(target_entropy)
        self.log_alpha = torch.tensor(
            [float(np.log(alpha_init))], requires_grad=True, device=self.device
        )
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, obs_np, z_np, deterministic=False):
        o = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = torch.as_tensor(z_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, z, deterministic=deterministic, with_logp=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch, z, next_z, encoder_module=None,
               encoder_optimizer=None, encoder_q_weight: float = 0.0):
        """One SAC update.

        If encoder_q_weight > 0, critic gradients also update the capability
        encoder. Actor gradients do NOT: this keeps the representation from
        learning to exploit entropy while still making it control-aware.
        """
        obs, act = batch["obs"], batch["act"]
        rew, next_obs = batch["rew"], batch["next_obs"]
        term, disc = batch["term"], batch["disc"]

        with torch.no_grad():
            a2, logp2 = self.actor(next_obs, next_z.detach())
            q1t, q2t = self.critic_targ(next_obs, next_z.detach(), a2)
            qt = torch.min(q1t, q2t) - self.alpha * logp2
            target = rew + disc * (1.0 - term) * qt

        q1, q2 = self.critic(obs, z, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad(set_to_none=True)

        # Optional adaptive control of critic -> capability-encoder gradients.
        #
        # Before zeroing the encoder gradients, the parameters still contain
        # the gradient from the immediately preceding self-supervised
        # capability update.  Use its norm as a physically grounded reference.
        #
        # JFCRL_Q_REL_CLIP=0.5 means:
        #   ||lambda_Q * g_Q(applied)|| <= 0.5 * ||g_cap||
        #
        # A value <= 0 disables the relative controller and reproduces the
        # original behavior.
        q_rel_clip = float(
            __import__("os").environ.get("JFCRL_Q_REL_CLIP", "0")
        )

        enc_ref_grad_norm = 0.0
        if (encoder_optimizer is not None and
                encoder_q_weight > 0 and
                encoder_module is not None):
            ref_sq = 0.0
            for p in encoder_module.parameters():
                if p.grad is not None:
                    ref_sq += float(p.grad.detach().pow(2).sum())
            enc_ref_grad_norm = ref_sq ** 0.5

            encoder_optimizer.zero_grad(set_to_none=True)

        q_loss.backward()

        # q_grad_norm is the critic gradient AFTER lambda_Q weighting but
        # BEFORE adaptive relative control.
        enc_grad_norm = 0.0
        enc_grad_applied_norm = 0.0
        enc_grad_scale = 1.0
        enc_q_to_ref_ratio = 0.0

        if encoder_optimizer is not None and encoder_q_weight > 0 and encoder_module is not None:
            sq = 0.0
            for p in encoder_module.parameters():
                if p.grad is not None:
                    p.grad.mul_(float(encoder_q_weight))
                    sq += float(p.grad.detach().pow(2).sum())

            enc_grad_norm = sq ** 0.5

            if enc_ref_grad_norm > 0.0:
                enc_q_to_ref_ratio = enc_grad_norm / (enc_ref_grad_norm + 1e-12)

            # Adaptive relative control.  Under normal operation this is
            # exactly 1.0, so the original critic signal is unchanged.
            if q_rel_clip > 0.0 and enc_grad_norm > 0.0:
                if enc_ref_grad_norm > 0.0:
                    max_q_norm = q_rel_clip * enc_ref_grad_norm
                else:
                    # Conservative fallback only if no reference gradient is
                    # available on this update.
                    max_q_norm = 0.1

                if enc_grad_norm > max_q_norm:
                    enc_grad_scale = max_q_norm / (enc_grad_norm + 1e-12)
                    for p in encoder_module.parameters():
                        if p.grad is not None:
                            p.grad.mul_(enc_grad_scale)

            enc_grad_applied_norm = enc_grad_norm * enc_grad_scale

            # Keep the original absolute safety net as well.
            torch.nn.utils.clip_grad_norm_(encoder_module.parameters(), 5.0)
            encoder_optimizer.step()

        self.q_opt.step()

        # Actor sees a detached capability estimate. This is deliberate: the
        # critic says which distinctions matter for control; actor entropy does
        # not directly reshape the embodiment representation.
        z_det = z.detach()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        a, logp = self.actor(obs, z_det)
        q1p, q2p = self.critic(obs, z_det, a)
        pi_loss = (self.alpha * logp - torch.min(q1p, q2p)).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.a_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return {
            "loss/critic": float(q_loss.detach()),
            "loss/actor": float(pi_loss.detach()),
            "loss/alpha": float(alpha_loss.detach()),
            "sac/alpha": float(self.alpha),
            "sac/q_mean": float(q1.detach().mean()),
            "sac/target_mean": float(target.detach().mean()),
            "sac/entropy": float(-logp.detach().mean()),
            "cap/q_grad_norm": float(enc_grad_norm),
            "cap/q_grad_applied_norm": float(enc_grad_applied_norm),
            "cap/q_grad_scale": float(enc_grad_scale),
            "cap/q_ref_grad_norm": float(enc_ref_grad_norm),
            "cap/q_to_ref_ratio": float(enc_q_to_ref_ratio),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_targ": self.critic_targ.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "pi_opt": self.pi_opt.state_dict(),
            "q_opt": self.q_opt.state_dict(),
            "a_opt": self.a_opt.state_dict(),
        }

    def load_state_dict(self, d, load_optimizers=False):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_targ.load_state_dict(d["critic_targ"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))
        if load_optimizers:
            if d.get("pi_opt"):
                self.pi_opt.load_state_dict(d["pi_opt"])
            if d.get("q_opt"):
                self.q_opt.load_state_dict(d["q_opt"])
            if d.get("a_opt"):
                self.a_opt.load_state_dict(d["a_opt"])
