"""Joint-factorized kinematic capability encoder.

Purpose
-------
The previous shared GRU compressed the entire robot history into one free
latent. That makes it easy to memorize a small set of fault modes. This module
factorizes adaptation by actuator:

    history -> SAME temporal encoder for each of 7 joints -> c_i
            -> current kinematic query (q_i, qdot_i, J_i, a_VLA)
            -> permutation-equivariant cross-joint Transformer
            -> global capability latent z

No joint id, fault id, lock value, severity label, or fault family is given to
the learner. Joint identity is expressed only through physical state and the
current Jacobian column. The same temporal weights are applied to every joint.

Self-supervision is also factorized:
  * a shared local decoder predicts the next realized dq_i for every joint;
  * a global decoder predicts realized end-effector motion;
  * a kinematic consistency term checks J(q) @ predicted_dq against realized
    end-effector motion.

The RL reward remains task success only. These losses shape representation,
not environment reward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class ObsLayout:
    """Offsets inherited from ResidualLiberoEnv before the appended Jacobian."""

    proprio0: int = 0
    q0: int = 8
    qd0: int = 15
    abase0: int = 22
    phase: int = 29
    time: int = 30


class JointTemporalEncoder(nn.Module):
    """One GRU shared across all joints.

    Input is left-padded (real history at the right). We explicitly repack it
    into right-padded chronological sequences before calling the GRU. Merely
    zeroing left padding is not enough because GRU biases can evolve hidden
    state on padded steps.
    """

    def __init__(self, token_dim: int, hidden: int, cap_dim: int):
        super().__init__()
        self.token_dim = int(token_dim)
        self.inp = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, cap_dim)
        self.norm = nn.LayerNorm(cap_dim)

    @staticmethod
    def _left_to_right_padded(x: torch.Tensor, mask: torch.Tensor):
        """Move real suffix to the front while preserving chronology."""
        # x: (B, K, D), mask: (B, K), left padded.
        B, K, D = x.shape
        lengths = mask.sum(dim=1).long()
        safe_lengths = lengths.clamp(min=1)
        starts = K - lengths
        pos = torch.arange(K, device=x.device).unsqueeze(0).expand(B, K)
        src = (starts.unsqueeze(1) + pos).clamp(0, K - 1)
        y = x.gather(1, src.unsqueeze(-1).expand(B, K, D))
        valid = pos < lengths.unsqueeze(1)
        y = y * valid.unsqueeze(-1)
        return y, safe_lengths, lengths

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B*J, K, token_dim); mask: (B*J, K)
        x = self.inp(x)
        x, safe_lengths, real_lengths = self._left_to_right_padded(x, mask)
        packed = pack_padded_sequence(
            x, safe_lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        out = self.norm(self.proj(h[-1]))
        out = out * (real_lengths > 0).to(out.dtype).unsqueeze(-1)
        return out


class JointFactorizedCapabilityModule(nn.Module):
    """Shared per-joint capability inference + kinematic graph aggregation."""

    n_joints = 7
    dyn_dim = 13

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 6,
        token_dim: int = 28,
        context_len: int = 16,
        temporal_hidden: int = 128,
        cap_dim: int = 32,
        z_dim: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn: int = 256,
        lr: float = 1e-4,
        lambda_joint: float = 1.0,
        lambda_eef: float = 1.0,
        lambda_kin: float = 0.25,
        device: str = "cpu",
    ):
        super().__init__()
        if cap_dim % transformer_heads != 0:
            raise ValueError("cap_dim must be divisible by transformer_heads")
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.token_dim = int(token_dim)
        self.context_len = int(context_len)
        self.cap_dim = int(cap_dim)
        self.latent_dim = int(z_dim)
        self.lambda_joint = float(lambda_joint)
        self.lambda_eef = float(lambda_eef)
        self.lambda_kin = float(lambda_kin)
        self.layout = ObsLayout()
        self.jac_dim = 42
        self.jac_offset = self.obs_dim - self.jac_dim
        if self.jac_offset <= self.layout.time:
            raise ValueError(
                f"obs_dim={obs_dim} is too small for appended 6x7 Jacobian"
            )

        self.temporal = JointTemporalEncoder(token_dim, temporal_hidden, cap_dim)

        # Current pre-action query per joint: q_i, qdot_i, J_i(6), a_VLA_arm(6).
        self.query_dim = 14
        self.query_proj = nn.Sequential(
            nn.Linear(self.query_dim, cap_dim),
            nn.LayerNorm(cap_dim),
            nn.SiLU(),
            nn.Linear(cap_dim, cap_dim),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=cap_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cross_joint = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.cross_norm = nn.LayerNorm(cap_dim)
        self.pool = nn.Sequential(
            nn.Linear(cap_dim, z_dim),
            nn.SiLU(),
            nn.LayerNorm(z_dim),
        )

        # Shared local dynamics decoder. Same weights for every joint.
        # [capability token, current physical query, residual action] -> dq_i.
        self.local_decoder = nn.Sequential(
            nn.Linear(cap_dim + self.query_dim + act_dim, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

        # Global eef dynamics decoder. This handles contacts / controller
        # nonlinearities not captured by first-order J*dq alone.
        self.eef_decoder = nn.Sequential(
            nn.Linear(z_dim + obs_dim + act_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 6),
        )

        # Running normalization for the *shared token feature semantics*.
        # One 28-D normalizer is shared over all joints: no joint-specific
        # statistics that could become a hidden joint identifier.
        self.ctx_mean = np.zeros(token_dim, dtype=np.float64)
        self.ctx_var = np.ones(token_dim, dtype=np.float64)
        self.ctx_count = 1e-4

        self.tgt_mean = np.zeros(self.dyn_dim, dtype=np.float64)
        self.tgt_var = np.ones(self.dyn_dim, dtype=np.float64)
        self.tgt_count = 1e-4

        self.to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Running statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _welford(mean, var, count, x: np.ndarray):
        x = np.atleast_2d(x).astype(np.float64)
        if x.shape[0] == 0:
            return mean, var, count
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - mean
        total = count + bc
        mean = mean + delta * bc / total
        var = (var * count + bv * bc + delta**2 * count * bc / total) / total
        return mean, var, total

    def update_context_stats(self, flat_step: np.ndarray):
        x = np.asarray(flat_step, dtype=np.float64).reshape(-1, self.token_dim)
        self.ctx_mean, self.ctx_var, self.ctx_count = self._welford(
            self.ctx_mean, self.ctx_var, self.ctx_count, x
        )

    def update_target_stats(self, y: np.ndarray):
        y = np.asarray(y, dtype=np.float64).reshape(-1, self.dyn_dim)
        self.tgt_mean, self.tgt_var, self.tgt_count = self._welford(
            self.tgt_mean, self.tgt_var, self.tgt_count, y
        )

    def _norm_ctx(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.ctx_mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(np.sqrt(self.ctx_var + 1e-8), dtype=x.dtype, device=x.device)
        return (x - m) / s

    def _norm_target(self, y: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.tgt_mean, dtype=y.dtype, device=y.device)
        s = torch.as_tensor(np.sqrt(self.tgt_var + 1e-8), dtype=y.dtype, device=y.device)
        return (y - m) / s

    def _denorm_dq(self, dq_n: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.tgt_mean[:7], dtype=dq_n.dtype, device=dq_n.device)
        s = torch.as_tensor(
            np.sqrt(self.tgt_var[:7] + 1e-8), dtype=dq_n.dtype, device=dq_n.device
        )
        return dq_n * s + m

    def _norm_eef_pred(self, eef: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.tgt_mean[7:13], dtype=eef.dtype, device=eef.device)
        s = torch.as_tensor(
            np.sqrt(self.tgt_var[7:13] + 1e-8), dtype=eef.dtype, device=eef.device
        )
        return (eef - m) / s

    # ------------------------------------------------------------------
    # Physical query extraction
    # ------------------------------------------------------------------
    def query_from_raw_obs(self, raw_obs: torch.Tensor):
        """Return query (B,7,14) and J (B,6,7) from current observation."""
        if raw_obs.ndim == 1:
            raw_obs = raw_obs.unsqueeze(0)
        q = raw_obs[:, self.layout.q0:self.layout.q0 + 7]
        qd = raw_obs[:, self.layout.qd0:self.layout.qd0 + 7]
        abase = raw_obs[:, self.layout.abase0:self.layout.abase0 + 6]
        jflat = raw_obs[:, self.jac_offset:self.jac_offset + 42]
        if jflat.shape[-1] != 42:
            raise RuntimeError("current observation does not contain a 6x7 Jacobian")
        # Environment stores joint-major [J_0(6), ..., J_6(6)].
        j_joint = jflat.reshape(-1, 7, 6)
        J = j_joint.transpose(1, 2)  # (B,6,7)
        abase_rep = abase.unsqueeze(1).expand(-1, 7, -1)
        query = torch.cat([q.unsqueeze(-1), qd.unsqueeze(-1), j_joint, abase_rep], dim=-1)
        return query, J

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def encode(self, ctx: torch.Tensor, mask: torch.Tensor, raw_obs: torch.Tensor,
               return_joint: bool = False):
        """Encode past execution plus current kinematics.

        ctx shape: (B,K,7*token_dim), left padded.
        mask shape: (B,K), one per real environment step.
        raw_obs: current observation with appended current Jacobian.
        """
        if ctx.ndim != 3 or ctx.shape[-1] != 7 * self.token_dim:
            raise ValueError(
                f"ctx must be (B,K,{7*self.token_dim}), got {tuple(ctx.shape)}"
            )
        B, K, _ = ctx.shape
        x = ctx.reshape(B, K, 7, self.token_dim)
        x = self._norm_ctx(x)

        # Shared temporal encoder: B*7 independent sequences, identical weights.
        xj = x.permute(0, 2, 1, 3).reshape(B * 7, K, self.token_dim)
        mj = mask.unsqueeze(1).expand(B, 7, K).reshape(B * 7, K)
        cap_hist = self.temporal(xj, mj).reshape(B, 7, self.cap_dim)

        query, _ = self.query_from_raw_obs(raw_obs)
        cap = cap_hist + self.query_proj(query)
        cap = self.cross_norm(self.cross_joint(cap))
        z = self.pool(cap.mean(dim=1))
        if return_joint:
            return z, cap
        return z

    @torch.no_grad()
    def encode_numpy(self, ctx_np: np.ndarray, mask_np: np.ndarray,
                     raw_obs_np: np.ndarray):
        ctx = torch.as_tensor(ctx_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(mask_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs = torch.as_tensor(raw_obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.encode(ctx, mask, obs)
        return z.squeeze(0).cpu().numpy()

    def encode_for_policy(self, ctx, mask, raw_obs, detach: bool = True):
        z = self.encode(ctx, mask, raw_obs)
        return z.detach() if detach else z

    # ------------------------------------------------------------------
    # Representation learning
    # ------------------------------------------------------------------
    def representation_loss(self, ctx, mask, obs_norm, raw_obs, act, y_target):
        z, cap = self.encode(ctx, mask, raw_obs, return_joint=True)
        query, J = self.query_from_raw_obs(raw_obs)
        a_rep = act.unsqueeze(1).expand(-1, 7, -1)
        local_in = torch.cat([cap, query, a_rep], dim=-1)
        pred_dq_n = self.local_decoder(local_in).squeeze(-1)

        target_n = self._norm_target(y_target)
        loss_joint = F.huber_loss(pred_dq_n, target_n[:, :7])

        pred_eef_n = self.eef_decoder(torch.cat([z, obs_norm, act], dim=-1))
        loss_eef = F.huber_loss(pred_eef_n, target_n[:, 7:13])

        # First-order physical consistency using current geometric Jacobian.
        pred_dq = self._denorm_dq(pred_dq_n)
        pred_eef_phys = torch.einsum("bij,bj->bi", J, pred_dq)
        loss_kin = F.huber_loss(self._norm_eef_pred(pred_eef_phys), target_n[:, 7:13])

        total = (
            self.lambda_joint * loss_joint
            + self.lambda_eef * loss_eef
            + self.lambda_kin * loss_kin
        )
        # Token diversity is diagnostic, not an extra objective.
        joint_var = cap.detach().var(dim=1).mean()
        batch_var = z.detach().var(dim=0).mean()
        return total, {
            "cap/L_joint": float(loss_joint.detach()),
            "cap/L_eef": float(loss_eef.detach()),
            "cap/L_kin": float(loss_kin.detach()),
            "cap/L_total": float(total.detach()),
            "cap/joint_token_var": float(joint_var),
            "cap/z_var": float(batch_var),
            "cap/z_norm": float(z.detach().norm(dim=-1).mean()),
        }

    def update(self, ctx, mask, obs_norm, raw_obs, act, y_target):
        self.opt.zero_grad(set_to_none=True)
        loss, metrics = self.representation_loss(
            ctx, mask, obs_norm, raw_obs, act, y_target
        )
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        self.opt.step()
        metrics["cap/rep_grad_norm"] = float(grad)
        return metrics

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def checkpoint_state(self):
        return {
            "model": super().state_dict(),
            "opt": self.opt.state_dict(),
            "ctx_mean": self.ctx_mean,
            "ctx_var": self.ctx_var,
            "ctx_count": self.ctx_count,
            "tgt_mean": self.tgt_mean,
            "tgt_var": self.tgt_var,
            "tgt_count": self.tgt_count,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "token_dim": self.token_dim,
            "context_len": self.context_len,
            "cap_dim": self.cap_dim,
            "z_dim": self.latent_dim,
            "lambda_joint": self.lambda_joint,
            "lambda_eef": self.lambda_eef,
            "lambda_kin": self.lambda_kin,
        }

    def load_checkpoint_state(self, d, load_optimizer: bool = True):
        if int(d["token_dim"]) != self.token_dim or int(d["z_dim"]) != self.latent_dim:
            raise ValueError("capability checkpoint architecture mismatch")
        super().load_state_dict(d["model"])
        if load_optimizer and d.get("opt"):
            self.opt.load_state_dict(d["opt"])
        self.ctx_mean = np.asarray(d["ctx_mean"], dtype=np.float64)
        self.ctx_var = np.asarray(d["ctx_var"], dtype=np.float64)
        self.ctx_count = float(d["ctx_count"])
        self.tgt_mean = np.asarray(d["tgt_mean"], dtype=np.float64)
        self.tgt_var = np.asarray(d["tgt_var"], dtype=np.float64)
        self.tgt_count = float(d["tgt_count"])


def build_left_padded_history(hist, K: int, ctx_dim: int):
    """Online/history helper shared by trainer and evaluator."""
    ctx = np.zeros((K, ctx_dim), dtype=np.float32)
    mask = np.zeros(K, dtype=np.float32)
    n = min(len(hist), K)
    if n:
        ctx[K - n:] = np.asarray(list(hist)[-n:], dtype=np.float32)
        mask[K - n:] = 1.0
    return ctx, mask
