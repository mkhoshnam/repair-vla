"""
context_encoder.py -- capability inference from closed-loop execution history.

    z_t = E_psi(H_t)          H_t = last K context features, ending at t-1
    yhat = D_omega(o_t, a_t, z_t)  ->  [dq_{t}, dx_EE_{t}]   (Huber loss)

The encoder never sees a fault label. It sees what the deployed controller
sees: what was commanded and what the body actually did. A locked joint
makes realized motion unpredictable from the command alone, so a decoder
forced to predict realized motion cannot succeed without z carrying the
actuator's current capability. That is the whole mechanism.

--------------------------------------------------------------------------
THE WINDOW ENDS AT t-1, NOT t. THIS IS A LEAKAGE BOUNDARY.
--------------------------------------------------------------------------
The context feature for step t contains the motion realized DURING step t.
If the window included step t, then z_t would already contain the very
quantity the dynamics decoder is asked to predict, L_dyn would fall to
zero immediately, and the latent would learn nothing about capability.

So the window is [t-K, ..., t-1] and the target is the realized delta at
step t. This also matches what is available online: at the moment of
choosing an action for step t, steps up to t-1 have happened and step t
has not. Offline sampling and online rollout therefore see the SAME
information, which is what makes a checkpoint behave in evaluation the way
it behaved in training.

--------------------------------------------------------------------------
GRU FIRST (spec section 3.2)
--------------------------------------------------------------------------
K = 16 is two OpenVLA action chunks -- short. A 1-layer GRU is the lower
risk option and recurrent off-policy SAC is a well-established baseline
under partial observability. The Transformer lives behind the same
interface as a controlled ablation and is not the default.

Padding is LEFT-side (early-episode windows are short) and padded steps are
zeroed AND masked. Left padding means real steps are always the most recent
inputs to the GRU, so taking the final hidden state is correct without
packing. `none` returns a zero-width latent so the downstream concat is a
no-op and the vanilla MLP path is bit-identical.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoContextEncoder(nn.Module):
    """Zero-width latent. Keeps the vanilla SAC path bit-identical.

    Not a stub: `--context_encoder none` must reproduce the validated
    shared-SAC action path exactly and must be able to load the preserved
    baseline checkpoint, because that run is the paper's baseline arm.
    """

    latent_dim = 0

    def forward(self, ctx, mask):
        return ctx.new_zeros((ctx.shape[0], 0))

    def initial_state(self):
        return None


class GRUContextEncoder(nn.Module):
    def __init__(self, ctx_dim: int, hidden: int = 128, latent_dim: int = 32,
                 num_layers: int = 1):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.latent_dim = latent_dim
        self.gru = nn.GRU(ctx_dim, hidden, num_layers=num_layers,
                          batch_first=True)
        self.proj = nn.Linear(hidden, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, ctx, mask):
        """ctx: (B, K, ctx_dim)  mask: (B, K) with 1 = real step."""
        ctx = ctx * mask.unsqueeze(-1)          # padded steps contribute zero
        out, _ = self.gru(ctx)
        # Left padding => the last position is always the most recent REAL
        # step whenever the window contains any real step at all.
        h = out[:, -1]
        return self.norm(self.proj(h))


class TransformerContextEncoder(nn.Module):
    """Ablation only (spec 3.2). Same interface, never the default."""

    def __init__(self, ctx_dim: int, hidden: int = 128, latent_dim: int = 32,
                 n_layers: int = 2, n_heads: int = 4, ffn: int = 256,
                 max_len: int = 64):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.latent_dim = latent_dim
        self.inp = nn.Linear(ctx_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=ffn,
            batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(hidden, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, ctx, mask):
        B, K, _ = ctx.shape
        x = self.inp(ctx * mask.unsqueeze(-1)) + self.pos[:, :K]
        pad = mask < 0.5
        # A fully padded row would make softmax produce NaN, so keep at least
        # the final position attendable; its input is zeroed anyway.
        pad = pad & ~(pad.all(dim=1, keepdim=True)
                      & (torch.arange(K, device=ctx.device) == K - 1))
        h = self.enc(x, src_key_padding_mask=pad)
        return self.norm(self.proj(h[:, -1]))


class DynamicsDecoder(nn.Module):
    """Predicts the realized motion of step t from (o_t, a_t, z_t).

    Target is [dq (7), d_eef_pos (3), d_eef_rot (3)] = 13-D, normalized.
    Deliberately small: it exists to shape z, not to be a world model. If it
    were large enough to predict the dynamics from (o, a) alone, z would be
    free to carry nothing.
    """

    out_dim = 13

    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, obs, act, z):
        return self.net(torch.cat([obs, act, z], dim=-1))


def build_context_encoder(kind: str, ctx_dim: int, hidden: int = 128,
                          latent_dim: int = 32, context_len: int = 16):
    kind = (kind or "none").lower()
    if kind == "none":
        return NoContextEncoder()
    if kind == "gru":
        return GRUContextEncoder(ctx_dim, hidden, latent_dim)
    if kind == "transformer":
        return TransformerContextEncoder(
            ctx_dim, hidden, latent_dim, max_len=max(context_len, 8))
    raise ValueError(f"unknown context_encoder '{kind}'; "
                     f"use none | gru | transformer")


class ContextModule(nn.Module):
    """Encoder + dynamics decoder + target normalizer, with one optimizer.

    Kept separate from the SAC optimizer on purpose (spec 4.2): with the
    representation and the control losses on different optimizers, a
    representation failure and a control failure stay distinguishable in the
    logs. `detach_for_policy=True` (the default) means SAC gradients never
    reach the encoder in the first stable version.
    """

    def __init__(self, ctx_dim: int, obs_dim: int, act_dim: int,
                 kind: str = "gru", hidden: int = 128, latent_dim: int = 32,
                 context_len: int = 16, lr: float = 1e-4,
                 lambda_dyn: float = 1.0, lambda_slow: float = 0.0,
                 detach_for_policy: bool = True,
                 gate_beta: float = 0.0, gate_min: float = 0.0,
                 device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.kind = (kind or "none").lower()
        self.context_len = context_len
        self.lambda_dyn = lambda_dyn
        self.lambda_slow = lambda_slow
        self.detach_for_policy = detach_for_policy

        self.encoder = build_context_encoder(
            self.kind, ctx_dim, hidden, latent_dim, context_len).to(self.device)
        self.latent_dim = self.encoder.latent_dim

        self.decoder = None
        self.opt = None
        if self.latent_dim > 0:
            self.decoder = DynamicsDecoder(
                obs_dim, act_dim, self.latent_dim).to(self.device)
            self.opt = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=lr)

        # Targets mix joint radians and metres; without normalization the
        # Huber loss is dominated by whichever unit happens to be larger.
        self.tgt_mean = np.zeros(DynamicsDecoder.out_dim, dtype=np.float64)
        self.tgt_var = np.ones(DynamicsDecoder.out_dim, dtype=np.float64)
        self.tgt_count = 1e-4

        # Novelty statistics: what dynamics-prediction error looks like on
        # the TRAINING faults. gate_beta = 0 disables gating entirely, which
        # is the exact ungated policy -- that is the ablation.
        self.gate_beta = float(gate_beta)
        self.gate_min = float(gate_min)
        self.nov_mean = 0.0
        self.nov_var = 1.0
        self.nov_count = 1e-4
        # Statistics accumulated DURING training are not usable for gating.
        # Early on the decoder is bad, the encoder is still moving, and the
        # target normalizer is still shifting, so an error from step 2k and
        # one from step 150k are not the same quantity -- yet a running
        # mean mixes them and saves the average forever.
        #
        # The gate therefore refuses to fire until a dedicated calibration
        # pass has run with the encoder and decoder FROZEN
        # (scripts/calibrate_novelty.py). Until then `gate()` returns 1.0,
        # which is exactly the ungated policy.
        self.nov_calibrated = False

    # ------------------------------------------------------------------ #

    def update_target_stats(self, y: np.ndarray):
        y = np.atleast_2d(y).astype(np.float64)
        bm, bv, bc = y.mean(0), y.var(0), y.shape[0]
        d = bm - self.tgt_mean
        tot = self.tgt_count + bc
        self.tgt_mean += d * bc / tot
        self.tgt_var = ((self.tgt_var * self.tgt_count + bv * bc
                         + d**2 * self.tgt_count * bc / tot) / tot)
        self.tgt_count = tot

    def norm_target(self, y_t: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.tgt_mean, dtype=y_t.dtype, device=y_t.device)
        s = torch.as_tensor(np.sqrt(self.tgt_var + 1e-8), dtype=y_t.dtype,
                            device=y_t.device)
        return (y_t - m) / s

    def encode(self, ctx, mask):
        return self.encoder(ctx, mask)

    def encode_for_policy(self, ctx, mask):
        z = self.encoder(ctx, mask)
        return z.detach() if self.detach_for_policy else z

    @torch.no_grad()
    def encode_numpy(self, ctx_np: np.ndarray, mask_np: np.ndarray):
        """Online path: one window -> one latent, as a numpy vector."""
        if self.latent_dim == 0:
            return np.zeros(0, dtype=np.float32)
        ctx = torch.as_tensor(ctx_np, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        mask = torch.as_tensor(mask_np, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        return self.encoder(ctx, mask).squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------ #

    def update(self, ctx, mask, obs, act, y_target, prev_ctx=None,
               prev_mask=None):
        """One representation step. Returns metrics; empty if disabled."""
        if self.latent_dim == 0:
            return {}

        z = self.encoder(ctx, mask)
        pred = self.decoder(obs, act, z)
        tgt = self.norm_target(y_target)
        loss_dyn = F.huber_loss(pred, tgt)
        loss = self.lambda_dyn * loss_dyn

        out = {
            "ctx/L_dyn": loss_dyn.detach().item(),
            "ctx/z_norm": z.detach().norm(dim=-1).mean().item(),
            # Per-dimension variance across the batch. If this collapses to
            # ~0 the latent is a constant and conditioning is doing nothing,
            # which looks identical to "the encoder learned nothing useful".
            "ctx/z_var": z.detach().var(dim=0).mean().item(),
        }

        if self.lambda_slow > 0 and prev_ctx is not None:
            z_prev = self.encoder(prev_ctx, prev_mask).detach()
            loss_slow = F.mse_loss(z, z_prev)
            loss = loss + self.lambda_slow * loss_slow
            out["ctx/L_slow"] = loss_slow.detach().item()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        out["ctx/loss_total"] = loss.detach().item()
        return out

    # ------------------------------------------------------------------ #
    # NOVELTY: a free, self-supervised out-of-distribution signal
    # ------------------------------------------------------------------ #
    #
    # The decoder is trained to predict realized motion under the faults the
    # encoder has SEEN. On an unfamiliar fault it predicts badly. That error
    # is therefore an OOD detector that costs nothing extra and -- crucially
    # -- uses no fault label, so it does not touch the headline claim.
    #
    # Why this matters here: the 20k GRU checkpoint scored 0/10 on unseen j2
    # while the FROZEN VLA scored 8/10. That is not a failure to recover, it
    # is negative transfer -- the policy matched an unfamiliar fault to the
    # nearest memorized mode and applied that correction at full authority.
    # A residual that shrinks when the dynamics are unfamiliar falls back to
    # the frozen VLA instead of corrupting it.
    #
    # Be precise about what this buys: it converts catastrophic negative
    # transfer into "no worse than frozen". It is a SAFETY NET, not
    # generalization. Actually recovering an unseen joint requires the
    # training distribution to cover the latent space (spec section 5).

    @torch.no_grad()
    def per_sample_dyn_error(self, ctx, mask, obs, act, y_target):
        """Per-row Huber error, no gradient step. Shape (B,)."""
        if self.latent_dim == 0:
            return torch.zeros(ctx.shape[0], device=ctx.device)
        z = self.encoder(ctx, mask)
        pred = self.decoder(obs, act, z)
        return F.huber_loss(pred, self.norm_target(y_target),
                            reduction="none").mean(dim=-1)

    def reset_novelty_stats(self):
        """Discard training-time statistics before a calibration pass."""
        self.nov_mean, self.nov_var, self.nov_count = 0.0, 1.0, 1e-4
        self.nov_calibrated = False

    def update_novelty_stats(self, errs):
        """Running mean/std of the dyn error ON TRAINING CONDITIONS ONLY.

        These define what "familiar" means. Feeding held-out conditions in
        here would teach the detector that the held-out fault is normal,
        which is exactly the leak the whole design is trying to avoid.
        """
        e = np.atleast_1d(np.asarray(errs, dtype=np.float64))
        bm, bv, bc = e.mean(), e.var(), e.size
        d = bm - self.nov_mean
        tot = self.nov_count + bc
        self.nov_mean += d * bc / tot
        self.nov_var = ((self.nov_var * self.nov_count + bv * bc
                         + d**2 * self.nov_count * bc / tot) / tot)
        self.nov_count = tot

    def novelty(self, err) -> float:
        """One-sided z-score of the dyn error against training statistics.

        One-sided on purpose: predicting the motion BETTER than usual is not
        evidence of an unfamiliar fault, so only positive excursions count.
        """
        e = float(err)
        sd = float(np.sqrt(self.nov_var + 1e-12))
        if self.nov_count < 100 or sd <= 0:
            return 0.0
        return max(0.0, (e - self.nov_mean) / sd)

    def gate(self, err) -> float:
        """Residual authority in [gate_min, 1] from the novelty score.

        g = exp(-beta * novelty), floored. Deliberately a FIXED function
        with no learned parameters:
          * the env applies it outside the actor's graph, so a learned gate
            would receive no gradient without extra plumbing;
          * a declared inductive bias is easier to defend than a learned
            one that could itself overfit to the two training faults;
          * beta = 0 recovers the ungated policy exactly, so the ablation is
            a single flag.
        """
        if self.latent_dim == 0 or self.gate_beta <= 0:
            return 1.0
        if not self.nov_calibrated:
            # Uncalibrated statistics would gate on noise. Fail OPEN: the
            # policy behaves exactly as it does today rather than
            # unpredictably.
            return 1.0
        g = float(np.exp(-self.gate_beta * self.novelty(err)))
        return float(np.clip(g, self.gate_min, 1.0))

    @torch.no_grad()
    def eval_dyn_loss(self, ctx, mask, obs, act, y_target) -> float:
        """L_dyn WITHOUT a gradient step -- for held-out conditions.

        Reporting prediction error on a fault the encoder never trained on
        is the cleanest evidence that z generalizes rather than memorizes.
        """
        if self.latent_dim == 0:
            return float("nan")
        z = self.encoder(ctx, mask)
        pred = self.decoder(obs, act, z)
        return F.huber_loss(pred, self.norm_target(y_target)).item()

    # ------------------------------------------------------------------ #

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict() if self.decoder else None,
            "opt": self.opt.state_dict() if self.opt else None,
            "kind": self.kind,
            "latent_dim": self.latent_dim,
            "context_len": self.context_len,
            "tgt_mean": self.tgt_mean,
            "tgt_var": self.tgt_var,
            "tgt_count": self.tgt_count,
            "gate_beta": self.gate_beta,
            "gate_min": self.gate_min,
            "nov_mean": self.nov_mean,
            "nov_var": self.nov_var,
            "nov_count": self.nov_count,
            "nov_calibrated": self.nov_calibrated,
        }

    def load_state_dict(self, d, strict: bool = True):
        if d.get("kind", "none") != self.kind:
            raise ValueError(
                f"checkpoint context_encoder is '{d.get('kind')}' but this "
                f"module is '{self.kind}'. The latent means something "
                f"different; refusing to load.")
        self.encoder.load_state_dict(d["encoder"])
        if self.decoder is not None and d.get("decoder"):
            self.decoder.load_state_dict(d["decoder"])
        if self.opt is not None and d.get("opt"):
            self.opt.load_state_dict(d["opt"])
        self.tgt_mean = np.asarray(d["tgt_mean"])
        self.tgt_var = np.asarray(d["tgt_var"])
        self.tgt_count = d["tgt_count"]
        # Novelty stats travel WITH the checkpoint. Recomputing them at
        # evaluation time on whatever conditions happen to be evaluated
        # would silently redefine "familiar" to include the held-out fault.
        self.nov_mean = d.get("nov_mean", 0.0)
        self.nov_var = d.get("nov_var", 1.0)
        self.nov_count = d.get("nov_count", 1e-4)
        self.nov_calibrated = bool(d.get("nov_calibrated", False))
        if "gate_beta" in d:
            self.gate_beta = float(d["gate_beta"])
            self.gate_min = float(d["gate_min"])
