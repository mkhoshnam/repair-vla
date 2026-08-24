"""Train Joint-Factorized Kinematic Capability SAC.

Headline pilot:
  * frozen OpenVLA-OFT
  * task reward only
  * train faults: j0, j6, plus a small healthy fraction
  * entire j2 joint held out (all j2 variants forbidden)
  * no fault id / lock angle / fault family in learner input
  * joint-factorized shared temporal encoder + current Jacobian geometry
  * cross-joint Transformer + FiLM-conditioned SAC

This is intentionally a separate trainer so the validated GRU/shared SAC stack
and its checkpoints remain untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.build import TASK_MAX_STEPS, VLACfg, set_headless_env  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--joint_pool", nargs="+", default=["0", "6", "healthy"])
    p.add_argument("--fault_probs", type=float, nargs="+", default=[0.45, 0.45, 0.10])
    p.add_argument("--heldout_conditions", nargs="+", default=["2"])
    p.add_argument("--n_eval_states", type=int, default=10)
    p.add_argument("--fault_block", type=int, default=1)

    # Capability architecture.
    p.add_argument("--context_len", type=int, default=16)
    p.add_argument("--temporal_hidden", type=int, default=128)
    p.add_argument("--cap_dim", type=int, default=32)
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--transformer_layers", type=int, default=2)
    p.add_argument("--transformer_heads", type=int, default=4)
    p.add_argument("--transformer_ffn", type=int, default=256)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--lambda_joint", type=float, default=1.0)
    p.add_argument("--lambda_eef", type=float, default=1.0)
    p.add_argument("--lambda_kin", type=float, default=0.25)
    p.add_argument("--encoder_q_weight", type=float, default=0.05,
                   help="Critic-gradient weight into capability encoder. Actor "
                        "gradients never enter the encoder.")

    # Residual + SAC: preserve validated values.
    p.add_argument("--residual_scale", type=float, default=0.1)
    p.add_argument("--history_len", type=int, default=8)
    p.add_argument("--w_residual", type=float, default=0.0)
    p.add_argument("--total_steps", type=int, default=100_000)
    p.add_argument("--start_steps", type=int, default=2_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--utd", type=int, default=4)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=400_000)
    p.add_argument("--alpha_init", type=float, default=0.01)
    p.add_argument("--log_std_init", type=float, default=-1.0)
    p.add_argument("--stratified_replay", action="store_true", default=True)

    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--logdir", type=str, default="runs/task0_joint_factorized_j0_j6")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="One forced pass through training conditions; checks "
                        "fault compilation and Jacobian extraction only.")
    return p.parse_args()


def online_history(cap, hist, obs, K, ctx_dim):
    from rl.joint_factorized_encoder import build_left_padded_history

    ctx, mask = build_left_padded_history(hist, K, ctx_dim)
    return cap.encode_numpy(ctx, mask, obs)


def save_checkpoint(path, *, agent, cap, normalizer, args, meta, step,
                    names, cfg, heldout_joints, replay_fracs):
    import torch

    torch.save({
        "method": "joint_factorized_kinematic_capability_sac",
        "version": 1,
        "agent": agent.state_dict(),
        "capability": cap.checkpoint_state(),
        "obs_norm": normalizer.state_dict(),
        "args": vars(args),
        "meta": meta,
        "step": int(step),
        "joint_pool": list(args.joint_pool),
        "fault_probs": list(args.fault_probs),
        "fault_names": list(names),
        "heldout_conditions": list(args.heldout_conditions),
        "heldout_joints": sorted(heldout_joints),
        "residual_scale": float(args.residual_scale),
        "pretrained_checkpoint": cfg.pretrained_checkpoint,
        "replay_fraction_by_fault": replay_fracs,
    }, path)


def main():
    args = parse_args()
    set_headless_env()

    import torch
    from experiments.robot.robot_utils import set_seed_everywhere
    from faults.multi_fault import FaultSpec, joint_indices
    from rl.build import build_shared
    from rl.factorized_sac import FactorizedSACAgent
    from rl.joint_factorized_encoder import JointFactorizedCapabilityModule
    from rl.residual_env import ResidualCfg
    from rl.sac import NStepReplayBuffer, RunningNorm

    set_seed_everywhere(args.seed)

    if len(args.fault_probs) != len(args.joint_pool):
        raise SystemExit("--fault_probs must have one value per --joint_pool token")
    if abs(sum(args.fault_probs) - 1.0) > 1e-6:
        raise SystemExit(f"--fault_probs must sum to 1, got {sum(args.fault_probs)}")
    if args.w_residual != 0.0:
        raise SystemExit("headline method is task-reward-only: keep --w_residual 0")

    heldout_joints = joint_indices(args.heldout_conditions)
    clash = heldout_joints & joint_indices(args.joint_pool)
    if clash:
        offenders = [t for t in args.joint_pool
                     if FaultSpec.parse(t).joint_idx in clash]
        raise SystemExit(
            f"held-out JOINT(S) {sorted(clash)} appear in training via {offenders}"
        )

    logdir = Path(args.logdir)
    if logdir.exists() and any(logdir.glob("ckpt_*.pt")):
        raise SystemExit(f"{logdir} already contains checkpoints; refusing overwrite")
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.json").write_text(json.dumps(vars(args), indent=2))

    cfg = VLACfg(task_id=args.task_id, seed=args.seed)
    res_cfg = ResidualCfg(
        residual_scale=args.residual_scale,
        history_len=args.history_len,
        include_jacobian_obs=True,
        w_residual=args.w_residual,
        gamma=args.gamma,
        max_steps=TASK_MAX_STEPS[cfg.task_suite_name],
        num_steps_wait=cfg.num_steps_wait,
        seed=args.seed,
    )

    renv, train_ids, eval_ids, meta = build_shared(
        cfg, res_cfg,
        joint_pool=tuple(args.joint_pool),
        fault_probs=tuple(args.fault_probs),
        include_healthy=False,
        fault_block=args.fault_block,
        n_eval_states=args.n_eval_states,
        seed=args.seed,
        context_include_time=False,
    )
    names = renv.faults.names
    if not renv.faults.env_factory:
        raise SystemExit("build_shared has no env_factory; fault switching is unsafe")
    if not meta.get("jacobian_in_obs", False):
        raise SystemExit("build/residual_env mismatch: Jacobian is not in observation")
    if meta.get("joint_ctx_dim") != 7 * meta.get("joint_token_dim", 0):
        raise SystemExit(f"invalid joint context metadata: {meta}")

    obs_dim, act_dim = meta["obs_dim"], meta["act_dim"]
    K, joint_ctx_dim = args.context_len, meta["joint_ctx_dim"]

    cap = JointFactorizedCapabilityModule(
        obs_dim=obs_dim,
        act_dim=act_dim,
        token_dim=meta["joint_token_dim"],
        context_len=K,
        temporal_hidden=args.temporal_hidden,
        cap_dim=args.cap_dim,
        z_dim=args.z_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn=args.transformer_ffn,
        lr=args.encoder_lr,
        lambda_joint=args.lambda_joint,
        lambda_eef=args.lambda_eef,
        lambda_kin=args.lambda_kin,
        device=args.device,
    )
    agent = FactorizedSACAgent(
        obs_dim=obs_dim, z_dim=args.z_dim, act_dim=act_dim,
        device=args.device, hidden=args.hidden, lr=args.lr,
        gamma=args.gamma, tau=args.tau, alpha_init=args.alpha_init,
        zero_init_actor=True, log_std_init=args.log_std_init,
    )

    print(json.dumps({
        "method": "joint_factorized_kinematic_capability_sac",
        "pool": names,
        "probs": [float(p) for p in renv.faults.probs],
        "heldout_joints": sorted(heldout_joints),
        "n_train_states": len(train_ids),
        "obs_dim": obs_dim,
        "joint_ctx_dim": joint_ctx_dim,
        "joint_token_dim": meta["joint_token_dim"],
        "z_dim": args.z_dim,
        "encoder_q_weight": args.encoder_q_weight,
        "checkpoint": cfg.pretrained_checkpoint,
    }, indent=2))

    # Deterministic residual must be exactly zero at initialization for any z.
    with torch.no_grad():
        for zprobe in (torch.zeros(2, args.z_dim, device=args.device),
                       torch.randn(2, args.z_dim, device=args.device) * 5):
            oprobe = torch.randn(2, obs_dim, device=args.device)
            a0, _ = agent.actor(oprobe, zprobe, deterministic=True, with_logp=False)
            assert float(a0.abs().max()) < 1e-7
    print("zero-init verified: deterministic residual is exactly zero for arbitrary capability")

    if args.dry_run:
        zact = np.zeros(act_dim, dtype=np.float32)
        seq = list(renv.faults.specs) + [renv.faults.specs[0]]
        for sp in seq:
            obs = renv.reset(init_id=int(train_ids[0]), force_fault=sp)
            renv.faults.assert_exactly_one_lock(renv.env)
            J = renv.arm_jacobian()
            print(f"{sp.name:>10}: J shape={J.shape}, finite={np.isfinite(J).all()}, "
                  f"||J||F={np.linalg.norm(J):.3f}")
            _, _, term, trunc, info = renv.step(zact)
            renv.faults.assert_exactly_one_lock(renv.env)
            print(f"           step gate={info['gate']:.1f} drift={info['lock_drift']:.2e}")
        print("dry run passed")
        return

    buf = NStepReplayBuffer(
        obs_dim, act_dim, capacity=args.buffer_size, n_step=args.n_step,
        gamma=args.gamma, n_faults=len(names), ctx_dim=joint_ctx_dim,
        context_len=K,
    )
    normalizer = RunningNorm(obs_dim)

    if args.wandb:
        import wandb
        wandb.init(project="vla-fault-residual", name=logdir.name,
                   config={**vars(args), "method": "joint_factorized"})

    succ = {n: deque(maxlen=50) for n in names}
    n_eps = {n: 0 for n in names}
    res_by_fault = {n: deque(maxlen=2000) for n in names}
    trans_by_fault = {n: 0 for n in names}
    metrics, cap_metrics = {}, {}
    hist = deque(maxlen=K)

    obs = renv.reset()
    renv.faults.assert_exactly_one_lock(renv.env)
    normalizer.update(obs)
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        cur = renv.faults.active.name
        fid = renv.faults.index_of(cur)
        z_np = online_history(cap, hist, obs, K, joint_ctx_dim)
        obs_n = normalizer(obs).astype(np.float32)
        # Same actor-centred warmup as the validated SAC: stochastic but mean zero.
        a = agent.act(obs_n, z_np, deterministic=False)

        next_obs, r, terminated, truncated, info = renv.step(a, gate=1.0)
        normalizer.update(next_obs)

        s_now = buf.push_context(
            renv.last_joint_context, renv.episode_id, renv.t - 1
        )
        buf.add(
            obs.astype(np.float32), a, r, next_obs.astype(np.float32),
            terminated, truncated, fault_id=fid,
            s_obs=s_now, s_next=s_now,
            context_episode_id=renv.episode_id,
            context_t=renv.t - 1,
            dyn_target=renv.last_joint_dyn_target,
        )
        cap.update_context_stats(renv.last_joint_context)
        cap.update_target_stats(renv.last_joint_dyn_target[None, :])
        hist.append(renv.last_joint_context.copy())

        obs = next_obs
        res_by_fault[cur].append(info["residual_norm"])
        trans_by_fault[cur] += 1

        if terminated or truncated:
            succ[cur].append(int(info["success"]))
            n_eps[cur] += 1
            drift = renv.faults.monitor.max_drift if renv.faults.monitor else 0.0
            if drift > 1e-2:
                raise SystemExit(
                    f"lock infrastructure failed on {cur}: drift={drift:.3e}"
                )
            buf.end_episode()
            obs = renv.reset()
            renv.faults.assert_exactly_one_lock(renv.env)
            normalizer.update(obs)
            hist.clear()

        if step > args.start_steps and buf.size >= args.batch_size:
            for _ in range(args.utd):
                batch = buf.sample(
                    args.batch_size, agent.device, obs_norm=normalizer,
                    stratified=args.stratified_replay,
                )
                # 1) self-supervised actuator / kinematic representation step.
                cap_metrics = cap.update(
                    batch["ctx"], batch["mask"], batch["obs"],
                    batch["raw_obs"], batch["act"], batch["dyn_target"],
                )
                # 2) fresh capability after representation update. Current z
                # keeps graph so the critic can send a SMALL control-aware
                # gradient into the encoder; next z is a target and detached.
                z = cap.encode_for_policy(
                    batch["ctx"], batch["mask"], batch["raw_obs"], detach=False
                )
                with torch.no_grad():
                    nz = cap.encode_for_policy(
                        batch["next_ctx"], batch["next_mask"],
                        batch["raw_next_obs"], detach=True
                    )
                metrics = agent.update(
                    batch, z, nz,
                    encoder_module=cap,
                    encoder_optimizer=cap.opt,
                    encoder_q_weight=args.encoder_q_weight,
                )

        if step % args.log_every == 0:
            sps = step / max(1e-6, time.time() - t0)
            log = {
                "step": step,
                "throughput/env_steps_per_s": sps,
                "buffer/size": buf.size,
                "fault/env_rebuilds": renv.faults.stats()["fault/n_env_rebuilds"],
                **metrics, **cap_metrics,
            }
            fracs = buf.fault_fractions()
            for n in names:
                log[f"train/success_{n}"] = (
                    float(np.mean(succ[n])) if succ[n] else float("nan")
                )
                log[f"train/episodes_{n}"] = n_eps[n]
                log[f"residual/norm_{n}"] = (
                    float(np.mean(res_by_fault[n])) if res_by_fault[n]
                    else float("nan")
                )
                log[f"buffer/frac_{n}"] = fracs.get(renv.faults.index_of(n), 0.0)
            print(" | ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in log.items()
            ))
            if args.wandb:
                import wandb
                wandb.log(log)
            q = metrics.get("sac/q_mean")
            if q is not None and not np.isfinite(q):
                raise SystemExit("NaN/Inf critic")

        if step % args.ckpt_every == 0 or step == args.total_steps:
            replay_fracs = {
                n: trans_by_fault[n] / max(1, step) for n in names
            }
            save_checkpoint(
                logdir / f"ckpt_{step}.pt", agent=agent, cap=cap,
                normalizer=normalizer, args=args, meta=meta, step=step,
                names=names, cfg=cfg, heldout_joints=heldout_joints,
                replay_fracs=replay_fracs,
            )

    print(f"done. checkpoints in {logdir}")


if __name__ == "__main__":
    main()
