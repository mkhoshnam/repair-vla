"""Train ONE Joint-Factorized Capability SAC policy across multiple tasks.

Default experiment:
  task A: libero_spatial:0
  task B: libero_goal:6
  task sampling: 50 / 50 per episode
  per-task fault sampling: j0 45%, j6 45%, healthy 10%
  j2 is held out globally from EVERY task and is never used for training,
  checkpoint selection, or online evaluation.

The trainer can either preserve the legacy suite-specific VLA path or load ONE
frozen multi-suite OpenVLA-OFT checkpoint once and reuse it across every task.
The JFCRL capability encoder + SAC policy remain the only shared trainable
controller. Task/fault identities are replay/logging metadata only; they are
never concatenated to the learner input.
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
from rl.build import set_headless_env  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["libero_spatial:0", "libero_goal:6"])
    p.add_argument("--task_probs", type=float, nargs="+", default=[0.5, 0.5])
    p.add_argument("--joint_pool", nargs="+", default=["0", "6", "healthy"])
    p.add_argument("--fault_probs", type=float, nargs="+", default=[0.45, 0.45, 0.10])
    p.add_argument("--heldout_conditions", nargs="+", default=["2"])
    p.add_argument(
        "--shared_vla_checkpoint", default=None,
        help=("If set, load this frozen OpenVLA-OFT stack ONCE and share it "
              "across all tasks. Per-task unnorm keys/instructions remain distinct."),
    )
    p.add_argument("--n_eval_states", type=int, default=10)
    p.add_argument("--fault_block", type=int, default=1)

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
    p.add_argument("--encoder_q_weight", type=float, default=0.05)

    p.add_argument("--residual_scale", type=float, default=0.1)
    p.add_argument("--history_len", type=int, default=8)
    p.add_argument("--w_residual", type=float, default=0.0)
    # Set total_steps ~= 50k * number_of_tasks to preserve the successful
    # single-task exposure scale (e.g. 5 tasks -> 250k, 10 tasks -> 500k).
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
    p.add_argument("--curriculum", default=None,
                   help="held-out-blind per-task curriculum JSON")
    p.add_argument("--arm", choices=["none", "gate"], default="none")
    p.add_argument("--sampler_mode", choices=["equal", "generation", "uniform"],
                   default="equal")
    p.add_argument("--gate_hidden", type=int, default=64)
    p.add_argument("--gate_bias_init", type=float, default=4.0)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--gate_no_jacobian", action="store_true")

    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--logdir", default="runs/jfcrl_multitask_spatial0_goal6_seed7")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def validate_probs(xs, n, name):
    if len(xs) != n:
        raise SystemExit(f"{name} needs {n} values, got {len(xs)}")
    if any(x < 0 for x in xs) or abs(sum(xs) - 1.0) > 1e-6:
        raise SystemExit(f"{name} must be non-negative and sum to 1; got {xs}")


def online_history(cap, hist, obs, K, ctx_dim):
    from rl.joint_factorized_encoder import build_left_padded_history
    ctx, mask = build_left_padded_history(hist, K, ctx_dim)
    return cap.encode_numpy(ctx, mask, obs)


def save_checkpoint(path, *, agent, cap, normalizer, args, bundles, step,
                    heldout_joints, strata, transition_counts, per_task_pools=None):
    import torch
    total = max(1, sum(transition_counts.values()))
    torch.save({
        "method": "joint_factorized_kinematic_capability_sac_multitask",
        "version": 3,
        "agent": agent.state_dict(),
        "capability": cap.checkpoint_state(),
        "obs_norm": normalizer.state_dict(),
        "args": vars(args),
        "step": int(step),
        "tasks": [b.spec.to_dict() for b in bundles],
        "task_meta": [b.meta for b in bundles],
        "task_probs": list(args.task_probs),
        "shared_vla_checkpoint": args.shared_vla_checkpoint,
        "joint_pool": list(args.joint_pool),
        "fault_probs": list(args.fault_probs),
        "per_task_fault_pools": per_task_pools,
        "curriculum_path": args.curriculum,
        "sampler_mode": args.sampler_mode,
        "arm": args.arm,
        "heldout_conditions": list(args.heldout_conditions),
        "heldout_joints": sorted(heldout_joints),
        "residual_scale": float(args.residual_scale),
        "replay_strata": list(strata),
        "replay_fraction_by_task_fault": {
            k: float(v / total) for k, v in transition_counts.items()
        },
    }, path)


def main():
    args = parse_args()
    set_headless_env()

    import torch
    from experiments.robot.robot_utils import set_seed_everywhere
    from faults.multi_fault import FaultSpec, joint_indices
    from rl.factorized_sac import FactorizedSACAgent
    from rl.joint_factorized_encoder import JointFactorizedCapabilityModule
    from rl.multitask_build import (
        assert_compatible_bundles, build_shared_vla_handles,
        build_task_bundle, parse_task_spec,
    )
    from rl.residual_env import ResidualCfg
    from rl.sac import NStepReplayBuffer, RunningNorm
    from rl.jfcrl_v2_runtime import (
        generation_stratum_weights, install_gate, load_curriculum,
        sample_weighted, gate_stats,
    )

    set_seed_everywhere(args.seed)

    per_task_pools = None
    if args.curriculum:
        tasks, task_probs, per_task_pools, heldout = load_curriculum(args.curriculum)
        args.tasks = tasks
        args.task_probs = task_probs
        args.heldout_conditions = [heldout]
        # Global union exists only for clash checks / backward-compatible metadata.
        # Actual training pools and probabilities are per-task below.
        args.joint_pool = sorted(
            {j for v in per_task_pools.values() for j in v["joint_pool"]},
            key=lambda x: (x == "healthy", x),
        )
        args.fault_probs = [1.0 / len(args.joint_pool)] * len(args.joint_pool)

    if args.sampler_mode == "generation" and not per_task_pools:
        raise SystemExit("--sampler_mode generation requires --curriculum")

    specs = [parse_task_spec(x) for x in args.tasks]
    if len({s.key for s in specs}) != len(specs):
        raise SystemExit(f"duplicate task in --tasks: {args.tasks}")
    validate_probs(args.task_probs, len(specs), "--task_probs")
    validate_probs(args.fault_probs, len(args.joint_pool), "--fault_probs")
    if args.w_residual != 0.0:
        raise SystemExit("headline method is task-reward-only: keep --w_residual 0")

    heldout_joints = joint_indices(args.heldout_conditions)
    clash = heldout_joints & joint_indices(args.joint_pool)
    if clash:
        offenders = [t for t in args.joint_pool if FaultSpec.parse(t).joint_idx in clash]
        raise SystemExit(
            f"held-out JOINT(S) {sorted(clash)} appear in training via {offenders}"
        )

    logdir = Path(args.logdir)
    if logdir.exists() and any(logdir.glob("ckpt_*.pt")):
        raise SystemExit(f"{logdir} already contains checkpoints; refusing overwrite")
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.json").write_text(json.dumps(vars(args), indent=2))

    def res_cfg_factory(max_steps, cfg):
        return ResidualCfg(
            residual_scale=args.residual_scale,
            history_len=args.history_len,
            include_jacobian_obs=True,
            w_residual=args.w_residual,
            gamma=args.gamma,
            max_steps=max_steps,
            num_steps_wait=cfg.num_steps_wait,
            seed=args.seed,
        )

    shared_vla_handles = None
    if args.shared_vla_checkpoint:
        print(
            "\n=== loading ONE shared frozen VLA for all tasks ===\n"
            f"checkpoint: {args.shared_vla_checkpoint}"
        )
        shared_vla_handles = build_shared_vla_handles(
            args.shared_vla_checkpoint,
            seed=args.seed,
            representative_suite=specs[0].suite,
        )

    bundles = []
    model_ids = []
    for i, spec in enumerate(specs):
        print(f"\n=== building task {i+1}/{len(specs)}: {spec.key} ===")
        pool = per_task_pools[spec.key]["joint_pool"] if per_task_pools else args.joint_pool
        fprobs = per_task_pools[spec.key]["fault_probs"] if per_task_pools else args.fault_probs
        b = build_task_bundle(
            spec,
            residual_cfg_factory=res_cfg_factory,
            joint_pool=pool,
            fault_probs=fprobs,
            fault_block=args.fault_block,
            n_eval_states=args.n_eval_states,
            seed=args.seed + i * 1009,
            pretrained_checkpoint=args.shared_vla_checkpoint,
            shared_vla_handles=shared_vla_handles,
        )
        bundles.append(b)
        model_ids.append(id(b.renv.vla.model))
        print(json.dumps(b.meta, indent=2))
    assert_compatible_bundles(bundles)
    if args.shared_vla_checkpoint:
        if len(set(model_ids)) != 1:
            raise RuntimeError(
                f"shared-VLA invariant failed: expected one model object, got {model_ids}"
            )
        print(f"shared-VLA invariant verified: one model object for {len(bundles)} tasks")

    ref = bundles[0].meta
    obs_dim, act_dim = ref["obs_dim"], ref["act_dim"]
    K, joint_ctx_dim = args.context_len, ref["joint_ctx_dim"]

    cap = JointFactorizedCapabilityModule(
        obs_dim=obs_dim, act_dim=act_dim, token_dim=ref["joint_token_dim"],
        context_len=K, temporal_hidden=args.temporal_hidden,
        cap_dim=args.cap_dim, z_dim=args.z_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn=args.transformer_ffn,
        lr=args.encoder_lr, lambda_joint=args.lambda_joint,
        lambda_eef=args.lambda_eef, lambda_kin=args.lambda_kin,
        device=args.device,
    )
    agent = FactorizedSACAgent(
        obs_dim=obs_dim, z_dim=args.z_dim, act_dim=act_dim,
        device=args.device, hidden=args.hidden, lr=args.lr,
        gamma=args.gamma, tau=args.tau, alpha_init=args.alpha_init,
        zero_init_actor=True, log_std_init=args.log_std_init,
    )
    if args.arm == "gate":
        agent = install_gate(agent, obs_dim, args.z_dim, act_dim, args)
        print(f"[arm] capability-driven gate installed; init={args.gate_bias_init}")

    # Replay is stratified over TASK x FAULT, not only fault. These ids are
    # metadata only; they never enter the actor, critic, or capability encoder.
    strata = []
    stratum_id = {}
    for b in bundles:
        for fault_name in b.renv.faults.names:
            key = f"{b.spec.key}/{fault_name}"
            stratum_id[(b.spec.key, fault_name)] = len(strata)
            strata.append(key)

    print(json.dumps({
        "method": "joint_factorized_kinematic_capability_sac_multitask",
        "tasks": [b.spec.to_dict() for b in bundles],
        "task_probs": args.task_probs,
        "shared_vla_checkpoint": args.shared_vla_checkpoint,
        "per_task_fault_pools": {
            b.spec.key: {
                "faults": list(b.renv.faults.names),
                "fault_probs": (per_task_pools[b.spec.key]["fault_probs"]
                                if per_task_pools else list(args.fault_probs)),
            } for b in bundles
        },
        "replay_strata": strata,
        "heldout_joints": sorted(heldout_joints),
        "obs_dim": obs_dim,
        "joint_ctx_dim": joint_ctx_dim,
        "joint_token_dim": ref["joint_token_dim"],
        "z_dim": args.z_dim,
    }, indent=2))

    stratum_weights = None
    if args.sampler_mode == "generation":
        stratum_weights = generation_stratum_weights(
            stratum_id, args.tasks, args.task_probs, per_task_pools
        )
        print("[replay] generation-weighted TASK x FAULT sampling enabled")
    elif args.sampler_mode == "equal":
        print("[replay] equal-share TASK x FAULT sampling (legacy control)")
    else:
        print("[replay] uniform-over-transition sampling")

    with torch.no_grad():
        for zprobe in (torch.zeros(2, args.z_dim, device=args.device),
                       torch.randn(2, args.z_dim, device=args.device) * 5):
            oprobe = torch.randn(2, obs_dim, device=args.device)
            a0, _ = agent.actor(oprobe, zprobe, deterministic=True, with_logp=False)
            assert float(a0.abs().max()) < 1e-7
    print("zero-init verified: deterministic residual is exactly zero")

    if args.dry_run:
        zero = np.zeros(act_dim, dtype=np.float32)
        for b in bundles:
            print(f"\n--- dry run {b.spec.key}: {b.meta['task_description']} ---")
            for fault in b.renv.faults.specs:
                obs = b.renv.reset(init_id=int(b.train_ids[0]), force_fault=fault)
                b.renv.faults.assert_exactly_one_lock(b.renv.env)
                J = b.renv.arm_jacobian()
                _, _, term, trunc, info = b.renv.step(zero)
                b.renv.faults.assert_exactly_one_lock(b.renv.env)
                print(f"{fault.name:>10}: J={J.shape}, finite={np.isfinite(J).all()}, "
                      f"drift={info['lock_drift']:.2e}, term={term}, trunc={trunc}")
        print("multitask dry run passed")
        return

    buf = NStepReplayBuffer(
        obs_dim, act_dim, capacity=args.buffer_size, n_step=args.n_step,
        gamma=args.gamma, n_faults=len(strata), ctx_dim=joint_ctx_dim,
        context_len=K,
    )
    normalizer = RunningNorm(obs_dim)

    if args.wandb:
        import wandb
        wandb.init(project="vla-fault-residual", name=logdir.name,
                   config={**vars(args), "method": "joint_factorized_multitask"})

    succ = {k: deque(maxlen=50) for k in strata}
    n_eps = {k: 0 for k in strata}
    res_norm = {k: deque(maxlen=2000) for k in strata}
    transition_counts = {k: 0 for k in strata}
    task_episode_counts = {b.spec.key: 0 for b in bundles}
    task_transition_counts = {b.spec.key: 0 for b in bundles}
    metrics, cap_metrics = {}, {}
    hist = deque(maxlen=K)
    rng = np.random.default_rng(args.seed + 424242)
    sampled_window_counts = np.zeros(len(strata), dtype=np.int64)
    episode_uid = 0

    def choose_task():
        # Choose only at episode boundaries, but balance ENV STEP exposure,
        # not just episode counts. The two suites have different horizons and
        # success lengths, so naive 50/50 episode sampling can produce a very
        # uneven transition budget. For target probability p_i, minimizing
        # steps_i / p_i keeps long-run step fractions near p_i. Random tie
        # breaking avoids always starting with task 0.
        p = np.asarray(args.task_probs, dtype=float)
        scaled = np.asarray([task_transition_counts[x.spec.key] for x in bundles],
                            dtype=float) / np.maximum(p, 1e-12)
        m = scaled.min()
        candidates = np.flatnonzero(np.isclose(scaled, m))
        return int(rng.choice(candidates))

    task_i = choose_task()
    b = bundles[task_i]
    obs = b.renv.reset()
    b.renv.faults.assert_exactly_one_lock(b.renv.env)
    normalizer.update(obs)
    task_episode_counts[b.spec.key] += 1
    t0 = time.time()

    # Logging-only gate diagnostics.
    # Windowed averages over all visited states, split by physical condition
    # and by task x condition. These do NOT affect training.
    gate_cond_sum = {}
    gate_cond_count = {}
    gate_cell_sum = {}
    gate_cell_count = {}

    for step in range(1, args.total_steps + 1):
        cur_fault = b.renv.faults.active.name
        global_key = f"{b.spec.key}/{cur_fault}"
        fid = stratum_id[(b.spec.key, cur_fault)]

        z_np = online_history(cap, hist, obs, K, joint_ctx_dim)

        # Logging only: measure the deterministic capability gate on the
        # current state before action sampling.
        if args.arm == "gate":
            with torch.no_grad():
                z_gate = torch.as_tensor(
                    z_np, dtype=torch.float32, device=agent.device
                )[None]
                g_val = float(agent.actor.gate(z_gate).item())

            gate_cond_sum[cur_fault] = gate_cond_sum.get(cur_fault, 0.0) + g_val
            gate_cond_count[cur_fault] = gate_cond_count.get(cur_fault, 0) + 1

            gate_cell_sum[global_key] = gate_cell_sum.get(global_key, 0.0) + g_val
            gate_cell_count[global_key] = gate_cell_count.get(global_key, 0) + 1

        obs_n = normalizer(obs).astype(np.float32)
        a = agent.act(obs_n, z_np, deterministic=False)
        next_obs, r, terminated, truncated, info = b.renv.step(a, gate=1.0)
        normalizer.update(next_obs)

        s_now = buf.push_context(
            b.renv.last_joint_context, episode_uid, b.renv.t - 1
        )
        buf.add(
            obs.astype(np.float32), a, r, next_obs.astype(np.float32),
            terminated, truncated, fault_id=fid,
            s_obs=s_now, s_next=s_now,
            context_episode_id=episode_uid,
            context_t=b.renv.t - 1,
            dyn_target=b.renv.last_joint_dyn_target,
        )
        cap.update_context_stats(b.renv.last_joint_context)
        cap.update_target_stats(b.renv.last_joint_dyn_target[None, :])
        hist.append(b.renv.last_joint_context.copy())

        obs = next_obs
        res_norm[global_key].append(info["residual_norm"])
        transition_counts[global_key] += 1
        task_transition_counts[b.spec.key] += 1

        if terminated or truncated:
            succ[global_key].append(int(info["success"]))
            n_eps[global_key] += 1
            drift = b.renv.faults.monitor.max_drift if b.renv.faults.monitor else 0.0
            if drift > 1e-2:
                raise SystemExit(
                    f"lock infrastructure failed on {global_key}: drift={drift:.3e}"
                )
            buf.end_episode()
            episode_uid += 1
            task_i = choose_task()
            b = bundles[task_i]
            obs = b.renv.reset()
            b.renv.faults.assert_exactly_one_lock(b.renv.env)
            normalizer.update(obs)
            hist.clear()
            task_episode_counts[b.spec.key] += 1

        if step > args.start_steps and buf.size >= args.batch_size:
            for _ in range(args.utd):
                batch = sample_weighted(
                    buf, args.batch_size, agent.device, obs_norm=normalizer,
                    weights=stratum_weights, mode=args.sampler_mode, rng=rng,
                )
                sampled_window_counts += np.bincount(
                    buf.fault_id[batch["idx"]], minlength=len(strata)
                )[:len(strata)]
                cap_metrics = cap.update(
                    batch["ctx"], batch["mask"], batch["obs"],
                    batch["raw_obs"], batch["act"], batch["dyn_target"],
                )
                z = cap.encode_for_policy(
                    batch["ctx"], batch["mask"], batch["raw_obs"], detach=False
                )
                with torch.no_grad():
                    nz = cap.encode_for_policy(
                        batch["next_ctx"], batch["next_mask"],
                        batch["raw_next_obs"], detach=True,
                    )
                metrics = agent.update(
                    batch, z, nz,
                    encoder_module=cap,
                    encoder_optimizer=cap.opt,
                    encoder_q_weight=args.encoder_q_weight,
                )

        if step % args.log_every == 0:
            log = {
                "step": step,
                "throughput/env_steps_per_s": step / max(1e-6, time.time() - t0),
                "buffer/size": buf.size,
                "fault/env_rebuilds": sum(x.renv.faults.stats()["fault/n_env_rebuilds"] for x in bundles),
                **metrics, **cap_metrics,
            }
            log.update(gate_stats(agent, z_np))

            # Windowed gate averages by condition and task x condition.
            if args.arm == "gate":
                for cond, total in sorted(gate_cond_sum.items()):
                    n = gate_cond_count.get(cond, 0)
                    if n:
                        log[f"gate/cond_{cond}"] = total / n

                for cell, total in sorted(gate_cell_sum.items()):
                    n = gate_cell_count.get(cell, 0)
                    if n:
                        safe_cell = cell.replace(":", "_").replace("/", "_")
                        log[f"gate/cell_{safe_cell}"] = total / n

                # Start a fresh diagnostic window after every logging interval.
                gate_cond_sum.clear()
                gate_cond_count.clear()
                gate_cell_sum.clear()
                gate_cell_count.clear()

            fracs = buf.fault_fractions()
            sampled_total = int(sampled_window_counts.sum())
            for ti, task_b in enumerate(bundles):
                log[f"task/episodes_{task_b.spec.key}"] = task_episode_counts[task_b.spec.key]
                log[f"task/transitions_{task_b.spec.key}"] = task_transition_counts[task_b.spec.key]
                log[f"task/transition_frac_{task_b.spec.key}"] = (
                    task_transition_counts[task_b.spec.key] / max(1, step)
                )
                for fname in task_b.renv.faults.names:
                    key = f"{task_b.spec.key}/{fname}"
                    safe = key.replace(":", "_").replace("/", "_")
                    sid = stratum_id[(task_b.spec.key, fname)]
                    log[f"train/success_{safe}"] = (
                        float(np.mean(succ[key])) if succ[key] else float("nan")
                    )
                    log[f"train/episodes_{safe}"] = n_eps[key]
                    log[f"residual/norm_{safe}"] = (
                        float(np.mean(res_norm[key])) if res_norm[key] else float("nan")
                    )
                    log[f"buffer/frac_{safe}"] = fracs.get(sid, 0.0)
                    if sampled_total:
                        log[f"sampled/frac_{safe}"] = float(
                            sampled_window_counts[sid] / sampled_total
                        )
            sampled_window_counts.fill(0)
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
            save_checkpoint(
                logdir / f"ckpt_{step}.pt", agent=agent, cap=cap,
                normalizer=normalizer, args=args, bundles=bundles, step=step,
                heldout_joints=heldout_joints, strata=strata,
                transition_counts=transition_counts, per_task_pools=per_task_pools,
            )

    print(f"done. checkpoints in {logdir}")


if __name__ == "__main__":
    main()
