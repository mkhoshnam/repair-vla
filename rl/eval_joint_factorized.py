"""Evaluate joint-factorized capability SAC on explicit fault conditions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.build import TASK_MAX_STEPS, VLACfg, set_headless_env  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--conditions", nargs="+", default=["healthy", "0", "6", "2"])
    p.add_argument("--policies", nargs="+", choices=["zero", "ckpt"],
                   default=["zero", "ckpt"])
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--held_out", action="store_true")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--n_eval_states", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--outdir", default="rollouts_joint_factorized")
    return p.parse_args()


def main():
    args = parse_args()
    set_headless_env()

    import torch
    from experiments.robot.robot_utils import set_seed_everywhere
    from faults.multi_fault import FaultSpec
    from rl.build import build_shared
    from rl.factorized_sac import FactorizedSACAgent
    from rl.joint_factorized_encoder import (
        JointFactorizedCapabilityModule, build_left_padded_history,
    )
    from rl.residual_env import ResidualCfg
    from rl.sac import RunningNorm

    set_seed_everywhere(args.seed)
    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if blob.get("method") != "joint_factorized_kinematic_capability_sac":
        raise SystemExit(f"wrong checkpoint method: {blob.get('method')}")

    train_names = {FaultSpec.parse(t).name for t in blob["joint_pool"]}
    specs = []
    seen = set()
    for tok in args.conditions:
        sp = FaultSpec.parse(tok)
        if sp.name not in seen:
            specs.append(sp)
            seen.add(sp.name)

    ckargs = blob["args"]
    cfg = VLACfg(task_id=args.task_id, seed=args.seed)
    res_cfg = ResidualCfg(
        residual_scale=float(blob["residual_scale"]),
        history_len=int(ckargs.get("history_len", 8)),
        include_jacobian_obs=True,
        w_residual=0.0,
        gamma=float(ckargs.get("gamma", 0.99)),
        max_steps=TASK_MAX_STEPS[cfg.task_suite_name],
        num_steps_wait=cfg.num_steps_wait,
        seed=args.seed,
    )
    renv, train_ids, eval_ids, meta = build_shared(
        cfg, res_cfg, joint_pool=tuple(specs), include_healthy=False,
        n_eval_states=args.n_eval_states, seed=args.seed,
        context_include_time=False,
    )

    cap_state = blob["capability"]
    cap = JointFactorizedCapabilityModule(
        obs_dim=meta["obs_dim"], act_dim=meta["act_dim"],
        token_dim=int(cap_state["token_dim"]),
        context_len=int(cap_state["context_len"]),
        temporal_hidden=int(ckargs.get("temporal_hidden", 128)),
        cap_dim=int(cap_state["cap_dim"]), z_dim=int(cap_state["z_dim"]),
        transformer_layers=int(ckargs.get("transformer_layers", 2)),
        transformer_heads=int(ckargs.get("transformer_heads", 4)),
        transformer_ffn=int(ckargs.get("transformer_ffn", 256)),
        lr=float(ckargs.get("encoder_lr", 1e-4)),
        lambda_joint=float(cap_state.get("lambda_joint", 1.0)),
        lambda_eef=float(cap_state.get("lambda_eef", 1.0)),
        lambda_kin=float(cap_state.get("lambda_kin", 0.25)),
        device=args.device,
    )
    cap.load_checkpoint_state(cap_state, load_optimizer=False)
    cap.eval()

    agent = FactorizedSACAgent(
        obs_dim=meta["obs_dim"], z_dim=cap.latent_dim, act_dim=meta["act_dim"],
        device=args.device, hidden=int(ckargs.get("hidden", 256)),
        lr=float(ckargs.get("lr", 3e-4)), gamma=float(ckargs.get("gamma", 0.99)),
        tau=float(ckargs.get("tau", 0.005)),
        alpha_init=float(ckargs.get("alpha_init", 0.01)),
        log_std_init=float(ckargs.get("log_std_init", -1.0)),
    )
    agent.load_state_dict(blob["agent"], load_optimizers=False)
    agent.actor.eval()

    normalizer = RunningNorm(meta["obs_dim"])
    normalizer.load_state_dict(blob["obs_norm"])

    ids = list(eval_ids) if args.held_out else list(range(meta["n_init_states"]))
    ids = ids[:args.n_episodes]
    K = cap.context_len
    ctx_dim = meta["joint_ctx_dim"]
    zero = np.zeros(meta["act_dim"], dtype=np.float32)

    rows, table = [], {}
    for spec in renv.faults.specs:
        table[spec.name] = {}
        for policy in args.policies:
            succ, worst, z_norms = 0, 0.0, []
            for epi, init_id in enumerate(ids):
                obs = renv.reset(init_id=int(init_id), force_fault=spec)
                renv.faults.assert_exactly_one_lock(renv.env)
                hist = deque(maxlen=K)
                done = False
                while not done:
                    if policy == "zero":
                        a = zero
                        zn = 0.0
                    else:
                        ctx, mask = build_left_padded_history(hist, K, ctx_dim)
                        z = cap.encode_numpy(ctx, mask, obs)
                        zn = float(np.linalg.norm(z))
                        a = agent.act(normalizer(obs).astype(np.float32), z,
                                      deterministic=True)
                    obs, _, term, trunc, info = renv.step(a, gate=1.0)
                    hist.append(renv.last_joint_context.copy())
                    z_norms.append(zn)
                    worst = max(worst, info["lock_drift"])
                    done = term or trunc
                succ += int(info["success"])
                rows.append({
                    "fault": spec.name, "policy": policy, "episode": epi,
                    "init_id": int(init_id), "success": int(info["success"]),
                    "steps": info["t"],
                    "seen_in_training": int(spec.name in train_names),
                    "mean_z_norm": float(np.mean(z_norms)) if z_norms else 0.0,
                    "max_lock_drift_rad": worst,
                })
            table[spec.name][policy] = {
                "success": succ, "n": len(ids), "rate": succ / max(1, len(ids)),
                "seen_in_training": spec.name in train_names,
                "worst_lock_drift_rad": worst,
            }

    out = Path(args.outdir) / ("heldout" if args.held_out else "screen")
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / "per_episode.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    summary = {
        "ckpt": args.ckpt,
        "method": blob["method"],
        "trained_pool": sorted(train_names),
        "conditions": [s.name for s in renv.faults.specs],
        "heldout_joint_indices": blob.get("heldout_joints", []),
        "init_ids": [int(i) for i in ids],
        "table": table,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print(f"{'condition':>14} | " + " | ".join(f"{p:>10}" for p in args.policies) + " | seen?")
    print("-" * 72)
    for name, per in table.items():
        cells = " | ".join(
            f"{per[p]['success']}/{per[p]['n']}={per[p]['rate']:.0%}".rjust(10)
            for p in args.policies
        )
        tag = "trained" if name in train_names else "UNSEEN"
        print(f"{name:>14} | {cells} | {tag}")
    print("=" * 72)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
