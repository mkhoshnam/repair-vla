"""Deterministic held-out evaluation for a multitask JFCRL checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.build import set_headless_env  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--conditions", nargs="+", default=["2"])
    p.add_argument("--policies", nargs="+", choices=["zero", "ckpt"],
                   default=["zero", "ckpt"])
    p.add_argument("--held_out", action="store_true")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--n_eval_states", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--outdir", default="rollouts_jfcrl_multitask")
    p.add_argument("--curriculum", default=None,
                   help="optional consistency check against checkpoint task list")
    return p.parse_args()


def main():
    args = parse_args()
    set_headless_env()

    import torch
    from experiments.robot.robot_utils import set_seed_everywhere
    from faults.multi_fault import FaultSpec
    from rl.factorized_sac import FactorizedSACAgent
    from rl.joint_factorized_encoder import (
        JointFactorizedCapabilityModule, build_left_padded_history,
    )
    from rl.multitask_build import (
        build_shared_vla_handles, build_task_bundle, parse_task_spec,
    )
    from rl.residual_env import ResidualCfg
    from rl.sac import RunningNorm

    set_seed_everywhere(args.seed)
    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if blob.get("method") != "joint_factorized_kinematic_capability_sac_multitask":
        raise SystemExit(f"wrong checkpoint method: {blob.get('method')}")

    ckargs = blob["args"]
    task_specs = [parse_task_spec(f"{t['suite']}:{t['task_id']}") for t in blob["tasks"]]
    if args.curriculum:
        from rl.jfcrl_v2_runtime import load_curriculum
        cur_tasks, _, _, cur_heldout = load_curriculum(args.curriculum)
        ck_tasks = [s.key for s in task_specs]
        if cur_tasks != ck_tasks:
            raise SystemExit(
                f"curriculum/checkpoint task mismatch:\n  curriculum={cur_tasks}\n  checkpoint={ck_tasks}"
            )
        held = {str(x) for x in blob.get("heldout_conditions", ckargs.get("heldout_conditions", []))}
        if held and str(cur_heldout) not in held:
            raise SystemExit(f"curriculum heldout j{cur_heldout} != checkpoint heldout {sorted(held)}")
    cond_specs = []
    seen_cond = set()
    for tok in args.conditions:
        sp = FaultSpec.parse(tok)
        if sp.name not in seen_cond:
            cond_specs.append(sp); seen_cond.add(sp.name)

    def res_cfg_factory(max_steps, cfg):
        return ResidualCfg(
            residual_scale=float(blob["residual_scale"]),
            history_len=int(ckargs.get("history_len", 8)),
            include_jacobian_obs=True, w_residual=0.0,
            gamma=float(ckargs.get("gamma", 0.99)),
            max_steps=max_steps, num_steps_wait=cfg.num_steps_wait,
            seed=args.seed,
        )

    # Reconstruct the frozen VLA path exactly as recorded by training. New
    # shared-VLA checkpoints load ONE model once; old checkpoints stay backward
    # compatible and use their original suite-specific construction.
    shared_ckpt = blob.get("shared_vla_checkpoint", ckargs.get("shared_vla_checkpoint"))
    shared_vla_handles = None
    if shared_ckpt:
        print(
            "\n=== loading ONE shared frozen VLA for evaluation ===\n"
            f"checkpoint: {shared_ckpt}"
        )
        shared_vla_handles = build_shared_vla_handles(
            shared_ckpt, seed=args.seed,
            representative_suite=task_specs[0].suite,
        )

    bundles = []
    model_ids = []
    for i, spec in enumerate(task_specs):
        b = build_task_bundle(
            spec, residual_cfg_factory=res_cfg_factory,
            joint_pool=[s.name for s in cond_specs],
            fault_probs=[1.0 / len(cond_specs)] * len(cond_specs),
            fault_block=1, n_eval_states=args.n_eval_states,
            seed=args.seed + i * 1009,
            pretrained_checkpoint=shared_ckpt,
            shared_vla_handles=shared_vla_handles,
        )
        bundles.append(b)
        model_ids.append(id(b.renv.vla.model))
    if shared_ckpt and len(set(model_ids)) != 1:
        raise RuntimeError(
            f"shared-VLA invariant failed during eval: expected one model object, got {model_ids}"
        )
    if shared_ckpt:
        print(f"shared-VLA invariant verified: one model object for {len(bundles)} tasks")

    ref = bundles[0].meta
    cap_state = blob["capability"]
    cap = JointFactorizedCapabilityModule(
        obs_dim=ref["obs_dim"], act_dim=ref["act_dim"],
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
        obs_dim=ref["obs_dim"], z_dim=cap.latent_dim, act_dim=ref["act_dim"],
        device=args.device, hidden=int(ckargs.get("hidden", 256)),
        lr=float(ckargs.get("lr", 3e-4)), gamma=float(ckargs.get("gamma", 0.99)),
        tau=float(ckargs.get("tau", 0.005)),
        alpha_init=float(ckargs.get("alpha_init", 0.01)),
        log_std_init=float(ckargs.get("log_std_init", -1.0)),
    )
    arm = blob.get("arm", ckargs.get("arm", "none"))
    if arm == "gate":
        from types import SimpleNamespace
        from rl.jfcrl_v2_runtime import install_gate
        agent = install_gate(
            agent, ref["obs_dim"], cap.latent_dim, ref["act_dim"],
            SimpleNamespace(**ckargs),
        )
    agent.load_state_dict(blob["agent"], load_optimizers=False)
    agent.actor.eval()
    normalizer = RunningNorm(ref["obs_dim"])
    normalizer.load_state_dict(blob["obs_norm"])

    trained_faults = {FaultSpec.parse(t).name for t in blob["joint_pool"]}
    K = cap.context_len
    ctx_dim = ref["joint_ctx_dim"]
    zero = np.zeros(ref["act_dim"], dtype=np.float32)
    rows, table = [], {}

    for b in bundles:
        ids = list(b.eval_ids) if args.held_out else list(range(b.meta["n_init_states"]))
        ids = ids[:args.n_episodes]
        table[b.spec.key] = {}
        for spec in b.renv.faults.specs:
            table[b.spec.key][spec.name] = {}
            for policy in args.policies:
                succ, worst = 0, 0.0
                gate_vals = []
                for epi, init_id in enumerate(ids):
                    obs = b.renv.reset(init_id=int(init_id), force_fault=spec)
                    b.renv.faults.assert_exactly_one_lock(b.renv.env)
                    hist = deque(maxlen=K)
                    z_norms = []
                    done = False
                    while not done:
                        if policy == "zero":
                            a = zero; zn = 0.0
                        else:
                            ctx, mask = build_left_padded_history(hist, K, ctx_dim)
                            z = cap.encode_numpy(ctx, mask, obs)
                            zn = float(np.linalg.norm(z))
                            if hasattr(agent.actor, "gate"):
                                zt = torch.as_tensor(
                                    z, dtype=torch.float32, device=agent.device
                                ).unsqueeze(0)
                                gate_vals.append(float(agent.actor.gate(zt).item()))
                            a = agent.act(normalizer(obs).astype(np.float32), z,
                                          deterministic=True)
                        obs, _, term, trunc, info = b.renv.step(a, gate=1.0)
                        hist.append(b.renv.last_joint_context.copy())
                        z_norms.append(zn)
                        worst = max(worst, info["lock_drift"])
                        done = term or trunc
                    succ += int(info["success"])
                    rows.append({
                        "task": b.spec.key,
                        "task_description": b.meta["task_description"],
                        "fault": spec.name, "policy": policy,
                        "episode": epi, "init_id": int(init_id),
                        "success": int(info["success"]), "steps": info["t"],
                        "seen_joint_in_training": int(spec.name in trained_faults),
                        "mean_z_norm": float(np.mean(z_norms)) if z_norms else 0.0,
                        "max_lock_drift_rad": worst,
                    })
                table[b.spec.key][spec.name][policy] = {
                    "success": succ, "n": len(ids),
                    "rate": succ / max(1, len(ids)),
                    "seen_in_training": spec.name in trained_faults,
                    "worst_lock_drift_rad": worst,
                    "mean_gate": (float(np.mean(gate_vals)) if gate_vals else None),
                }

    out = Path(args.outdir) / ("heldout" if args.held_out else "screen")
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / "per_episode.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "ckpt": args.ckpt, "method": blob["method"],
        "tasks": blob["tasks"], "trained_faults": sorted(trained_faults),
        "shared_vla_checkpoint": shared_ckpt,
        "arm": arm,
        "heldout_joint_indices": blob.get("heldout_joints", []),
        "table": table,
    }, indent=2))

    print("\n" + "=" * 94)
    print(f"{'task':>20} | {'condition':>10} | " +
          " | ".join(f"{p:>10}" for p in args.policies) + " | seen?")
    print("-" * 94)
    for task_key, conds in table.items():
        for name, per in conds.items():
            cells = " | ".join(
                f"{per[p]['success']}/{per[p]['n']}={per[p]['rate']:.0%}".rjust(10)
                for p in args.policies
            )
            tag = "trained" if name in trained_faults else "UNSEEN"
            print(f"{task_key:>20} | {name:>10} | {cells} | {tag}")
    print("=" * 94)
    if arm == "gate":
        print("\ngate by TASK x CONDITION (checkpoint policy; 1 = full correction authority)")
        for task_key, conds in table.items():
            cells = []
            for name, per in conds.items():
                mg = per.get("ckpt", {}).get("mean_gate")
                if mg is not None:
                    cells.append(f"{name}={mg:.3f}")
            if cells:
                print(f"  {task_key}: " + "  ".join(cells))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
