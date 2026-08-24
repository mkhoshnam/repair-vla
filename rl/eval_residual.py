"""
eval_residual.py -- deterministic evaluation and, more importantly, the
smoke test that decides whether any later number means anything.

Three policies:

  --policy zero    residual identically 0. MUST reproduce the frozen-VLA
                   baseline. This is handoff step 13.4 and it is the single
                   most valuable half hour in the project: if the wrapper
                   quietly changed the control loop -- an off-by-one in the
                   chunk queue, a gripper sign flip, a missed dummy-action
                   prefix -- it shows up here as a success rate that is not
                   ~20% faulted / ~95% healthy, and it shows up BEFORE you
                   spend a night training against a broken environment.

  --policy random  uniform residual at the configured scale. This is the
                   perturbation control (handoff section 15). If random
                   noise also lifts success, the story is "jitter helps a
                   stuck policy", not "RL learned a compensation". You want
                   this number in the paper whichever way it comes out.

  --policy ckpt    the trained actor, deterministic (mean action, no sample).

Run order on Denver:
  1. --policy zero --no_fault   -> expect ~19/20 on init states 0..19
  2. --policy zero              -> expect ~4/20  on init states 0..19
  3. --policy random            -> the control
  4. --policy ckpt --held_out   -> the result
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.build import TASK_MAX_STEPS, VLACfg, set_headless_env  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["zero", "random", "ckpt"], default="zero")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--joint_idx", type=int, default=0)
    p.add_argument("--no_fault", action="store_true",
                   help="healthy condition (upper reference)")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--held_out", action="store_true",
                   help="evaluate on the held-out tail states instead of 0..N-1")
    p.add_argument("--n_eval_states", type=int, default=10)
    p.add_argument("--residual_scale", type=float, default=0.2)
    p.add_argument("--history_len", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--outdir", type=str, default=None,
                   help="default: rollouts_residual/eval_j{joint_idx}, so a "
                        "j2 run can never overwrite the frozen j0 results")
    p.add_argument("--save_video", action="store_true")
    return p.parse_args()


# Task 0 all-joint screening, 12 Aug 2026, zero residual, init states 0..19.
# These are SCREENING numbers on states 0..19. They are NOT the held-out
# baseline on 40..49, which is a different set of initial states and must be
# measured separately for every joint before that joint's training run.
SCREENING_0_TO_19 = {0: 0.20, 1: 0.00, 2: 0.35, 3: 0.00,
                     4: 1.00, 5: 1.00, 6: 0.60}
HEALTHY_0_TO_19 = 1.00  # 20/20, validated


def main():
    args = parse_args()
    set_headless_env()

    from experiments.robot.robot_utils import set_seed_everywhere

    from rl.build import build_all
    from rl.residual_env import ResidualCfg

    set_seed_everywhere(args.seed)

    cfg = VLACfg(task_id=args.task_id, seed=args.seed)
    res_cfg = ResidualCfg(
        residual_scale=args.residual_scale,
        history_len=args.history_len,
        max_steps=TASK_MAX_STEPS[cfg.task_suite_name],
        num_steps_wait=cfg.num_steps_wait,
        seed=args.seed,
    )

    renv, train_ids, eval_ids, meta = build_all(
        cfg,
        res_cfg,
        joint_idx=args.joint_idx,
        fault_enabled=not args.no_fault,
        n_eval_states=args.n_eval_states,
        collect_images=args.save_video,
    )

    if args.held_out:
        ids = eval_ids
    else:
        # The indices your validated screening used. Comparing against 4/20
        # or 19/20 only means something on the same initial states.
        ids = list(range(min(args.n_episodes, meta["n_init_states"])))
    ids = ids[: args.n_episodes]

    agent, normalizer = None, None
    if args.policy == "ckpt":
        import torch

        from rl.sac import RunningNorm, SACAgent

        assert args.ckpt, "--policy ckpt requires --ckpt"
        blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        agent = SACAgent(meta["obs_dim"], meta["act_dim"], device=args.device)
        agent.load_state_dict(blob["agent"])
        agent.actor.eval()
        normalizer = RunningNorm(meta["obs_dim"])
        normalizer.load_state_dict(blob["obs_norm"])
        saved_scale = blob.get("args", {}).get("residual_scale")
        if saved_scale is not None and abs(saved_scale - args.residual_scale) > 1e-9:
            raise SystemExit(
                f"residual_scale mismatch: checkpoint trained at {saved_scale}, "
                f"evaluating at {args.residual_scale}. The actor's outputs mean "
                f"something different at a different scale."
            )

    rng = np.random.default_rng(args.seed)
    outdir = Path(
        args.outdir
        if args.outdir
        else (
            "rollouts_residual/eval_healthy"
            if args.no_fault
            else f"rollouts_residual/eval_j{args.joint_idx}"
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.policy}{'_healthy' if args.no_fault else f'_j{args.joint_idx}'}"
    tag += "_heldout" if args.held_out else "_screen"

    rows, n_succ = [], 0
    for k, init_id in enumerate(ids):
        obs = renv.reset(init_id=int(init_id))
        done, t = False, 0
        res_norms, drifts = [], []

        while not done:
            if args.policy == "zero":
                a = np.zeros(meta["act_dim"], dtype=np.float32)
            elif args.policy == "random":
                a = rng.uniform(-1, 1, size=meta["act_dim"]).astype(np.float32)
            else:
                a = agent.act(normalizer(obs).astype(np.float32), deterministic=True)

            obs, r, terminated, truncated, info = renv.step(a)
            res_norms.append(info["residual_norm"])
            drifts.append(info["lock_drift"])
            t += 1
            done = terminated or truncated

        succ = bool(info["success"])
        n_succ += int(succ)
        max_drift = float(np.max(drifts)) if drifts else 0.0

        # A faulted episode whose joint moved is not a faulted episode.
        lock_ok = args.no_fault or max_drift <= 1e-2
        if not lock_ok:
            print(f"  !! LOCK FAILED init_id={init_id} max_drift={max_drift:.3e} rad")

        rows.append(
            {
                "episode": k,
                "init_id": int(init_id),
                "success": int(succ),
                "steps": t,
                "residual_norm_mean": float(np.mean(res_norms)) if res_norms else 0.0,
                "max_lock_drift_rad": max_drift,
                "lock_ok": int(lock_ok),
            }
        )
        print(
            f"[{k + 1}/{len(ids)}] init={init_id} success={succ} steps={t} "
            f"drift={max_drift:.2e} running={n_succ}/{k + 1}"
        )

        if args.save_video and renv.replay_images:
            from experiments.robot.libero.libero_utils import save_rollout_video

            save_rollout_video(
                renv.replay_images, k, success=succ,
                task_description=meta["task_description"],
            )

        # incremental write: an interrupted run still leaves usable data
        with open(outdir / f"results_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = {
        "policy": args.policy,
        "fault": (None if args.no_fault else f"j{args.joint_idx}_lock"),
        "init_state_set": ("held_out" if args.held_out else "screening_0_to_N"),
        "init_ids": [int(i) for i in ids],
        "n_episodes": len(ids),
        "n_success": n_succ,
        "success_rate": n_succ / max(1, len(ids)),
        "worst_lock_drift_rad": max((r["max_lock_drift_rad"] for r in rows), default=0.0),
        "all_locks_ok": all(r["lock_ok"] for r in rows),
        "residual_scale": args.residual_scale,
        "ckpt": args.ckpt,
        "seed": args.seed,
        **renv.faults.stats(),
        **meta,
    }
    with open(outdir / f"summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 62)
    print(f"{tag}: {n_succ}/{len(ids)} = {100 * n_succ / max(1, len(ids)):.1f}%")
    print(f"worst lock drift: {summary['worst_lock_drift_rad']:.3e} rad")
    if args.policy == "zero":
        if args.held_out:
            print(
                "\nThis is the HELD-OUT frozen-VLA baseline on states 40-49.\n"
                "It is the number the trained residual must beat, and it must\n"
                "be recorded BEFORE training. Do not expect it to match the\n"
                "screening rate: these are different initial states. For j0\n"
                "the screening rate was 20% but the held-out baseline was 50%."
            )
        elif args.no_fault:
            print(f"\nSanity target: healthy ~{HEALTHY_0_TO_19:.0%} on states 0-19.")
            print("A materially different number means the WRAPPER changed,")
            print("not the environment. Fix it before training anything.")
        else:
            exp = SCREENING_0_TO_19.get(args.joint_idx)
            if exp is None:
                print(f"\nNo screening reference on record for j{args.joint_idx}.")
            else:
                print(f"\nSanity target for j{args.joint_idx} on states 0-19: "
                      f"~{exp:.0%} (12 Aug all-joint sweep).")
            print("A materially different number means the WRAPPER changed,")
            print("not the environment. Fix it before training anything.")
    print("=" * 62)


if __name__ == "__main__":
    main()
