"""
probe_interfaces.py -- run this FIRST on Denver, before anything else.

`test_offline.py` verifies the arithmetic. It cannot verify facts about your
actual LIBERO build. These five can only be answered on the machine, and
every one of them can invalidate a training run silently:

1. Does the LIBERO obs dict contain `robot0_joint_pos` / `robot0_joint_vel`?
   The observation builder indexes them directly. If robosuite 1.4.1 names
   them differently in your build, reset() raises immediately -- annoying but
   safe. Worse would be discovering it after a night of training.

2. Do arm action dims land in [-1, 1]?
   The residual is scaled in those units and the sum is clipped there. If
   the unnormalized VLA output is on a different scale, `residual_scale=0.2`
   is either a rounding error or a catastrophe, and `clipped_frac` lies.

3. Is `done` True ONLY on success, never at the horizon?
   `success = bool(done)` is the upstream convention this wrapper copies. If
   your LIBERO also sets done at the robosuite horizon, every timeout gets
   recorded as a success and the reward signal is garbage.

4. What is the real throughput, in env-steps/s, with the VLA in the loop?
   Decides whether `--total_steps` fits in the time you have.

5. Is NUM_ACTIONS_CHUNK actually 8?

It also runs a two-episode lock check: healthy vs faulted on the same init
state, reporting max drift. That is the smallest possible confirmation that
`joint_lock.py` reproduces the mechanism your validated evaluator used.

    python rl/probe_interfaces.py --task_id 0 --joint_idx 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.build import TASK_MAX_STEPS, VLACfg, set_headless_env  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--joint_idx", type=int, default=0)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--out", type=str, default="probe_report.json")
    args = p.parse_args()

    set_headless_env()
    report = {}

    from rl.build import build_all
    from rl.residual_env import ResidualCfg

    cfg = VLACfg(task_id=args.task_id)
    res_cfg = ResidualCfg(max_steps=TASK_MAX_STEPS[cfg.task_suite_name],
                          num_steps_wait=cfg.num_steps_wait)

    # ---- 5. chunk length -------------------------------------------------
    try:
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK
        report["chunk_len_constant"] = int(NUM_ACTIONS_CHUNK)
        report["chunk_len_matches_cfg"] = (
            int(NUM_ACTIONS_CHUNK) == cfg.num_open_loop_steps
        )
    except Exception as e:
        report["chunk_len_constant"] = f"UNAVAILABLE: {e!r}"

    print("building env + VLA (this loads a 7B model, give it a minute)...")
    renv, train_ids, eval_ids, meta = build_all(
        cfg, res_cfg, joint_idx=args.joint_idx, fault_enabled=True
    )
    report["meta"] = meta
    report["n_train_states"] = len(train_ids)
    report["n_eval_states"] = len(eval_ids)

    # ---- 1. observation keys --------------------------------------------
    obs = renv.reset(init_id=0)
    raw = renv._env_obs
    needed = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
              "robot0_joint_pos", "robot0_joint_vel"]
    report["obs_keys_present"] = {k: (k in raw) for k in needed}
    report["obs_keys_missing"] = [k for k in needed if k not in raw]
    report["all_obs_keys"] = sorted(str(k) for k in raw.keys())
    report["residual_obs_dim"] = int(obs.shape[0])
    report["residual_obs_finite"] = bool(np.isfinite(obs).all())

    # ---- 2, 3, 4. rollout with zero residual ----------------------------
    z = np.zeros(renv.act_dim, dtype=np.float32)
    base_min = np.full(7, np.inf)
    base_max = np.full(7, -np.inf)
    drifts, dones_before_horizon = [], 0

    t0 = time.time()
    n = 0
    for _ in range(args.steps):
        b = renv._a_base
        base_min = np.minimum(base_min, b)
        base_max = np.maximum(base_max, b)
        _, r, term, trunc, info = renv.step(z)
        drifts.append(info["lock_drift"])
        n += 1
        if term:
            dones_before_horizon += 1
            renv.reset(init_id=0)
    dt = time.time() - t0

    report["base_action_min"] = base_min.round(4).tolist()
    report["base_action_max"] = base_max.round(4).tolist()
    report["arm_dims_within_pm1"] = bool(
        base_min[:6].min() >= -1.001 and base_max[:6].max() <= 1.001
    )
    report["gripper_values_seen"] = sorted(
        {round(float(base_min[6]), 3), round(float(base_max[6]), 3)}
    )
    report["env_steps_per_s"] = round(n / dt, 3)
    report["vla_queries"] = renv.vla.n_queries
    report["steps_per_vla_query"] = round(n / max(1, renv.vla.n_queries), 2)
    report["early_dones"] = dones_before_horizon

    for total in (50_000, 100_000, 200_000):
        report[f"projected_hours_{total // 1000}k"] = round(
            total / max(1e-9, n / dt) / 3600, 2
        )

    # ---- lock verification ----------------------------------------------
    report["faulted_max_drift_rad"] = float(np.max(drifts))
    report["lock_holding"] = bool(np.max(drifts) < 1e-2)
    report["fault_stats"] = renv.faults.stats()

    print("\n" + "=" * 68)
    print(json.dumps(report, indent=2, default=str))
    print("=" * 68)

    problems = []
    if report["obs_keys_missing"]:
        problems.append(
            f"missing obs keys {report['obs_keys_missing']} -- fix the key "
            f"names in ResidualLiberoEnv._build_obs against 'all_obs_keys'"
        )
    if not report.get("arm_dims_within_pm1", True):
        problems.append(
            "arm action dims exceed [-1,1] -- residual_scale is in the wrong "
            "units and the clip is wrong"
        )
    if not report.get("lock_holding", True):
        problems.append(
            f"joint drifted {report['faulted_max_drift_rad']:.3e} rad -- the "
            f"constraint is not holding; compare against your validated "
            f"1.25e-4 before training"
        )
    if report.get("chunk_len_matches_cfg") is False:
        problems.append("NUM_ACTIONS_CHUNK != cfg.num_open_loop_steps")
    if report["env_steps_per_s"] < 5:
        problems.append(
            f"{report['env_steps_per_s']:.1f} env-steps/s is very slow; "
            f"100k steps would take {report['projected_hours_100k']:.1f} h"
        )

    print("\nBLOCKERS:" if problems else "\nNo blockers found.")
    for x in problems:
        print(f"  - {x}")
    print(
        "\nNOTE: `early_dones` counts steps where done=True. With a zero "
        "residual under the fault this should be rare and should coincide "
        "with genuine task success. A done at exactly the horizon means "
        "`success = bool(done)` is unsafe in your build -- fix it before "
        "training, or every timeout scores as a success."
    )

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
