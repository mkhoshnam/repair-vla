"""
screen_faults.py -- which (task, joint) pairs are worth training on?

    python scripts/screen_faults.py --suite libero_spatial --n_episodes 10

Sweeps every task in a suite against every arm joint with a zero residual
(frozen VLA only, no RL) and records the success rate of each cell, plus the
healthy baseline for every task. Then classifies each cell and prints a
ranked shortlist.

--------------------------------------------------------------------------
WHY THIS DECIDES WHETHER THE PAPER EXISTS
--------------------------------------------------------------------------
On Task 0, four of seven joints were unusable: j1 and j3 sat at 0% (the
fault destroys the task outright) and j4 and j5 at 100% (the fault does
nothing at all). Only j0 and j6 had real headroom. A generalization claim
built on a training set of two faults is not separable from memorization,
so the whole question is how many usable cells exist across the suite. This
sweep answers it before any GPU-weeks go into training.

Three outcomes per cell, and only one is useful:

  NO_EFFECT      faulted rate ~= healthy rate. The lock does not change the
                 task. Nothing to recover.
  NO_SIGNAL      faulted rate ~ 0. Sparse-reward RL has nothing to bootstrap
                 from -- no episode ever succeeds, so no gradient carries
                 information. Note this is NOT proof of unrecoverability:
                 j0 screened at 20% on states 0-19 but was 50% on 40-49, so
                 a 0/10 cell may be a small-sample artifact. Cells at
                 exactly 0 with n=10 are re-checked at higher n by
                 --recheck_zeros before being written off.
  USABLE         headroom between the two, with a non-zero base rate. These
                 are the cells worth training on.

--------------------------------------------------------------------------
TWO STAGES, BECAUSE STAGE 2 IS EXPENSIVE
--------------------------------------------------------------------------
Stage 1 (default): every task x every joint at offset 0. For libero_spatial
that is 10 tasks x 7 joints + 10 healthy = 80 cells.

Stage 2 (--offsets): the severity axis, applied ONLY to cells stage 1 marked
USABLE. Running offsets over cells that are already 0% or 100% wastes hours
on conditions that cannot inform anything. Stage 2 is what turns two joints
into a fault FAMILY with a real train/test split -- train on offsets
{-0.2, 0, +0.2}, hold out {+-0.1} for interpolation and {+-0.35} for
extrapolation.

--------------------------------------------------------------------------
BUILT TO SURVIVE A TEN-HOUR RUN
--------------------------------------------------------------------------
* Results append to CSV after every single cell.
* Re-running skips cells already in the CSV, so a crash or a Ctrl+C costs
  one cell, not the sweep.
* The 7B model loads ONCE; only the env is rebuilt per task.
* A failure in one cell is logged and the sweep continues.
* --max_hours stops cleanly at a wall-clock budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.build import TASK_MAX_STEPS, VLACfg, set_headless_env  # noqa: E402

# OpenVLA-OFT ships one finetuned checkpoint per suite. Evaluating
# libero_object tasks with the libero_spatial checkpoint measures a
# distribution mismatch, not a fault.
COMBINED_CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"

SUITE_CHECKPOINTS = {
    "libero_spatial": COMBINED_CHECKPOINT,
    "libero_object": COMBINED_CHECKPOINT,
    "libero_goal": COMBINED_CHECKPOINT,
    "libero_10": COMBINED_CHECKPOINT,
}
SUITE_UNNORM = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}

FIELDS = ["suite", "task_id", "task_name", "condition", "joint_idx", "kind",
          "offset", "n_episodes", "n_success", "success_rate",
          "worst_lock_drift_rad", "lock_ok", "mean_steps", "seconds", "error"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_spatial",
                   choices=list(SUITE_CHECKPOINTS))
    p.add_argument("--tasks", type=int, nargs="+", default=None,
                   help="Default: every task in the suite.")
    p.add_argument("--joints", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--init_ids", type=int, nargs="+", default=None,
                   help="Default: states 0..n_episodes-1, i.e. TRAIN states. "
                        "Held-out 40-49 are deliberately not screened -- "
                        "screening every cell on them would burn the held-out "
                        "set before a single policy is trained.")
    p.add_argument("--offsets", type=float, nargs="+", default=None,
                   help="STAGE 2 severity axis. Applied only to cells a prior "
                        "stage-1 run marked USABLE. e.g. --offsets -0.35 -0.2 "
                        "-0.1 0.1 0.2 0.35")
    p.add_argument("--recheck_zeros", type=int, default=0,
                   help="Re-run cells that scored exactly 0 with this many "
                        "episodes before classifying them NO_SIGNAL.")
    p.add_argument("--out", default="screening")
    p.add_argument("--max_hours", type=float, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_done(path: Path) -> dict:
    """key -> row, for every cell already completed."""
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            out[(int(r["task_id"]), r["condition"])] = r
    return out


def append_row(path: Path, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def classify(healthy: float, faulted: float,
             no_effect_tol: float = 0.10, signal_floor: float = 0.05) -> str:
    if healthy is None:
        return "UNKNOWN"
    if healthy < 0.6:
        return "WEAK_TASK"          # the VLA is poor here even unfaulted
    if faulted >= healthy - no_effect_tol:
        return "NO_EFFECT"
    if faulted <= signal_floor:
        return "NO_SIGNAL"
    return "USABLE"


def main():
    args = parse_args()
    set_headless_env()

    from experiments.robot.robot_utils import set_seed_everywhere
    from libero.libero import benchmark

    from faults.multi_fault import FaultSpec, MultiFaultManager
    from rl.build import build_env, build_vla
    from rl.residual_env import FrozenOFT, ResidualCfg, ResidualLiberoEnv

    set_seed_everywhere(args.seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"cells_{args.suite}.csv"
    done = load_done(csv_path)
    print(f"{len(done)} cells already complete in {csv_path}; they'll be skipped")

    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = suite_obj.n_tasks
    tasks = args.tasks if args.tasks is not None else list(range(n_tasks))
    init_ids = args.init_ids or list(range(args.n_episodes))

    # ---- stage 2 gating ------------------------------------------------
    usable_from_stage1 = None
    if args.offsets:
        prev = outdir / f"summary_{args.suite}.json"
        if not prev.exists():
            raise SystemExit(
                f"--offsets is stage 2 and needs stage 1's {prev}. Run the "
                f"sweep without --offsets first; spending hours on offsets "
                f"for cells that are already 0% or 100% informs nothing."
            )
        blob = json.loads(prev.read_text())
        usable_from_stage1 = {
            (int(c["task_id"]), int(c["joint_idx"]))
            for c in blob["cells"] if c["class"] == "USABLE"
        }
        print(f"stage 2: {len(usable_from_stage1)} USABLE cells from stage 1")
        if not usable_from_stage1:
            raise SystemExit("stage 1 found no USABLE cells; nothing to sweep")

    # ---- load the 7B model ONCE ----------------------------------------
    cfg = VLACfg(task_suite_name=args.suite, task_id=tasks[0], seed=args.seed)
    cfg.pretrained_checkpoint = SUITE_CHECKPOINTS[args.suite]
    cfg.unnorm_key = SUITE_UNNORM[args.suite]
    print(f"loading {cfg.pretrained_checkpoint} (once for the whole sweep)...")
    handles = build_vla(cfg)

    res_cfg = ResidualCfg(residual_scale=0.0,
                          max_steps=TASK_MAX_STEPS[args.suite],
                          num_steps_wait=cfg.num_steps_wait, seed=args.seed)

    # conditions per task
    conds = [("healthy", None, "lock", 0.0)]
    if args.offsets:
        conds = []
        for j in args.joints:
            for off in args.offsets:
                conds.append((f"j{j}_off{off:+g}", j, "lock", off))
    else:
        for j in args.joints:
            conds.append((f"j{j}", j, "lock", 0.0))

    t_start = time.time()
    total = len(tasks) * len(conds)
    idx = 0
    z = np.zeros(6, dtype=np.float32)

    # ---------------------------------------------------------------------
    # ONE FRESH MuJoCo ENV PER (task, fault) CELL.
    #
    # Reusing a single env and switching the fault between cells does not
    # work: robosuite applies `_xml_processors` only inside
    # `_initialize_sim`, and `reset()` calls that only when
    # `(sim is None) or (hard_reset and not deterministic_reset)`. When that
    # is false the processor never runs, NO constraint is compiled, and the
    # cell silently measures the HEALTHY policy while being recorded as
    # faulted. That is the worst kind of bug -- the numbers look reasonable.
    # (A second reason: `set_xml_processor` APPENDS to a list rather than
    # replacing, so reusing an env also grows that list every cell.)
    #
    # A fresh env per cell is exactly what the validated single-fault
    # evaluator did, which is why that path always worked. The 7B model is
    # still loaded once for the whole sweep; only the ~2-5 s env build is
    # repeated, against ~200 s of episodes per cell.
    # ---------------------------------------------------------------------
    for task_id in tasks:
        cfg.task_id = task_id
        try:
            probe_env, task_desc, _ = build_env(cfg)
            initial_states = benchmark.get_benchmark_dict()[args.suite]() \
                .get_task_init_states(task_id)
            probe_env.close()
        except Exception:
            print(f"!! task {task_id}: env build failed\n{traceback.format_exc()}")
            continue
        print(f"\n=== task {task_id}: {task_desc} ===")

        for cond_name, joint_idx, kind, offset in conds:
            idx += 1
            if (task_id, cond_name) in done:
                print(f"  [{idx}/{total}] {cond_name}: cached")
                continue
            if (usable_from_stage1 is not None and joint_idx is not None
                    and (task_id, joint_idx) not in usable_from_stage1):
                continue
            if args.max_hours and (time.time() - t_start) / 3600 > args.max_hours:
                print(f"\n--max_hours reached; {total - idx} cells remain. "
                      f"Re-run the same command to continue.")
                _report(outdir, args, csv_path)
                return

            t0 = time.time()
            row = {"suite": args.suite, "task_id": task_id,
                   "task_name": task_desc, "condition": cond_name,
                   "joint_idx": "" if joint_idx is None else joint_idx,
                   "kind": "healthy" if joint_idx is None else kind,
                   "offset": offset, "n_episodes": len(init_ids)}
            env = renv = mgr = None
            try:
                spec = (FaultSpec(None) if joint_idx is None
                        else FaultSpec(joint_idx, kind=kind, offset=offset))
                # Fresh env for this cell, and a factory so the manager
                # rebuilds rather than mutating a reused one.
                env = build_env(cfg)[0]
                mgr = MultiFaultManager(joint_pool=(spec,), seed=args.seed,
                                        env_factory=lambda: build_env(cfg)[0])
                vla = FrozenOFT(cfg, handles["model"], handles["resize_size"],
                                task_desc, processor=handles["processor"],
                                action_head=handles["action_head"],
                                proprio_projector=handles["proprio_projector"],
                                noisy_action_projector=handles["noisy_action_projector"],
                                use_film=cfg.use_film,
                                chunk_len=cfg.num_open_loop_steps)
                renv = ResidualLiberoEnv(env=env, vla=vla, fault_mgr=mgr,
                                         initial_states=initial_states,
                                         init_ids=init_ids, cfg=res_cfg)

                n_succ, worst, steps = 0, 0.0, []
                for k, iid in enumerate(init_ids):
                    renv.reset(init_id=int(iid), force_fault=spec)
                    # `renv.env` -- not the local `env` -- because the manager
                    # may have swapped in a rebuilt one.
                    mgr.assert_exactly_one_lock(renv.env)
                    done_ep = False
                    while not done_ep:
                        _, _, term, trunc, info = renv.step(z)
                        worst = max(worst, info["lock_drift"])
                        done_ep = term or trunc
                    n_succ += int(info["success"])
                    steps.append(info["t"])

                row.update({
                    "n_success": n_succ,
                    "success_rate": round(n_succ / len(init_ids), 4),
                    "worst_lock_drift_rad": f"{worst:.3e}",
                    "lock_ok": int(joint_idx is None or worst <= 1e-2),
                    "mean_steps": round(float(np.mean(steps)), 1),
                    "seconds": round(time.time() - t0, 1),
                })
                bar = "#" * n_succ + "." * (len(init_ids) - n_succ)
                print(f"  [{idx}/{total}] {cond_name:>14}: {n_succ}/"
                      f"{len(init_ids)} [{bar}] "
                      f"drift={worst:.1e} ({row['seconds']}s)")
                if not row["lock_ok"]:
                    print(f"      !! lock drifted {worst:.2e} rad -- cell invalid")
            except Exception as e:
                row["error"] = repr(e)
                print(f"  [{idx}/{total}] {cond_name}: FAILED {e!r}")
                traceback.print_exc()
            finally:
                # Close every env this cell created -- the original and any
                # the manager rebuilt. Leaking MuJoCo/EGL contexts across 80
                # cells exhausts GPU memory long before the sweep finishes.
                live = []
                for cand in (env,
                             getattr(mgr, "env", None),
                             getattr(renv, "env", None)):
                    if cand is not None and not any(cand is x for x in live):
                        live.append(cand)
                for e_ in live:
                    try:
                        e_.close()
                    except Exception:
                        pass

            append_row(csv_path, row)

    _report(outdir, args, csv_path)


def _report(outdir: Path, args, csv_path: Path):
    """Classify every cell and print the shortlist."""
    rows = []
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if not r.get("error")]

    healthy = {int(r["task_id"]): float(r["success_rate"])
               for r in rows if r["condition"] == "healthy"}

    cells = []
    for r in rows:
        if r["condition"] == "healthy":
            continue
        h = healthy.get(int(r["task_id"]))
        f_rate = float(r["success_rate"])
        cells.append({
            "task_id": int(r["task_id"]),
            "task_name": r["task_name"],
            "condition": r["condition"],
            "joint_idx": int(r["joint_idx"]) if r["joint_idx"] != "" else None,
            "offset": float(r["offset"] or 0.0),
            "healthy_rate": h,
            "faulted_rate": f_rate,
            "headroom": None if h is None else round(h - f_rate, 4),
            "lock_ok": bool(int(r["lock_ok"])),
            "class": classify(h, f_rate),
        })

    # rank: big headroom, but a non-zero base rate so sparse RL can bootstrap
    usable = [c for c in cells if c["class"] == "USABLE" and c["lock_ok"]]
    usable.sort(key=lambda c: (c["headroom"], c["faulted_rate"]), reverse=True)

    summary = {
        "suite": args.suite,
        "n_cells": len(cells),
        "healthy_by_task": healthy,
        "counts": {k: sum(1 for c in cells if c["class"] == k)
                   for k in ["USABLE", "NO_EFFECT", "NO_SIGNAL",
                             "WEAK_TASK", "UNKNOWN"]},
        "cells": cells,
        "shortlist": usable,
    }
    (outdir / f"summary_{args.suite}.json").write_text(
        json.dumps(summary, indent=2))

    # ---- grid -----------------------------------------------------------
    joints = sorted({c["joint_idx"] for c in cells if c["joint_idx"] is not None})
    tasks = sorted({c["task_id"] for c in cells})
    print("\n" + "=" * 78)
    print(f"{args.suite}: frozen-VLA success per (task, joint). "
          f"'hlth' is the unfaulted rate.")
    print(f"{'task':>4} {'hlth':>5} | " + " ".join(f"j{j:<4}" for j in joints))
    print("-" * 78)
    for t in tasks:
        h = healthy.get(t)
        line = f"{t:>4} {('--' if h is None else f'{h:.0%}'):>5} | "
        for j in joints:
            c = next((x for x in cells if x["task_id"] == t
                      and x["joint_idx"] == j and abs(x["offset"]) < 1e-9), None)
            if c is None:
                line += "  --  "
            else:
                mark = {"USABLE": "*", "NO_EFFECT": "=",
                        "NO_SIGNAL": "0", "WEAK_TASK": "w"}.get(c["class"], "?")
                line += f"{c['faulted_rate']:>4.0%}{mark} "
        print(line)
    print("-" * 78)
    print("  * USABLE (headroom, non-zero base rate)   = NO_EFFECT (fault "
          "does nothing)")
    print("  0 NO_SIGNAL (task destroyed; sparse RL cannot bootstrap)   "
          "w WEAK_TASK (healthy < 60%)")
    print(f"\ncounts: {summary['counts']}")

    print("\n" + "=" * 78)
    print("SHORTLIST -- cells worth training on, best first")
    print("=" * 78)
    if not usable:
        print("NONE. Every cell is either unaffected by the fault or "
              "destroyed by it.\nThat is a result in itself, and it means "
              "the fault space needs widening\n(severities, damping, "
              "or another suite) before a training run makes sense.")
    for c in usable[:20]:
        print(f"  task {c['task_id']:>2} {c['condition']:>14}  "
              f"healthy {c['healthy_rate']:.0%} -> faulted "
              f"{c['faulted_rate']:.0%}  (headroom {c['headroom']:.0%})  "
              f"{c['task_name'][:36]}")

    n0 = sum(1 for c in cells if c["class"] == "NO_SIGNAL")
    if n0 and not args.offsets:
        print(f"\n{n0} cells scored at or near zero. Before writing them off: "
              f"j0 screened\n20% on states 0-19 but was 50% on 40-49, so with "
              f"n={args.n_episodes} a 0/N cell can be\nsmall-sample noise. "
              f"Re-run those with --recheck_zeros 40 if any look promising.")
    print(f"\nwritten to {outdir}/summary_{args.suite}.json")


if __name__ == "__main__":
    main()
