#!/usr/bin/env python3
"""
screen_to_curriculum.py  (v2 -- held-out-blind selection)

Build a per-task fault curriculum from a frozen-VLA screening log.

SELECTION IS BLIND TO THE HELD-OUT JOINT.
    Tasks are kept using only:
      * healthy frozen-VLA success on that task
      * whether at least one SEEN training joint is USABLE on that task
    The held-out joint is never read during selection or ranking. Cells where
    the held-out joint screens 0% are kept, because a 0% baseline is a valid
    and informative evaluation cell -- object:6 went 0% -> 50% in the 5-task
    run.

    Two files are written:
      <out>                      the curriculum. Contains no held-out data.
      <out>.heldout_reference    held-out screen values, for reporting only.
    They are separate so the curriculum is provably independent of held-out
    performance, which is what makes the selection defensible in review.

Cell classes as printed by the screener:
    '*'  USABLE     headroom, non-zero base rate     -> train here
    '='  NO_EFFECT  fault does nothing               -> skip; healthy covers it
    '0'  NO_SIGNAL  task destroyed, no bootstrap     -> skip for training
    'w'  WEAK_TASK  healthy < 60%

Usage
-----
  # main 20+ task pool, j2 held out, selection never looks at j2
  python screen_to_curriculum.py \
      --log 40_task_screening_combinedvla.log \
      --train_joints 0 4 5 6 --heldout 2 \
      --out configs/curriculum_main.json

  # 6-task diagnostic subset
  python screen_to_curriculum.py \
      --log 40_task_screening_combinedvla.log \
      --train_joints 0 4 5 6 --heldout 2 \
      --tasks libero_spatial:0 libero_goal:6 libero_object:6 \
              libero_goal:0 libero_goal:3 libero_10:3 \
      --out configs/curriculum_diag6.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

USABLE, NO_EFFECT, NO_SIGNAL = "*", "=", "0"

TABLE_RE = re.compile(
    r"(libero_\w+): frozen-VLA success per.*?\n(.*?)\n-{10,}\n\s*\*", re.S
)
ROW_RE = re.compile(r"\s*(\d+)\s+(\d+)%\s*\|(.*)")
CELL_RE = re.compile(r"(\d+)%([=0*w])")


def parse_screening(path):
    """-> {task_key: {"healthy": int, "cells": [(pct, flag)] * 7}}"""
    txt = open(path).read()
    out = {}
    for m in TABLE_RE.finditer(txt):
        suite = m.group(1)
        for line in m.group(2).split("\n"):
            rm = ROW_RE.match(line)
            if not rm:
                continue
            tid, healthy = int(rm.group(1)), int(rm.group(2))
            cells = [(int(v), f) for v, f in CELL_RE.findall(rm.group(3))]
            if len(cells) != 7:
                print(f"[warn] {suite}:{tid}: {len(cells)} cells parsed, skipped",
                      file=sys.stderr)
                continue
            out[f"{suite}:{tid}"] = {"healthy": healthy, "cells": cells}
    if not out:
        sys.exit(f"[fatal] no screening tables parsed from {path}")
    return out


def build(screen, train_joints, min_healthy, p_healthy, min_train_cells,
          task_filter):
    """Selection uses healthy + seen joints ONLY. `heldout` is not a parameter
    of this function by design -- it cannot influence the result."""
    curriculum, why = {}, Counter()

    for key, rec in sorted(screen.items()):
        if task_filter and key not in task_filter:
            continue
        cells, healthy = rec["cells"], rec["healthy"]

        if healthy < min_healthy:
            why["healthy_too_low"] += 1
            continue

        usable = [j for j in train_joints if cells[j][1] == USABLE]
        if len(usable) < min_train_cells:
            why["too_few_usable_train_joints"] += 1
            continue

        # Weight seen joints by headroom: a cell at 20% has more room to move
        # than one at 80%, and headroom is what sparse RL can actually reach.
        headroom = {j: max(healthy - cells[j][0], 1) for j in usable}
        tot = sum(headroom.values())
        pool = [str(j) for j in usable] + ["healthy"]
        probs = [(1.0 - p_healthy) * headroom[j] / tot for j in usable]
        probs.append(p_healthy)

        curriculum[key] = {
            "joint_pool": pool,
            "fault_probs": [round(p, 4) for p in probs],
            "screen_healthy": healthy,
            "screen_train_cells": {f"j{j}": cells[j][0] for j in usable},
            "screen_skipped": {
                f"j{j}": {"pct": cells[j][0],
                          "class": {"=": "NO_EFFECT", "0": "NO_SIGNAL",
                                    "w": "WEAK_TASK"}.get(cells[j][1], "?")}
                for j in train_joints if cells[j][1] != USABLE
            },
        }
    return curriculum, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--train_joints", type=int, nargs="+", default=[0, 4, 5, 6])
    ap.add_argument("--heldout", type=int, default=2,
                    help="recorded and excluded from training; NEVER used to "
                         "select or rank tasks")
    ap.add_argument("--min_healthy", type=int, default=80)
    ap.add_argument("--p_healthy", type=float, default=0.10)
    ap.add_argument("--min_train_cells", type=int, default=1,
                    help="minimum USABLE seen joints required per task")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.heldout in a.train_joints:
        sys.exit(f"[fatal] held-out j{a.heldout} is also in --train_joints")

    screen = parse_screening(a.log)
    print(f"[info] parsed {len(screen)} tasks from {a.log}")

    curriculum, why = build(
        screen, a.train_joints, a.min_healthy, a.p_healthy,
        a.min_train_cells, set(a.tasks) if a.tasks else None,
    )
    if not curriculum:
        sys.exit("[fatal] no tasks survived selection")
    if a.tasks:
        missing = sorted(set(a.tasks) - set(curriculum))
        if missing:
            sys.exit(
                "[fatal] explicitly requested task(s) failed held-out-blind "
                f"selection: {missing}. Choose replacements or change the seen-joint "
                "selection rule; do not silently shrink a diagnostic."
            )

    tasks = sorted(curriculum)
    doc = {
        "schema": "jfcrl-curriculum-v2",
        "heldout_joint": str(a.heldout),
        "train_joints": [str(j) for j in a.train_joints],
        "tasks": tasks,
        "task_probs": [1.0 / len(tasks)] * len(tasks),
        "per_task": curriculum,
        "provenance": {
            # Keep generated curricula portable and avoid publishing a local
            # workstation or cluster path in their provenance.
            "screening_log": Path(a.log).name,
            "min_healthy": a.min_healthy,
            "min_train_cells": a.min_train_cells,
            "selection_used_heldout": False,
        },
    }
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)

    # Held-out values go in a SEPARATE file. Reporting only.
    ref_path = a.out + ".heldout_reference"
    ref = {k: {"heldout_frozen_pct": screen[k]["cells"][a.heldout][0],
               "heldout_class": screen[k]["cells"][a.heldout][1],
               "healthy": screen[k]["healthy"]}
           for k in tasks}
    with open(ref_path, "w") as f:
        json.dump({"heldout_joint": str(a.heldout),
                   "note": "reporting only; not used for task selection",
                   "per_task": ref}, f, indent=2)

    print(f"[info] kept {len(tasks)} tasks -> {a.out}")
    print(f"[info] held-out reference (not used in selection) -> {ref_path}")
    print(f"[info] rejected: {dict(why)}\n")

    print(f"{'task':24s} {'hlth':>5s}  {'train pool (screened %)':<34s} "
          f"{'p(fault)':<26s}")
    print("-" * 96)
    for k in tasks:
        c = curriculum[k]
        cells = " ".join(f"{j}={v}%" for j, v in c["screen_train_cells"].items())
        pr = " ".join(f"{p:.2f}" for p in c["fault_probs"])
        print(f"{k:24s} {c['screen_healthy']:4d}%  {cells:<34s} {pr:<26s}")

    n_cells = sum(len(curriculum[k]["screen_train_cells"]) for k in tasks)
    print(f"\n{len(tasks)} tasks, {n_cells} usable training cells "
          f"({n_cells/len(tasks):.1f} per task)")

    # Printed last and separately, so it is visibly not part of selection.
    zero = [k for k in tasks if ref[k]["heldout_frozen_pct"] == 0]
    mean_ho = sum(ref[k]["heldout_frozen_pct"] for k in tasks) / len(tasks)
    print(f"\n--- held-out j{a.heldout} reference (NOT used in selection) ---")
    print(f"mean frozen j{a.heldout} over pool : {mean_ho:.1f}%")
    print(f"tasks with j{a.heldout} at 0%      : {len(zero)}/{len(tasks)}")
    print("These 0% cells are kept. They are valid evaluation cells. Prior runs")
    print("show that some can recover with roughly 25--40k transitions/task, while")
    print("many remained at 0 around 26k/task; sufficient budget still matters.")


if __name__ == "__main__":
    main()
