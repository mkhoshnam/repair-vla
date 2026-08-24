"""
compare_results.py -- builds the section 8.2 result table from the CSVs that
`eval_residual.py` already writes, with paired statistics.

    python scripts/compare_results.py --joints 0 2

--------------------------------------------------------------------------
WHAT YOU CAN AND CANNOT CLAIM FROM THIS
--------------------------------------------------------------------------
PRIMARY, and safe to report: the per-joint exact McNemar test. Within one
joint, the ten held-out initial states are ten independent scenes, the same
frozen policy and the same trained residual are evaluated on each, and the
discordant pairs are the evidence. That test is valid.

For j0 it gives p = 0.0625 two-sided (five discordant pairs, all favourable).
That does not clear 0.05. This is a consequence of n = 10, which is the
ceiling of a 40/10 split -- states 0..39 were trained on and can never serve
as held-out evidence. It is not a defect in the method: 10/10 versus 5/10 is
a large effect measured with a small sample.

NOT VALID, and removed from this script's output as a claim: pooling j0 and
j2 discordant pairs into one 20-pair McNemar test. An earlier version did
that and reported p ~ 0.002. It is wrong. The two joints are evaluated on
the SAME ten initial states, so a scene that is intrinsically hard is likely
hard under both faults. The pairs are clustered by initial state, and
treating twenty clustered observations as twenty independent ones inflates
significance by roughly the design effect.

WHAT IS REPORTED INSTEAD: a cluster-level exact sign test that takes the
INITIAL STATE as the unit of analysis and sums the effect across joints
within each state. Ten clusters, so the floor is p = 2^-10 ~ 0.001. This
respects the state-level dependence. It still does NOT account for training-
seed variability -- one training run per joint means the policy is a single
draw from the training distribution, and nothing here can tell you how much
of the effect is that draw.

THE ACTUAL FIX for a paper-grade claim is seeds: three training runs per
joint, then test on run-level held-out success rates, where the independent
unit is the training run rather than the episode. That is 6 runs for two
joints. At the throughput in the probe report that is affordable, and it is
what a reviewer will ask for. `train_residual_sac.py --seed N` already
supports it and writes to a per-seed directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


# --------------------------------------------------------------------------
# exact statistics, no scipy dependency
# --------------------------------------------------------------------------

def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant counts.

    b = baseline failed and the policy succeeded (improvements)
    c = baseline succeeded and the policy failed (regressions)
    Concordant pairs carry no information about the difference and drop out.
    """
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(b, c), n, 0.5))


def cluster_sign_test(per_state_delta: dict) -> tuple[float, int, int, int]:
    """Exact two-sided sign test with the INITIAL STATE as the unit.

    `per_state_delta[init_id]` is (improvements - regressions) summed over
    joints for that state. States with a net of zero are uninformative and
    drop out, exactly as ties do in an ordinary sign test.

    This is the honest way to combine joints that were evaluated on the same
    scenes: it never counts one scene twice, whatever happens inside it.
    """
    pos = sum(1 for d in per_state_delta.values() if d > 0)
    neg = sum(1 for d in per_state_delta.values() if d < 0)
    ties = sum(1 for d in per_state_delta.values() if d == 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg, ties
    return min(1.0, 2.0 * binom_cdf(min(pos, neg), n, 0.5)), pos, neg, ties


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval. Normal-approximation CIs are meaningless at
    k = n = 10 (they give zero width); Wilson stays honest at the boundary."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------


def find_csv(root: Path, joint: int, policy: str):
    """Locate a results CSV, tolerating the pre-j2 file layout.

    j0 was evaluated before eval_residual.py started writing per-joint
    directories and `_heldout` tags, so its CSVs sit at the old paths. Rather
    than make anyone move validated files by hand -- the exact operation that
    loses a result -- try the new name first and fall back.
    """
    candidates = [
        root / f"eval_j{joint}" / f"results_{policy}_j{joint}_heldout.csv",
        root / f"eval_j{joint}" / f"results_{policy}_j{joint}.csv",
        root / "eval" / f"results_{policy}_j{joint}.csv",
        root / f"results_{policy}_j{joint}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(root.rglob(f"results_{policy}_j{joint}*.csv"))
    return hits[0] if hits else None


def load(path):
    """init_id -> success, from an eval_residual.py results CSV."""
    if path is None or not Path(path).exists():
        return None
    with open(path) as f:
        return {int(r["init_id"]): int(r["success"]) for r in csv.DictReader(f)}


def paired(base: dict, treat: dict):
    """Discordant counts over the initial states both conditions share."""
    shared = sorted(set(base) & set(treat))
    b = sum(1 for i in shared if base[i] == 0 and treat[i] == 1)
    c = sum(1 for i in shared if base[i] == 1 and treat[i] == 0)
    return shared, b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--root", type=str, default="rollouts_residual")
    ap.add_argument("--out", type=str, default="results_table.json")
    args = ap.parse_args()

    root = Path(args.root)
    report = {}
    per_state_delta: dict = {}   # init_id -> net (improvements - regressions)

    print("=" * 74)
    print("HELD-OUT COMPARISON (init states 40-49)")
    print("=" * 74)

    for j in args.joints:
        p_base = find_csv(root, j, "zero")
        p_rl = find_csv(root, j, "ckpt")
        p_rand = find_csv(root, j, "random")
        base, rl, rand = load(p_base), load(p_rl), load(p_rand)

        if base is None or rl is None:
            missing = [n for n, v in (("zero", base), ("ckpt", rl)) if v is None]
            print(f"\nj{j}: SKIPPED, no results CSV for: {', '.join(missing)}")
            print(f"      searched under {root}/ for results_<policy>_j{j}*.csv")
            continue

        shared, b, c = paired(base, rl)
        nb, nr = sum(base[i] for i in shared), sum(rl[i] for i in shared)
        n = len(shared)
        p = exact_mcnemar(b, c)
        lo_b, hi_b = wilson_ci(nb, n)
        lo_r, hi_r = wilson_ci(nr, n)

        for i in shared:
            per_state_delta[i] = per_state_delta.get(i, 0) + (rl[i] - base[i])

        entry = {
            "n_heldout": n,
            "source_baseline_csv": str(p_base),
            "source_policy_csv": str(p_rl),
            "frozen_vla": {"k": nb, "n": n, "rate": nb / n,
                           "wilson95": [round(lo_b, 3), round(hi_b, 3)]},
            "residual_sac": {"k": nr, "n": n, "rate": nr / n,
                             "wilson95": [round(lo_r, 3), round(hi_r, 3)]},
            "discordant_improved": b,
            "discordant_regressed": c,
            "exact_mcnemar_two_sided_p": round(p, 5),
            "significant_at_05": p < 0.05,
        }
        if rand is not None:
            sh_r = sorted(set(base) & set(rand))
            entry["random_control"] = {
                "k": sum(rand[i] for i in sh_r), "n": len(sh_r)
            }
        report[f"j{j}"] = entry

        print(f"\nj{j}  (robot0_joint{j + 1})")
        print(f"  frozen VLA    {nb}/{n} = {nb / n:5.0%}   "
              f"95% CI [{lo_b:.2f}, {hi_b:.2f}]")
        print(f"  residual SAC  {nr}/{n} = {nr / n:5.0%}   "
              f"95% CI [{lo_r:.2f}, {hi_r:.2f}]")
        if rand is not None:
            k = entry["random_control"]["k"]
            m = entry["random_control"]["n"]
            print(f"  random ctrl   {k}/{m} = {k / max(1, m):5.0%}")
        else:
            print("  random ctrl   NOT RUN -- without it, 'RL learned a "
                  "compensation' is not\n                separable from "
                  "'perturbation helps a stuck policy'")
        print(f"  discordant    +{b} / -{c}")
        print(f"  exact McNemar two-sided p = {p:.4f}"
              f"{'  (p < .05)' if p < 0.05 else '  (NOT significant at .05)'}")

    if len(report) >= 2:
        pp, pos, neg, ties = cluster_sign_test(per_state_delta)
        report["across_joints"] = {
            "test": "exact two-sided sign test, unit = initial state",
            "joints": [k for k in report if k.startswith("j")],
            "states_net_improved": pos,
            "states_net_regressed": neg,
            "states_tied": ties,
            "p_value": round(pp, 6),
            "significant_at_05": pp < 0.05,
            "NOT_VALID_ALTERNATIVE": (
                "Do NOT pool per-joint discordant pairs into one McNemar "
                "test. The joints share the same ten initial states, so the "
                "pairs are clustered by scene and pooling them inflates "
                "significance."
            ),
            "REMAINING_LIMITATION": (
                "One training run per joint. This test does not account for "
                "training-seed variability; the policy is a single draw. "
                "Three seeds per joint, tested on run-level success rates, "
                "is what makes the claim reviewer-proof."
            ),
        }
        print("\n" + "-" * 74)
        print("ACROSS JOINTS -- exact sign test, unit = initial state")
        print(f"  states net improved {pos}, net regressed {neg}, tied {ties}")
        print(f"  two-sided p = {pp:.5f}"
              f"{'  (p < .05)' if pp < 0.05 else '  (NOT significant at .05)'}")
        print("  This treats each scene once. Pooling the per-joint discordant")
        print("  pairs into a single McNemar test would NOT be valid: both")
        print("  joints are evaluated on the same ten scenes, so a scene that")
        print("  is intrinsically hard contributes twice.")
        print("  Still unaccounted for: training-seed variability. One run per")
        print("  joint means the policy is a single draw. Three seeds per")
        print("  joint, compared on run-level rates, is the real fix.")
        print("-" * 74)
    elif len(report) == 1:
        j = next(iter(report))
        if not report[j]["significant_at_05"]:
            print(f"\nNOTE: {j} alone does not reach p < .05 two-sided. That is")
            print("an n = 10 ceiling from the 40/10 split, not a method problem.")
            print("Report the rate and CI honestly; add joints and seeds for power.")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
