"""
eval_shared.py -- evaluate ONE shared checkpoint under every condition.

Pilot command (handoff section 13):

    python rl/eval_shared.py --ckpt runs/task0_shared_j0_j6_sac/ckpt_150000.pt \
      --conditions healthy j0 j6 j2 --held_out --n_episodes 20 \
      --policies zero random ckpt

Produces the section 8 matrix: for every condition, the frozen VLA, the
random-residual control, and the shared policy, all on the SAME initial
states, with lock drift reported for each.

--------------------------------------------------------------------------
CONDITIONS ARE EXPLICIT, AND UNSEEN ONES ARE MARKED
--------------------------------------------------------------------------
`--conditions` is the full evaluation set. Conditions absent from the
checkpoint's training pool are labelled UNSEEN in the table and in the JSON.
The pilot needs healthy (does the residual damage nominal behaviour?),
j0 and j6 (seen recovery), and j2 (zero-shot unseen joint). Evaluating only
the trained pool would answer neither of the outer two questions.

--------------------------------------------------------------------------
NO RANDOM-RESIDUAL CONTROL BY DEFAULT
--------------------------------------------------------------------------
`--policies` defaults to zero + ckpt. The random-residual arm is available
but must be asked for explicitly, and runner scripts do not add it. The
comparison that matters now is frozen VLA vs vanilla shared SAC vs
capability-conditioned SAC.

--------------------------------------------------------------------------
THE CONTEXT ENCODER IS REBUILT FROM THE CHECKPOINT
--------------------------------------------------------------------------
Encoder kind, context length and latent width all come from the checkpoint,
never from the command line. A latent produced by a differently-shaped
encoder means something different, so a mismatch is refused rather than
silently tolerated. The online window uses the same left padding and the
same exclusive-of-current-step convention as training.

--------------------------------------------------------------------------
NO AVERAGING ACROSS CONDITIONS
--------------------------------------------------------------------------
Each condition has a different frozen baseline. A mean over them is not a
meaningful quantity and it hides the specific failure that sharing
introduces: one condition improving while another is forgotten.
"""

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


def encode_online(ctxmod, hist, K, ctx_dim):
    """Same left padding and same exclusive-of-current-step window as
    training. If evaluation built the window differently, a checkpoint would
    behave differently here than it did during training, invisibly."""
    if ctxmod is None or ctxmod.latent_dim == 0:
        return None
    ctx = np.zeros((K, ctx_dim), dtype=np.float32)
    mask = np.zeros(K, dtype=np.float32)
    n = len(hist)
    if n:
        ctx[K - n:] = np.asarray(hist, dtype=np.float32)
        mask[K - n:] = 1.0
    return ctxmod.encode_numpy(ctx, mask)


def rollout_condition(renv, spec, policy, ids, meta, agent=None,
                      normalizer=None, ctxmod=None, K=16, seed=7,
                      device="cpu", trained_names=(), verbose=False):
    """One condition x one policy over `ids`. Extracted so it can be TESTED.

    It was previously inline in main(), which is how a NameError on `g`,
    `gates` and `last_err` survived a 100/100 offline suite: the tests
    exercised the gate functions but never executed the evaluator. This
    function runs against fake envs in `test_offline.py`.

    OBSERVATION SCALE: the dynamics decoder is trained on NORMALIZED
    observations (the replay batch passes `obs_norm`). The novelty error
    must therefore be computed on `normalizer(obs)` too. Feeding raw
    observations here would query the decoder off its training
    distribution, inflate the error on EVERY condition including the seen
    ones, and make the novelty score meaningless.
    """
    import numpy as _np

    rng = _np.random.default_rng(seed)
    act_dim = meta["act_dim"]
    use_ctx = ctxmod is not None and getattr(ctxmod, "latent_dim", 0) > 0
    rows, n_succ, worst = [], 0, 0.0

    for k, init_id in enumerate(ids):
        obs = renv.reset(init_id=int(init_id), force_fault=spec)
        renv.faults.assert_exactly_one_lock(renv.env)
        hist = deque(maxlen=K)
        last_err, gates = 0.0, []
        done = False

        while not done:
            # Gate from the PREVIOUS step's dynamics error against the
            # novelty statistics saved IN THE CHECKPOINT. Never recomputed
            # here: recomputing on the conditions being evaluated would
            # redefine "familiar" to include the held-out fault, and the
            # gate would stop firing exactly where it is needed.
            g = ctxmod.gate(last_err) if use_ctx else 1.0
            z = None
            if policy == "zero":
                a = _np.zeros(act_dim, dtype=_np.float32)
                g = 1.0                      # nothing to gate
            elif policy == "random":
                a = rng.uniform(-1, 1, act_dim).astype(_np.float32)
                g = 1.0
            else:
                z = encode_online(ctxmod, hist, K, renv.ctx_dim)
                a = agent.act(normalizer(obs).astype(_np.float32),
                              deterministic=True, z_np=z)

            obs_before_norm = (normalizer(obs).astype(_np.float32)
                               if normalizer is not None else obs)
            obs, _, term, trunc, info = renv.step(a, gate=g)
            gates.append(info["gate"])

            if use_ctx:
                last_err = _dyn_error_online(
                    ctxmod, hist, K, renv.ctx_dim, obs_before_norm, a,
                    renv.last_dyn_target, device)

            hist.append(renv.last_context.copy())
            worst = max(worst, info["lock_drift"])
            done = term or trunc

        n_succ += int(info["success"])
        rows.append({
            "fault": spec.name, "policy": policy, "episode": k,
            "init_id": int(init_id), "success": int(info["success"]),
            "steps": info["t"],
            "seen_in_training": int(spec.name in trained_names),
            "mean_gate": round(float(_np.mean(gates)), 4),
            "max_lock_drift_rad": worst,
        })
        if verbose:
            print(f"  {spec.name:>12} {policy:>6} [{k + 1}/{len(ids)}] "
                  f"init={init_id} success={info['success']} "
                  f"gate={_np.mean(gates):.2f} running={n_succ}/{k + 1}")

    lock_ok = spec.is_healthy or worst <= 1e-2
    return {
        "rows": rows,
        "summary": {
            "success": n_succ, "n": len(ids),
            "rate": n_succ / max(1, len(ids)),
            "mean_gate": round(float(_np.mean(
                [r["mean_gate"] for r in rows])), 4) if rows else 1.0,
            "worst_lock_drift_rad": worst, "lock_ok": lock_ok,
            "seen_in_training": spec.name in trained_names,
        },
    }


def _dyn_error_online(ctxmod, hist, K, ctx_dim, obs_norm_np, act, y, device):
    """Per-step decoder error on NORMALIZED obs, matching training."""
    import torch as _t
    ctx = np.zeros((K, ctx_dim), dtype=np.float32)
    m = np.zeros(K, dtype=np.float32)
    if hist:
        ctx[K - len(hist):] = np.asarray(hist, dtype=np.float32)
        m[K - len(hist):] = 1.0
    if m.sum() < 2:
        return 0.0
    dev = _t.device(device)
    e = ctxmod.per_sample_dyn_error(
        _t.as_tensor(ctx, device=dev).unsqueeze(0),
        _t.as_tensor(m, device=dev).unsqueeze(0),
        _t.as_tensor(np.asarray(obs_norm_np, dtype=np.float32),
                     device=dev).unsqueeze(0),
        _t.as_tensor(np.asarray(act, dtype=np.float32), device=dev).unsqueeze(0),
        _t.as_tensor(np.asarray(y, dtype=np.float32), device=dev).unsqueeze(0))
    return float(e.item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--policies", nargs="+", default=["zero", "ckpt"],
                   choices=["zero", "ckpt"],
                   help="Default is zero + ckpt. RANDOM IS NOT INCLUDED BY "
                        "DEFAULT and must be requested explicitly (spec 4.1 "
                        "and 13). Runner scripts must not add it.")
    p.add_argument("--conditions", type=str, nargs="+", default=None,
                   help="Full evaluation set, e.g. healthy j0 j6 j2. "
                        "Conditions outside the checkpoint's training pool "
                        "are marked UNSEEN. Default: the training pool plus "
                        "healthy.")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--held_out", action="store_true")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--n_eval_states", type=int, default=10)
    p.add_argument("--residual_scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--outdir", type=str, default="rollouts_shared")
    p.add_argument("--save_video", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_headless_env()

    from experiments.robot.robot_utils import set_seed_everywhere

    from faults.multi_fault import FaultSpec, count_active_fault_locks
    from rl.build import build_shared
    from rl.residual_env import ResidualCfg

    set_seed_everywhere(args.seed)

    blob = None
    if "ckpt" in args.policies:
        if not args.ckpt:
            raise SystemExit("--policies ckpt requires --ckpt")
        import torch
        blob = torch.load(args.ckpt, map_location=args.device,
                          weights_only=False)

    trained_pool = list(blob["joint_pool"]) if blob else ["0", "6"]
    trained_names = {FaultSpec.parse(t).name for t in trained_pool}
    if blob and blob.get("include_healthy"):
        trained_names.add("healthy")

    conds = args.conditions or (list(trained_pool) + ["healthy"])
    specs, seen_names = [], set()
    for tok in conds:
        sp = FaultSpec.parse(tok)
        if sp.name in seen_names:
            continue
        seen_names.add(sp.name)
        specs.append(sp)

    scale = args.residual_scale
    if scale is None:
        scale = blob["residual_scale"] if blob else 0.1
    elif blob and abs(scale - blob["residual_scale"]) > 1e-9:
        raise SystemExit(
            f"residual_scale mismatch: checkpoint trained at "
            f"{blob['residual_scale']}, evaluating at {scale}. The actor's "
            f"outputs mean something different at a different scale.")

    cfg = VLACfg(task_id=args.task_id, seed=args.seed)
    res_cfg = ResidualCfg(
        residual_scale=scale,
        max_steps=TASK_MAX_STEPS[cfg.task_suite_name],
        num_steps_wait=cfg.num_steps_wait,
        seed=args.seed,
    )
    # Build with the FULL evaluation set so healthy and j2 get a proper
    # FaultSpec and the manager's env_factory handles every switch.
    renv, train_ids, eval_ids, meta = build_shared(
        cfg, res_cfg, joint_pool=tuple(specs),
        include_healthy=False,
        n_eval_states=args.n_eval_states, seed=args.seed,
        collect_images=args.save_video,
        context_include_time=bool((blob or {}).get("context_include_time", False)),
    )

    agent = normalizer = ctxmod = None
    K, z_dim = 16, 0
    if blob is not None:
        from rl.context_encoder import ContextModule
        from rl.sac import RunningNorm, SACAgent

        kind = blob.get("context_encoder", "none")
        K = int(blob.get("context_len", 16))
        ctx_dim_ck = int(blob.get("ctx_dim", renv.ctx_dim))
        if ctx_dim_ck != renv.ctx_dim:
            raise SystemExit(
                f"context feature width mismatch: checkpoint {ctx_dim_ck}, "
                f"env {renv.ctx_dim}. Check --context_include_time; the "
                f"encoder was trained on a different feature layout.")
        ctxmod = ContextModule(
            ctx_dim=renv.ctx_dim, obs_dim=meta["obs_dim"],
            act_dim=meta["act_dim"], kind=kind,
            hidden=int(blob.get("context_hidden", 128)),
            latent_dim=int(blob.get("context_dim", 32)),
            context_len=K, device=args.device)
        if blob.get("context"):
            ctxmod.load_state_dict(blob["context"])
        ctxmod.encoder.eval()
        if ctxmod.decoder is not None:
            ctxmod.decoder.eval()
        z_dim = ctxmod.latent_dim

        agent = SACAgent(meta["obs_dim"] + z_dim, meta["act_dim"],
                         device=args.device)
        agent.load_state_dict(blob["agent"])
        agent.actor.eval()
        normalizer = RunningNorm(meta["obs_dim"])
        normalizer.load_state_dict(blob["obs_norm"])
        print(f"context encoder: {kind}, K={K}, z_dim={z_dim}")

    ids = (list(eval_ids) if args.held_out
           else list(range(min(args.n_episodes, meta["n_init_states"]))))
    ids = ids[: args.n_episodes]
    if args.held_out and args.n_episodes > len(eval_ids):
        print(f"NOTE: only {len(eval_ids)} held-out states exist; "
              f"evaluating {len(ids)} episodes, not {args.n_episodes}. "
              f"Raise --n_eval_states before training to widen the split.")

    outdir = Path(args.outdir) / ("heldout" if args.held_out else "screen")
    outdir.mkdir(parents=True, exist_ok=True)

    table, rows = {}, []
    for spec in renv.faults.specs:
        table[spec.name] = {}
        for policy in args.policies:
            res = rollout_condition(
                renv=renv, spec=spec, policy=policy, ids=ids, meta=meta,
                agent=agent, normalizer=normalizer, ctxmod=ctxmod, K=K,
                seed=args.seed, device=args.device,
                trained_names=trained_names, verbose=True)
            rows.extend(res["rows"])
            table[spec.name][policy] = res["summary"]
            if not res["summary"]["lock_ok"]:
                print(f"  !! {spec.name}/{policy}: lock drifted "
                      f"{res['summary']['worst_lock_drift_rad']:.2e} rad "
                      f"-- INFRASTRUCTURE failure, not an RL result")

    with open(outdir / "per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "ckpt": args.ckpt,
        "pretrained_checkpoint": (blob or {}).get(
            "pretrained_checkpoint", cfg.pretrained_checkpoint),
        "trained_pool": sorted(trained_names),
        "evaluated_conditions": [s.name for s in renv.faults.specs],
        "unseen_conditions": [s.name for s in renv.faults.specs
                              if s.name not in trained_names],
        "init_state_set": "held_out" if args.held_out else "screen",
        "init_ids": [int(i) for i in ids],
        "residual_scale": scale,
        "seed": args.seed,
        "context_encoder": (blob or {}).get("context_encoder", "none"),
        "context_len": K, "z_dim": z_dim,
        "gate_beta": float(getattr(ctxmod, "gate_beta", 0.0)) if ctxmod else 0.0,
        "table": table,
        "NOTE": ("Per-condition only. Do not average across conditions: they "
                 "have different frozen baselines and a mean hides one "
                 "condition being forgotten while another improves."),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 76)
    hdr = (f"{'condition':>10} | " + " | ".join(f"{p:>9}" for p in args.policies)
           + " | seen?")
    print(hdr)
    print("-" * len(hdr))
    for name, per in table.items():
        cells = " | ".join(
            f"{per[p]['success']:>2}/{per[p]['n']:<2}={per[p]['rate']:.0%}"
            .rjust(9) for p in args.policies)
        seen = "trained" if per[args.policies[0]]["seen_in_training"] else "UNSEEN"
        print(f"{name:>10} | {cells} | {seen}")
    print("=" * 76)

    # --- interpretation, per handoff section 9 ---------------------------
    if "zero" in args.policies and "ckpt" in args.policies:
        print("\nRead per condition:")
        for name, per in table.items():
            z, c = per["zero"]["rate"], per["ckpt"]["rate"]
            r = per.get("random", {}).get("rate")
            tag = "seen" if per["zero"]["seen_in_training"] else "UNSEEN"
            if name == "healthy":
                verdict = ("nominal behaviour preserved" if c >= z - 0.1
                           else "RESIDUAL DAMAGES NOMINAL BEHAVIOUR")
                print(f"  healthy: frozen {z:.0%} -> shared {c:.0%}  ({verdict})")
                continue
            gmean = per["ckpt"].get("mean_gate")
            line = f"  {name} ({tag}): {z:.0%} -> {c:.0%}"
            if gmean is not None:
                line += f"  [gate {gmean:.2f}]"
            if c < z - 0.05 and gmean is not None and gmean > 0.7:
                line += "  <- NEGATIVE TRANSFER with the gate wide open"
            if r is not None:
                line += (f", random {r:.0%}"
                         + ("" if c > r else "  -- DOES NOT BEAT NOISE"))
            print(line)
        seen_ok = [n for n, per in table.items()
                   if per["zero"]["seen_in_training"] and n != "healthy"
                   and per["ckpt"]["rate"] > per["zero"]["rate"]]
        n_seen = sum(1 for n, per in table.items()
                     if per["zero"]["seen_in_training"] and n != "healthy")
        if n_seen:
            print(f"\nPILOT GATE: {len(seen_ok)}/{n_seen} seen faults improved "
                  f"over their frozen baseline with one shared policy.")
            if len(seen_ok) < n_seen:
                print("  Not a pass. A high average across conditions does "
                      "not substitute -- check the per-condition rows.")
        unseen = [n for n, per in table.items()
                  if not per["zero"]["seen_in_training"] and n != "healthy"]
        if unseen:
            print(f"\nUnseen-joint transfer is a SEPARATE result: {unseen}. "
                  f"A failure here does not invalidate the shared j0+j6 "
                  f"recovery result. Do not tune on it and then report it as "
                  f"unbiased generalization.")
    print(f"\nwritten to {outdir}")


if __name__ == "__main__":
    main()
