"""
train_residual_sac.py -- train the residual on the validated Task 0 + j0 pair.

Frozen OpenVLA-OFT. Physically locked j0. Task reward only. No reference
trajectory, no fault demonstrations, no fault label in the observation.

--------------------------------------------------------------------------
READ THE THROUGHPUT MATH BEFORE LAUNCHING
--------------------------------------------------------------------------
One env step costs an OSC solve, a MuJoCo step, two 256x256 renders, and --
every 8th step -- a 7B forward pass. An episode is up to 220 steps, so ~28
VLA queries. Measure the real rate with `--dry_run` before committing a
night to it:

    env-steps/s | 100k steps
    ------------|-----------
        5       |  5.6 h
       10       |  2.8 h
       20       |  1.4 h

`--dry_run` runs 3 episodes with a zero residual and prints steps/s, VLA
queries, and the projected wall clock for `--total_steps`. If the projection
is longer than the time you have, cut `--total_steps` or raise `--utd` and
accept fewer environment interactions -- do not launch and hope.

--------------------------------------------------------------------------
WHY THE UPDATE-TO-DATA RATIO IS HIGH
--------------------------------------------------------------------------
Standard SAC does one gradient step per env step. Here an env step costs
~1e5 times more than a gradient step on a 2x256 MLP, so gradient steps are
free by comparison. `--utd 4` extracts four times the learning from the same
expensive interaction. Push it further if the critic loss stays stable;
back off if Q values diverge.

--------------------------------------------------------------------------
WHAT TO WATCH, AND WHAT EACH FAILURE LOOKS LIKE
--------------------------------------------------------------------------
* train/success_rate near this joint's screening rate and flat -> residual
  doing nothing yet (normal for the first ~10k steps; the buffer is still
  mostly baseline behaviour). Task 0 screening rates on states 0..19:
  j0 20%, j1 0%, j2 35%, j3 0%, j4 100%, j5 100%, j6 60%.
* train/success_rate collapsing below 0.05     -> residual_scale too large;
  RL is overwriting the VLA rather than correcting it. Halve it.
* sac/q_mean growing without bound             -> lower utd or lr
* sac/entropy pinned at its maximum            -> alpha too high, no
  exploitation; check target_entropy = -6
* fault/max_drift_rad rising above 1e-2        -> THE LOCK IS FAILING. Stop.
  The run is not measuring fault recovery.
* residual/clipped_frac near 1.0               -> the sum saturates at the
  action bound constantly; the residual has no headroom in that direction

The one number that decides everything is held-out success against THIS
JOINT's frozen-VLA held-out baseline on states 40-49 -- which you must
measure before training and pass as `--heldout_baseline`. Do not reuse j0's:
its screening rate was 20% but its held-out baseline was 50%, and j2's is
unmeasured until you run it. Everything else is diagnosis.
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
    # task / fault
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--joint_idx", type=int, default=0)
    p.add_argument("--n_eval_states", type=int, default=10)

    # residual
    p.add_argument("--residual_scale", type=float, default=0.2)
    p.add_argument("--history_len", type=int, default=8)

    # reward: both default OFF -> headline condition is task-reward-only
    p.add_argument("--w_residual", type=float, default=0.0,
                   help="Penalty lambda on ||delta||^2, applied every step. "
                        "Non-zero means this run is NOT the task-reward-only "
                        "condition and must be reported as a separate "
                        "arm. Note it references no healthy or reference "
                        "trajectory, so it does not compromise the claim that "
                        "separates this work from J-PARC -- but it does break "
                        "comparability with the j0 run, which used 0.0. "
                        "Suggested value if needed: 0.01.")
    p.add_argument("--collapse_margin", type=float, default=0.15,
                   help="Abort if rolling success falls this far below "
                        "--heldout_baseline for --collapse_patience checks.")
    p.add_argument("--collapse_patience", type=int, default=4)
    p.add_argument("--abort_on_collapse", action="store_true",
                   help="Stop automatically instead of burning hours watching "
                        "a residual destroy a working base policy.")

    # SAC
    p.add_argument("--total_steps", type=int, default=100_000)
    p.add_argument("--start_steps", type=int, default=2_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--utd", type=int, default=4)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--alpha_init", type=float, default=0.01,
                   help="Initial SAC temperature. NOT 1.0. See sac.py: with a "
                        "reward of 1.0 once per ~200 steps, alpha=1.0 makes the "
                        "critic ~150x the maximum task return and the actor "
                        "optimizes entropy instead of success.")
    p.add_argument("--log_std_init", type=float, default=-1.0,
                   help="Initial exploration width. std=exp(-1)=0.37.")
    p.add_argument("--no_zero_init_actor", action="store_true",
                   help="Disable zero-initialization of the actor head. Only "
                        "for reproducing the pre-fix j6 collapse.")
    p.add_argument("--buffer_size", type=int, default=300_000)

    # bookkeeping
    p.add_argument("--eval_every", type=int, default=10_000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--logdir", type=str, default=None,
                   help="default: runs/task0_j{joint_idx}_sac -- never write "
                        "j2 checkpoints into the frozen j0 run directory")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="3 zero-residual episodes; report throughput; exit")
    p.add_argument("--heldout_baseline", type=float, default=None,
                   help="Frozen-VLA held-out success on states 40-49 for THIS "
                        "joint, as a fraction. Purely for readable log lines: "
                        "every held-out eval is printed against it. Measure it "
                        "first with `eval_residual.py --policy zero --held_out`. "
                        "Leaving it unset is safe but the logs then have no "
                        "reference point, and j0's 50% is NOT j2's.")
    return p.parse_args()


def rollout_eval(renv, agent, normalizer, eval_ids, n_episodes, act_dim):
    """Deterministic held-out evaluation. Returns (success_rate, worst_drift)."""
    n_succ, worst = 0, 0.0
    ids = eval_ids[:n_episodes]
    for init_id in ids:
        obs = renv.reset(init_id=int(init_id))
        done = False
        while not done:
            a = agent.act(normalizer(obs).astype(np.float32), deterministic=True)
            obs, _, term, trunc, info = renv.step(a)
            worst = max(worst, info["lock_drift"])
            done = term or trunc
        n_succ += int(info["success"])
    return n_succ / max(1, len(ids)), worst


def main():
    args = parse_args()
    set_headless_env()

    from experiments.robot.robot_utils import set_seed_everywhere

    from rl.build import build_all
    from rl.residual_env import ResidualCfg
    from rl.sac import NStepReplayBuffer, RunningNorm, SACAgent

    set_seed_everywhere(args.seed)
    if args.logdir is None:
        # Seed 7 (the default, and the one j0 was trained with) keeps the
        # plain name the handoff specifies. Any other seed gets its own
        # directory, so a multi-seed sweep -- which is what turns this from
        # one draw into an inferential claim -- cannot clobber itself.
        base = f"runs/task{args.task_id}_j{args.joint_idx}_sac"
        args.logdir = base if args.seed == 7 else f"{base}_seed{args.seed}"
    logdir = Path(args.logdir)
    if logdir.exists() and any(logdir.glob("ckpt_*.pt")):
        raise SystemExit(
            f"{logdir} already contains checkpoints. Refusing to write into a "
            f"completed run -- handoff section 10 forbids mixing j0 and j2 "
            f"results. Pass a different --logdir, or move the old run aside."
        )
    logdir.mkdir(parents=True, exist_ok=True)

    cfg = VLACfg(task_id=args.task_id, seed=args.seed)
    res_cfg = ResidualCfg(
        residual_scale=args.residual_scale,
        history_len=args.history_len,
        w_residual=args.w_residual,
        gamma=args.gamma,
        max_steps=TASK_MAX_STEPS[cfg.task_suite_name],
        num_steps_wait=cfg.num_steps_wait,
        seed=args.seed,
    )

    renv, train_ids, eval_ids, meta = build_all(
        cfg, res_cfg, joint_idx=args.joint_idx, fault_enabled=True,
        n_eval_states=args.n_eval_states,
    )
    obs_dim, act_dim = meta["obs_dim"], meta["act_dim"]
    print(json.dumps({"meta": meta, "n_train_states": len(train_ids),
                      "n_eval_states": len(eval_ids)}, indent=2, default=str))

    if args.w_residual > 0:
        print("\n*** w_residual > 0: this is an ABLATION run, not the "
              "task-reward-only headline condition. Label it accordingly.\n")

    # ---------------------------------------------------------- dry run --
    if args.dry_run:
        t0, steps, ep = time.time(), 0, 0
        z = np.zeros(act_dim, dtype=np.float32)
        successes = 0
        for _ in range(3):
            renv.reset()
            done = False
            while not done:
                _, _, term, trunc, info = renv.step(z)
                steps += 1
                done = term or trunc
            successes += int(info["success"])
            ep += 1
        dt = time.time() - t0
        sps = steps / dt
        print("\n" + "=" * 62)
        print(f"episodes={ep}  steps={steps}  wall={dt:.1f}s  -> {sps:.2f} env-steps/s")
        print(f"VLA queries: {renv.vla.n_queries} "
              f"({steps / max(1, renv.vla.n_queries):.1f} steps per query)")
        print(f"zero-residual successes: {successes}/{ep} (expect ~20% over many)")
        print(f"projected wall clock for {args.total_steps} steps: "
              f"{args.total_steps / sps / 3600:.1f} h")
        print(f"lock drift so far: {renv.faults.monitor.max_drift:.3e} rad")
        print(json.dumps(renv.faults.stats(), indent=2))
        print("=" * 62)
        return

    # ------------------------------------------------------------ setup --
    agent = SACAgent(obs_dim, act_dim, device=args.device, hidden=args.hidden,
                     lr=args.lr, gamma=args.gamma, tau=args.tau,
                     alpha_init=args.alpha_init,
                     zero_init_actor=not args.no_zero_init_actor,
                     log_std_init=args.log_std_init)

    # The deterministic policy must start at exactly the frozen VLA. If this
    # ever fails, training begins from an arbitrary constant corruption of a
    # working controller -- the j6 failure mode.
    if not args.no_zero_init_actor:
        import torch as _t
        _probe = _t.zeros(1, obs_dim, device=agent.device)
        _a0, _ = agent.actor(_probe, deterministic=True, with_logp=False)
        _m = float(_a0.abs().max())
        assert _m < 1e-6, f"zero-init actor emits |a|={_m:.2e}, expected 0"
        print(f"zero-init verified: deterministic residual at step 0 = 0 "
              f"(exactly the frozen VLA)")
    buf = NStepReplayBuffer(obs_dim, act_dim, capacity=args.buffer_size,
                            n_step=args.n_step, gamma=args.gamma)
    normalizer = RunningNorm(obs_dim)

    if args.wandb:
        import wandb
        wandb.init(project="vla-fault-residual", name=Path(args.logdir).name,
                   config=vars(args))

    rng = np.random.default_rng(args.seed)
    ep_succ = deque(maxlen=50)
    metrics = {}
    t0 = time.time()

    obs = renv.reset()
    normalizer.update(obs)
    ep_ret, ep_len = 0.0, 0
    res_norms, clipped = [], []
    eval_pending = False
    n_below = 0
    warned_q = False

    for step in range(1, args.total_steps + 1):
        # ------------------------------------------------------- act ----
        if step <= args.start_steps:
            a = rng.uniform(-1, 1, size=act_dim).astype(np.float32)
        else:
            a = agent.act(normalizer(obs).astype(np.float32))

        next_obs, r, terminated, truncated, info = renv.step(a)
        normalizer.update(next_obs)
        # RAW observations go in. Normalization happens at sample time
        # against current statistics -- see NStepReplayBuffer's docstring.
        buf.add(obs.astype(np.float32), a, r,
                next_obs.astype(np.float32), terminated, truncated)

        obs = next_obs
        ep_ret += r
        ep_len += 1
        res_norms.append(info["residual_norm"])
        clipped.append(info["clipped_frac"])

        if terminated or truncated:
            ep_succ.append(int(info["success"]))
            drift = renv.faults.monitor.max_drift if renv.faults.monitor else 0.0
            if drift > 1e-2:
                print(f"!! LOCK FAILED at step {step}: drift={drift:.3e} rad. "
                      f"This episode did not measure fault recovery.")

            # Held-out evaluation, deferred to here. It must never fire
            # mid-episode: `rollout_eval` resets the shared `renv` out from
            # under the current episode, and the n-step staging queue would
            # then chain that episode's tail onto the next one's head,
            # fabricating rewards across a boundary that never existed.
            if eval_pending:
                eval_pending = False
                sr, worst = rollout_eval(renv, agent, normalizer, eval_ids,
                                         args.eval_episodes, act_dim)
                if args.heldout_baseline is None:
                    ref = "baseline not supplied (--heldout_baseline)"
                else:
                    ref = (f"frozen-VLA held-out baseline "
                           f"{args.heldout_baseline:.0%}, "
                           f"delta {sr - args.heldout_baseline:+.0%}")
                print(f"\n>>> HELD-OUT @ {step}: {sr:.1%}  ({ref})  "
                      f"worst_drift={worst:.2e}\n")
                if args.wandb:
                    import wandb
                    wandb.log({"step": step, "eval/heldout_success": sr,
                               "eval/worst_drift": worst})

            # Belt and braces: `add` already flushed on the terminal
            # transition, so this is a no-op in the normal path. It is here
            # so that any future early-exit from an episode cannot silently
            # leave staged transitions to be chained onto the next one.
            buf.end_episode()

            obs = renv.reset()
            normalizer.update(obs)
            ep_ret, ep_len = 0.0, 0

        # ---------------------------------------------------- learn -----
        if step > args.start_steps and buf.size >= args.batch_size:
            for _ in range(args.utd):
                metrics = agent.update(
                    buf.sample(args.batch_size, agent.device, obs_norm=normalizer)
                )

        # ------------------------------------------------------ log -----
        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            log = {
                "step": step,
                "throughput/env_steps_per_s": sps,
                "throughput/eta_hours": (args.total_steps - step) / sps / 3600,
                "train/success_rate_last50": float(np.mean(ep_succ)) if ep_succ else 0.0,
                "train/episodes": len(ep_succ),
                "buffer/size": buf.size,
                "residual/norm_mean": float(np.mean(res_norms[-2000:])),
                "residual/clipped_frac": float(np.mean(clipped[-2000:])),
                "fault/max_drift_rad": (
                    renv.faults.monitor.max_drift if renv.faults.monitor else 0.0
                ),
                **metrics,
            }
            print(" | ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in log.items()
            ))

            # --- Q sanity. The single most informative diagnostic here. -----
            q = metrics.get("sac/q_mean")
            if q is not None and not warned_q and abs(q) > 5.0:
                warned_q = True
                print(
                    f"\n!! sac/q_mean = {q:.1f}, but the maximum achievable\n"
                    f"   discounted task return is 1.0 (and ~{args.gamma ** 200:.2f}\n"
                    f"   for a success at step 200). The critic is measuring\n"
                    f"   ENTROPY, not task value, so the actor is optimizing\n"
                    f"   entropy too. Lower --alpha_init and restart; letting\n"
                    f"   this run continue wastes the GPU time.\n"
                )

            # --- collapse guard --------------------------------------------
            if args.heldout_baseline is not None and len(ep_succ) >= 20:
                cur = float(np.mean(ep_succ))
                if cur < args.heldout_baseline - args.collapse_margin:
                    n_below += 1
                    print(f"   [collapse watch {n_below}/{args.collapse_patience}] "
                          f"rolling success {cur:.0%} vs baseline "
                          f"{args.heldout_baseline:.0%}")
                    if args.abort_on_collapse and n_below >= args.collapse_patience:
                        raise SystemExit(
                            f"\nABORTED at step {step}: rolling success {cur:.0%} "
                            f"has stayed >{args.collapse_margin:.0%} below the "
                            f"frozen baseline for {n_below} consecutive checks.\n"
                            f"The residual is destroying a working base policy. "
                            f"Do not train through this.\n"
                        )
                else:
                    n_below = 0
            if args.wandb:
                import wandb
                wandb.log(log)

        # -------------------------------------------------- held-out ----
        # Arm it here; it fires at the next episode boundary, above.
        if step % args.eval_every == 0:
            eval_pending = True

        if step % args.ckpt_every == 0 or step == args.total_steps:
            import torch
            torch.save(
                {
                    "agent": agent.state_dict(),
                    "obs_norm": normalizer.state_dict(),
                    "args": vars(args),
                    "meta": meta,
                    "step": step,
                },
                logdir / f"ckpt_{step}.pt",
            )

    print(f"\ndone. checkpoints in {logdir}")


if __name__ == "__main__":
    main()
