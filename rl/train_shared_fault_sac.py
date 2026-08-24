"""
train_shared_fault_sac.py -- shared residual SAC, optionally conditioned on a
capability latent inferred from execution history.

    # vanilla baseline path, unchanged and still loadable
    --context_encoder none

    # capability-conditioned (the new method)
    --context_encoder gru --context_len 16 --context_hidden 128 \
    --context_dim 32 --encoder_lr 1e-4 --lambda_dyn 1.0 \
    --heldout_conditions 2

NEW: a temporal encoder E produces z from the last K context features; actor
and critics are conditioned on [obs, z]; a dynamics decoder predicts the
realized motion of the current step from (obs, action, z) under a Huber loss.
That loss is REPRESENTATION LEARNING, not reward shaping -- the RL reward is
still LIBERO task success alone and L_dyn is logged separately.

GRADIENT ROUTING (spec 4.2): z is DETACHED before it reaches actor and
critics by default, and the encoder is trained only by L_dyn on its own
optimizer. That is what keeps a representation failure and a control failure
distinguishable: if L_dyn falls but seen-fault success does not, the encoder
works and the controller does not. Three losses fighting over one encoder
from day one destroys that signal.
`--encoder_grad_from_critic` enables the coupled variant as an ablation.

ORIGINAL SHARED-PILOT NOTES BELOW STILL APPLY.

train_shared_fault_sac.py -- ONE residual policy over a pool of joint faults.

Pilot command (LIBERO Spatial Task 0, train j0 + j6, j2 never seen):

    python rl/train_shared_fault_sac.py \
      --task_id 0 --joint_pool 0 6 --fault_probs 0.5 0.5 \
      --total_steps 150000 --residual_scale 0.1 --alpha_init 0.01 \
      --warmup_mode actor --stratified_replay \
      --train_baselines <j0_on_0_39> <j6_on_0_39> --abort_on_collapse \
      --eval_every 0 --logdir runs/task0_shared_j0_j6_sac

==========================================================================
FIX 3 -- THE DRY RUN FORCES j0 -> j6 -> j0, IT DOES NOT SAMPLE
==========================================================================
Sampling three episodes and printing "switching verified" is unsound: with
any given seed the sampler can draw the same condition every time and the
transition that actually matters -- one compiled lock replaced by another --
never happens. `--dry_run` now walks every condition and returns to the
first, checking the COMPILED model after each reset AND again after
stepping (a constraint that fails under load looks identical to one that was
never compiled).

==========================================================================
FIX 4 -- THE COLLAPSE GUARD TAKES TRAIN-STATE BASELINES
==========================================================================
The guard compares against a rolling success rate computed over TRAINING
episodes, which run on states 0..39. Passing held-out (40..49) rates
compares two different state sets. j2 showed how large that gap can be:
~35% on screening states versus ~80% on held-out. The flag is
`--train_baselines` and it wants frozen success on states 0..39.

==========================================================================
FIX 5 -- HELD-OUT STATES ARE NOT TOUCHED DURING TRAINING
==========================================================================
`--eval_every 0` disables periodic evaluation entirely and is the default.
Repeatedly reading states 40..49 during training turns them into a
validation set. Train for a fixed budget, then evaluate once with
`eval_shared.py`. `eval_every=0` must never divide by zero -- there is a
test for that.

==========================================================================
EVERYTHING IS PER-FAULT
==========================================================================
With j0 and j6 in one pool, an aggregate 70% is equally consistent with
"j0 -> 90%, j6 held" and "j0 collapsed to 30%, j6 -> 100%". Separate rolling
deques, separate collapse guards, separate replay fractions, and no
averaging anywhere.
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
    # --- task / fault pool -------------------------------------------------
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--joint_pool", type=str, nargs="+", default=["0", "6"],
                   help="Tokens: '0', '6', 'healthy', '2:off=0.3', "
                        "'2:damp=50'. For the pilot: 0 6. j2 must NOT appear.")
    p.add_argument("--fault_probs", type=float, nargs="+", default=None)
    p.add_argument("--include_healthy", action="store_true")
    p.add_argument("--heldout_conditions", type=str, nargs="+", default=["2"],
                   help="Conditions that must NEVER enter training. The "
                        "trainer refuses to start if one appears in the pool: "
                        "a stray held-out condition invalidates the entire "
                        "zero-shot claim with nothing downstream noticing.")

    # --- capability context encoder (generalization spec section 3) --------
    p.add_argument("--context_encoder", choices=["none", "gru", "transformer"],
                   default="gru",
                   help="'none' reproduces the vanilla MLP action path "
                        "exactly and can load the preserved baseline ckpt. "
                        "'transformer' is an ablation, not the default.")
    p.add_argument("--context_len", type=int, default=16,
                   help="K control steps; 16 = two OpenVLA action chunks.")
    p.add_argument("--context_hidden", type=int, default=128)
    p.add_argument("--context_dim", type=int, default=32)
    p.add_argument("--context_include_time", action="store_true",
                   help="Add normalized episode time to the context feature. "
                        "Off by default (spec 3.1 marks it optional).")
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--lambda_dyn", type=float, default=1.0,
                   help="Weight on the one-step dynamics prediction loss. "
                        "This is REPRESENTATION learning, not reward shaping: "
                        "it never enters the environment reward.")
    p.add_argument("--lambda_slow", type=float, default=0.0,
                   help="Temporal consistency penalty on z. Start at 0.")
    p.add_argument("--gate_beta", type=float, default=0.0,
                   help="Novelty gate strength: residual is scaled by "
                        "exp(-beta * novelty), where novelty is the one-sided "
                        "z-score of the dynamics-decoder error against the "
                        "TRAINING conditions. 0.0 = ungated (the exact "
                        "current behaviour, and the ablation arm). Run "
                        "scripts/probe_novelty.py FIRST: if unseen locks do "
                        "not separate from seen ones, the gate never fires "
                        "and is a placebo.")
    p.add_argument("--gate_min", type=float, default=0.0,
                   help="Floor on the gate. 0.0 lets the residual go fully "
                        "silent and fall back to the frozen VLA.")
    p.add_argument("--encoder_grad_from_critic", action="store_true",
                   help="ABLATION: lets SAC gradients into the encoder. Off "
                        "by default so a representation failure and a control "
                        "failure stay distinguishable in the logs.")

    p.add_argument("--fault_block", type=int, default=1)
    p.add_argument("--n_eval_states", type=int, default=10)

    # --- residual ----------------------------------------------------------
    p.add_argument("--residual_scale", type=float, default=0.1)
    p.add_argument("--history_len", type=int, default=8)
    p.add_argument("--w_residual", type=float, default=0.0,
                   help="Headline condition is 0.0. Non-zero = ablation.")

    # --- SAC ---------------------------------------------------------------
    p.add_argument("--total_steps", type=int, default=150_000)
    p.add_argument("--start_steps", type=int, default=2_000)
    p.add_argument("--warmup_mode", choices=["actor", "uniform"], default="actor")
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
    p.add_argument("--no_zero_init_actor", action="store_true")
    p.add_argument("--stratified_replay", action="store_true",
                   help="Equal share per fault in every minibatch. 50/50 "
                        "EPISODES do not give 50/50 TRANSITIONS when one "
                        "fault's episodes run longer.")

    # --- guards ------------------------------------------------------------
    p.add_argument("--train_baselines", type=float, nargs="+", default=None,
                   help="FIX 4. Frozen-VLA success per condition ON TRAINING "
                        "STATES 0-39, in pool order. NOT the held-out rate: "
                        "the guard watches a rolling rate over training "
                        "episodes, so a held-out number compares different "
                        "state sets. Measure with eval_residual.py "
                        "--policy zero --joint_idx J --n_episodes 40.")
    p.add_argument("--collapse_margin", type=float, default=0.15)
    p.add_argument("--collapse_patience", type=int, default=4)
    p.add_argument("--abort_on_collapse", action="store_true")
    p.add_argument("--min_episodes_for_guard", type=int, default=15)

    # --- bookkeeping -------------------------------------------------------
    p.add_argument("--eval_every", type=int, default=0,
                   help="FIX 5. 0 = NEVER, and that is the default. Periodic "
                        "evaluation on held-out states turns them into a "
                        "validation set. Train a fixed budget, evaluate once.")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--logdir", type=str, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="FIX 3. Forced j0->j6->j0 switching check + "
                        "throughput. No RL updates.")
    return p.parse_args()


def encode_online(ctxmod, hist, K, ctx_dim):
    """Build the left-padded window from the live deque and encode it.

    Uses the SAME left padding and the SAME exclusive-of-current-step
    convention as the replay window builder. If the two disagreed, a
    checkpoint would behave differently in evaluation than it did in
    training and nothing would reveal it.
    """
    if ctxmod is None or ctxmod.latent_dim == 0:
        return None
    ctx = np.zeros((K, ctx_dim), dtype=np.float32)
    mask = np.zeros(K, dtype=np.float32)
    n = len(hist)
    if n:
        ctx[K - n:] = np.asarray(hist, dtype=np.float32)
        mask[K - n:] = 1.0
    return ctxmod.encode_numpy(ctx, mask)


def eval_per_fault(renv, agent, normalizer, eval_ids, n_episodes,
                   ctxmod=None, K=16):
    """Deterministic held-out eval, reported SEPARATELY per condition."""
    out = {}
    for spec in renv.faults.specs:
        n_succ, worst = 0, 0.0
        ids = list(eval_ids)[:n_episodes]
        for init_id in ids:
            obs = renv.reset(init_id=int(init_id), force_fault=spec)
            hist = deque(maxlen=K)
            done = False
            while not done:
                z = encode_online(ctxmod, hist, K, renv.ctx_dim)
                a = agent.act(normalizer(obs).astype(np.float32),
                              deterministic=True, z_np=z)
                obs, _, term, trunc, info = renv.step(a)
                hist.append(renv.last_context.copy())
                worst = max(worst, info["lock_drift"])
                done = term or trunc
            n_succ += int(info["success"])
        out[spec.name] = {"success": n_succ, "n": len(ids),
                          "rate": n_succ / max(1, len(ids)),
                          "worst_drift": worst}
    return out


def main():
    args = parse_args()
    set_headless_env()

    from experiments.robot.robot_utils import set_seed_everywhere

    import torch as _torch

    from faults.multi_fault import FaultSpec
    from rl.build import build_shared
    from rl.context_encoder import ContextModule
    from rl.residual_env import ResidualCfg
    from rl.sac import NStepReplayBuffer, RunningNorm, SACAgent

    set_seed_everywhere(args.seed)

    if args.encoder_grad_from_critic:
        raise SystemExit(
            "--encoder_grad_from_critic is disabled in this validated build. "
            "Use the default detached L_dyn-only encoder."
        )

    if args.logdir is None:
        pool = "_".join(str(j).replace(":", "").replace("=", "")
                        for j in args.joint_pool)
        if args.include_healthy:
            pool += "_healthy"
        base = (f"runs/task{args.task_id}_shared_{pool}_sac"
                if args.context_encoder == "none"
                else f"runs/task{args.task_id}_{args.context_encoder}_{pool}_sac")
        args.logdir = base if args.seed == 7 else f"{base}_seed{args.seed}"
    logdir = Path(args.logdir)
    if logdir.exists() and any(logdir.glob("ckpt_*.pt")):
        raise SystemExit(
            f"{logdir} already has checkpoints. Refusing to overwrite a "
            f"completed run. Move it aside or pass a different --logdir.")
    logdir.mkdir(parents=True, exist_ok=True)

    # HELD-OUT EXCLUSION, enforced rather than trusted. The held-out
    # condition must never contribute training gradients, encoder auxiliary
    # losses, replay samples, early stopping, or hyperparameter selection.
    from faults.multi_fault import joint_indices

    heldout = {FaultSpec.parse(t).name for t in (args.heldout_conditions or [])}
    # Exclude the whole JOINT, not just the exact condition name. `j2`,
    # `j2_off+0.2` and `j2_damp50` are three names and one joint; comparing
    # names alone would let a j2 variant into training and invalidate the
    # unseen-JOINT claim with nothing downstream noticing.
    heldout_joints = joint_indices(args.heldout_conditions)
    pool_joints = joint_indices(args.joint_pool)
    joint_clash = sorted(heldout_joints & pool_joints)
    if joint_clash:
        offenders = [t for t in args.joint_pool
                     if FaultSpec.parse(t).joint_idx in joint_clash]
        raise SystemExit(
            f"held-out JOINT(S) {joint_clash} appear in --joint_pool via "
            f"{offenders}. Every variant of a held-out joint -- lock, "
            f"offset, damping -- must be absent from training, or the "
            f"unseen-joint claim is void.")

    pool_names = {FaultSpec.parse(t).name for t in args.joint_pool}
    clash = sorted(heldout & pool_names)
    if clash:
        raise SystemExit(
            f"held-out condition(s) {clash} appear in --joint_pool. Remove "
            f"them from the pool or change --heldout_conditions. A stray "
            f"held-out condition invalidates the zero-shot claim silently.")

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

    renv, train_ids, eval_ids, meta = build_shared(
        cfg, res_cfg,
        joint_pool=tuple(args.joint_pool),
        fault_probs=tuple(args.fault_probs) if args.fault_probs else None,
        include_healthy=args.include_healthy,
        fault_block=args.fault_block,
        n_eval_states=args.n_eval_states,
        seed=args.seed,
        context_include_time=args.context_include_time,
    )
    names = renv.faults.names
    base_obs_dim, act_dim = meta["obs_dim"], meta["act_dim"]
    ctx_dim, K = meta["ctx_dim"], args.context_len

    ctxmod = ContextModule(
        ctx_dim=ctx_dim, obs_dim=base_obs_dim, act_dim=act_dim,
        kind=args.context_encoder, hidden=args.context_hidden,
        latent_dim=args.context_dim, context_len=K, lr=args.encoder_lr,
        lambda_dyn=args.lambda_dyn, lambda_slow=args.lambda_slow,
        gate_beta=args.gate_beta, gate_min=args.gate_min,
        detach_for_policy=True,
        device=args.device)
    z_dim = ctxmod.latent_dim
    # Conditioning is a CONCAT. With z_dim == 0 this is the vanilla path.
    obs_dim = base_obs_dim + z_dim
    print(json.dumps({"meta": meta, "pool": names,
                      "heldout_excluded": sorted(heldout),
                      "context_encoder": args.context_encoder,
                      "context_len": K, "ctx_dim": ctx_dim, "z_dim": z_dim,
                      "obs_dim_into_sac": obs_dim,
                      "probs": [float(p) for p in renv.faults.probs],
                      "n_train_states": len(train_ids),
                      "checkpoint": cfg.pretrained_checkpoint},
                     indent=2, default=str))

    if not renv.faults.env_factory:
        raise SystemExit(
            "the fault manager has no env_factory. Switching faults on a "
            "reused robosuite env can silently compile NO constraint. "
            "build_shared must pass env_factory=lambda: build_env(cfg)[0].")

    if args.w_residual > 0:
        print("\n*** w_residual > 0: ABLATION run, not the task-reward-only "
              "headline condition. Label it. ***\n")

    baselines = {}
    if args.train_baselines:
        if len(args.train_baselines) != len(names):
            raise SystemExit(
                f"--train_baselines has {len(args.train_baselines)} values "
                f"but the pool has {len(names)}: {names}")
        baselines = dict(zip(names, args.train_baselines))
        print(f"per-fault collapse guard on TRAIN states "
              f"0-{len(train_ids) - 1}: {baselines}")
    else:
        print("no --train_baselines: per-fault collapse guard is OFF. An "
              "aggregate rate can hide one condition collapsing.")

    # ---------------------------------------------------------- FIX 3 -----
    if args.dry_run:
        specs = list(renv.faults.specs)
        sequence = specs + [specs[0]] if len(specs) > 1 else specs * 2
        print("\n=== FORCED constraint-switching check ===")
        print("sequence: " + " -> ".join(sp.name for sp in sequence))
        print("(forced, not sampled: three sampled episodes can draw the "
              "same condition and never exercise a switch at all)")

        from faults.multi_fault import count_active_fault_locks

        t0, steps, per = time.time(), 0, {}
        z = np.zeros(act_dim, dtype=np.float32)
        for i, spec in enumerate(sequence):
            renv.reset(init_id=int(train_ids[0]), force_fault=spec)
            renv.faults.assert_exactly_one_lock(renv.env)
            print(f"  [{i + 1}/{len(sequence)}] forced {spec.name:>8} -> "
                  f"compiled: {count_active_fault_locks(renv.env) or '[] (none)'}")
            done, n = False, 0
            while not done:
                _, _, term, trunc, info = renv.step(z)
                steps += 1
                n += 1
                done = term or trunc
            # re-check AFTER stepping: a constraint that fails under load
            # looks identical to one that was never compiled
            renv.faults.assert_exactly_one_lock(renv.env)
            drift = renv.faults.monitor.max_drift if renv.faults.monitor else 0.0
            per.setdefault(spec.name, []).append(n)
            print(f"       {n} steps, success={info['success']}, "
                  f"max drift={drift:.3e} rad")
            if drift > 1e-2:
                raise SystemExit(
                    f"lock on {spec.name} drifted {drift:.3e} rad -- the "
                    f"constraint is not holding under load. Do not train.")

        dt = time.time() - t0
        sps = steps / dt
        print("\n" + "=" * 66)
        print(f"{steps} steps in {dt:.1f}s -> {sps:.2f} env-steps/s")
        print(f"projected for {args.total_steps} steps: "
              f"{args.total_steps / sps / 3600:.1f} h")
        print("\nzero-residual episode lengths per condition (this drives "
              "replay imbalance):")
        for n_, lens in per.items():
            print(f"  {n_:>8}: {lens}")
        print("\nEvery forced transition compiled exactly the intended "
              "constraint, verified against the model itself.")
        print(json.dumps(renv.faults.stats(), indent=2, default=str))
        print("=" * 66)
        return

    # -------------------------------------------------------------- setup --
    agent = SACAgent(obs_dim, act_dim, device=args.device, hidden=args.hidden,
                     lr=args.lr, gamma=args.gamma, tau=args.tau,
                     alpha_init=args.alpha_init,
                     zero_init_actor=not args.no_zero_init_actor,
                     log_std_init=args.log_std_init)
    if not args.no_zero_init_actor:
        # Must hold for EVERY context, not just a zero one: the residual has
        # to start at exactly the frozen VLA whatever the latent says.
        for probe in (_torch.zeros(1, obs_dim, device=agent.device),
                      _torch.randn(1, obs_dim, device=agent.device) * 5.0):
            _a0, _ = agent.actor(probe, deterministic=True, with_logp=False)
            assert float(_a0.abs().max()) < 1e-6, "zero-init actor is not zero"
        print("zero-init verified for zero AND random context: step-0 "
              "deterministic residual is exactly 0")

    buf = NStepReplayBuffer(base_obs_dim, act_dim, capacity=args.buffer_size,
                            n_step=args.n_step, gamma=args.gamma,
                            n_faults=len(names),
                            ctx_dim=ctx_dim if z_dim > 0 else 0,
                            context_len=K)
    normalizer = RunningNorm(base_obs_dim)

    if args.wandb:
        import wandb
        wandb.init(project="vla-fault-residual", name=logdir.name,
                   config=vars(args))

    rng = np.random.default_rng(args.seed)
    succ = {n: deque(maxlen=50) for n in names}
    n_eps = {n: 0 for n in names}
    n_below = {n: 0 for n in names}
    res_by_fault = {n: deque(maxlen=2000) for n in names}
    trans_by_fault = {n: 0 for n in names}
    metrics, ctx_metrics, warned_q = {}, {}, False
    t0 = time.time()

    obs = renv.reset()
    renv.faults.assert_exactly_one_lock(renv.env)
    normalizer.update(obs)
    hist = deque(maxlen=K)
    last_err = 0.0
    gate_by_fault = {n: deque(maxlen=2000) for n in names}
    err_by_fault = {n: deque(maxlen=2000) for n in names}
    eval_pending = False

    for step in range(1, args.total_steps + 1):
        cur = renv.faults.active.name
        fid = renv.faults.index_of(cur)

        z_np = encode_online(ctxmod, hist, K, ctx_dim)
        # Gate for THIS step, from the novelty of the step just completed.
        # It uses only training-time statistics, so it cannot leak anything
        # about a held-out condition.
        gate_t = ctxmod.gate(last_err) if z_dim > 0 else 1.0
        if step <= args.start_steps:
            if args.warmup_mode == "uniform":
                a = rng.uniform(-1, 1, size=act_dim).astype(np.float32)
            else:
                # zero-centred stochastic actor: explores without shoving a
                # base policy that already works off course
                a = agent.act(normalizer(obs).astype(np.float32),
                              deterministic=False, z_np=z_np)
        else:
            a = agent.act(normalizer(obs).astype(np.float32), z_np=z_np)

        next_obs, r, terminated, truncated, info = renv.step(a, gate=gate_t)
        normalizer.update(next_obs)

        # Dynamics error of the step just taken -> novelty for the NEXT step.
        # Stats update only on training conditions, which is all we ever see
        # here: held-out conditions never enter this loop.
        if z_dim > 0:
            _c = np.zeros((K, ctx_dim), dtype=np.float32)
            _m = np.zeros(K, dtype=np.float32)
            if hist:
                _c[K - len(hist):] = np.asarray(hist, dtype=np.float32)
                _m[K - len(hist):] = 1.0
            if _m.sum() >= 2:
                _e = ctxmod.per_sample_dyn_error(
                    _torch.as_tensor(_c, device=agent.device).unsqueeze(0),
                    _torch.as_tensor(_m, device=agent.device).unsqueeze(0),
                    # NORMALIZED, matching the decoder's training input.
                    # The replay batch passes obs_norm, so the decoder has
                    # only ever seen normalized observations; feeding raw
                    # ones here would query it off-distribution and inflate
                    # the error on every condition equally.
                    _torch.as_tensor(normalizer(obs).astype(np.float32),
                                     device=agent.device).unsqueeze(0),
                    _torch.as_tensor(a, device=agent.device).unsqueeze(0),
                    _torch.as_tensor(renv.last_dyn_target,
                                     device=agent.device).unsqueeze(0))
                last_err = float(_e.item())
                ctxmod.update_novelty_stats([last_err])

        # Push THIS step's context feature, then record the transition. The
        # window for obs is EXCLUSIVE of this push, so it is exactly what the
        # actor saw a moment ago -- and it cannot contain the realized motion
        # the dynamics decoder is asked to predict.
        s_now = (buf.push_context(renv.last_context, renv.episode_id,
                                  renv.t - 1) if z_dim > 0 else 0)
        # RAW observations; normalized at sample time against current stats
        buf.add(obs.astype(np.float32), a, r, next_obs.astype(np.float32),
                terminated, truncated, fault_id=fid,
                s_obs=s_now, s_next=s_now,
                context_episode_id=renv.episode_id,
                context_t=renv.t - 1,
                dyn_target=renv.last_dyn_target)
        if z_dim > 0:
            hist.append(renv.last_context.copy())
            ctxmod.update_target_stats(renv.last_dyn_target[None, :])
        obs = next_obs
        res_by_fault[cur].append(info["residual_norm"])
        gate_by_fault[cur].append(info["gate"])
        err_by_fault[cur].append(last_err)
        trans_by_fault[cur] += 1

        if terminated or truncated:
            succ[cur].append(int(info["success"]))
            n_eps[cur] += 1
            drift = renv.faults.monitor.max_drift if renv.faults.monitor else 0.0
            if drift > 1e-2:
                print(f"!! LOCK FAILED step {step} on {cur}: drift="
                      f"{drift:.3e} rad -- INFRASTRUCTURE failure, not an RL "
                      f"failure. This episode is not evidence.")

            if eval_pending:
                eval_pending = False
                res = eval_per_fault(renv, agent, normalizer, eval_ids,
                                     args.eval_episodes, ctxmod=ctxmod, K=K)
                print(f"\n>>> HELD-OUT @ {step}")
                for n, v in res.items():
                    print(f"    {n:>8}: {v['success']}/{v['n']} = "
                          f"{v['rate']:5.0%}  drift={v['worst_drift']:.1e}")
                print()
                if args.wandb:
                    import wandb
                    wandb.log({"step": step, **{
                        f"eval/{n}_success": v["rate"] for n, v in res.items()}})
                (logdir / f"heldout_{step}.json").write_text(
                    json.dumps(res, indent=2))

            buf.end_episode()
            obs = renv.reset()
            normalizer.update(obs)
            hist.clear()      # context NEVER crosses an episode boundary
            last_err = 0.0    # novelty must not carry across episodes

        if step > args.start_steps and buf.size >= args.batch_size:
            for _ in range(args.utd):
                batch = buf.sample(args.batch_size, agent.device,
                                   obs_norm=normalizer,
                                   stratified=args.stratified_replay)
                if z_dim > 0:
                    # Representation step first, on its OWN optimizer, then
                    # SAC on a detached latent.
                    ctx_metrics = ctxmod.update(
                        batch["ctx"], batch["mask"], batch["obs"],
                        batch["act"], batch["dyn_target"])
                    z = ctxmod.encode_for_policy(batch["ctx"], batch["mask"])
                    nz = ctxmod.encode_for_policy(batch["next_ctx"],
                                                  batch["next_mask"])
                    metrics = agent.update(batch, z=z, next_z=nz)
                else:
                    metrics = agent.update(batch)

        # ------------------------------------------------------ log ------
        if step % args.log_every == 0:
            sps = step / (time.time() - t0)
            log = {
                "step": step,
                "throughput/env_steps_per_s": sps,
                "throughput/eta_hours": (args.total_steps - step) / sps / 3600,
                "buffer/size": buf.size,
                "fault/env_rebuilds": renv.faults.stats()["fault/n_env_rebuilds"],
                **metrics, **ctx_metrics,
            }
            fracs = buf.fault_fractions()
            for n in names:
                log[f"train/success_{n}"] = (
                    float(np.mean(succ[n])) if succ[n] else float("nan"))
                log[f"train/episodes_{n}"] = n_eps[n]
                log[f"residual/norm_{n}"] = (
                    float(np.mean(res_by_fault[n])) if res_by_fault[n]
                    else float("nan"))
                log[f"buffer/frac_{n}"] = fracs.get(renv.faults.index_of(n), 0.0)
                log[f"gate/mean_{n}"] = (float(np.mean(gate_by_fault[n]))
                                         if gate_by_fault[n] else float("nan"))
                log[f"ctx/dyn_err_{n}"] = (float(np.mean(err_by_fault[n]))
                                           if err_by_fault[n] else float("nan"))
                log[f"buffer/mean_ep_len_{n}"] = (
                    trans_by_fault[n] / n_eps[n] if n_eps[n] else float("nan"))
            print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float)
                             else f"{k}={v}" for k, v in log.items()))
            if args.wandb:
                import wandb
                wandb.log(log)

            q = metrics.get("sac/q_mean")
            if q is not None and not np.isfinite(q):
                raise SystemExit(f"sac/q_mean is {q} -- NaN/Inf in the critic. "
                                 f"Stop; the run is not recoverable.")
            if q is not None and not warned_q and abs(q) > 5.0:
                warned_q = True
                print(f"\n!! sac/q_mean = {q:.1f} but the maximum discounted "
                      f"task return is 1.0. The critic is measuring ENTROPY, "
                      f"not task value. Lower --alpha_init and restart.\n")

            # ---- PER-FAULT collapse guard (FIX 4: train-state baselines) --
            for n in names:
                b = baselines.get(n)
                if b is None or n_eps[n] < args.min_episodes_for_guard:
                    continue
                cur_rate = float(np.mean(succ[n]))
                if cur_rate < b - args.collapse_margin:
                    n_below[n] += 1
                    print(f"   [collapse watch {n}: {n_below[n]}/"
                          f"{args.collapse_patience}] {cur_rate:.0%} vs frozen "
                          f"train-state {b:.0%} over {n_eps[n]} episodes")
                    if (args.abort_on_collapse
                            and n_below[n] >= args.collapse_patience):
                        rates = {k: (round(float(np.mean(v)), 3) if v else None)
                                 for k, v in succ.items()}
                        raise SystemExit(
                            f"\nABORTED at step {step}: condition '{n}' stayed "
                            f">{args.collapse_margin:.0%} below its frozen "
                            f"train-state baseline for {n_below[n]} checks.\n"
                            f"Other conditions may look fine -- that is "
                            f"exactly what a per-fault guard is for.\n"
                            f"per-condition rates: {rates}\n")
                else:
                    n_below[n] = 0

        # FIX 5: eval_every == 0 means never, and must not divide by zero.
        if args.eval_every and step % args.eval_every == 0:
            eval_pending = True

        if step % args.ckpt_every == 0 or step == args.total_steps:
            import torch
            torch.save({
                "agent": agent.state_dict(),
                "context": ctxmod.state_dict() if z_dim > 0 else None,
                "context_encoder": args.context_encoder,
                "context_len": K,
                "context_hidden": args.context_hidden,
                "context_dim": args.context_dim,
                "context_include_time": args.context_include_time,
                "ctx_dim": ctx_dim, "z_dim": z_dim,
                "base_obs_dim": base_obs_dim,
                "lambda_dyn": args.lambda_dyn,
                "gate_beta": args.gate_beta, "gate_min": args.gate_min,
                "heldout_conditions": sorted(heldout),
                "heldout_joints": sorted(heldout_joints),
                "obs_norm": normalizer.state_dict(),
                "args": vars(args),
                "meta": meta,
                "step": step,
                "joint_pool": list(args.joint_pool),
                "include_healthy": args.include_healthy,
                "fault_probs": [float(p) for p in renv.faults.probs],
                "fault_names": names,
                "residual_scale": args.residual_scale,
                "alpha_init": args.alpha_init,
                "warmup_mode": args.warmup_mode,
                "stratified_replay": args.stratified_replay,
                "seed": args.seed,
                "pretrained_checkpoint": cfg.pretrained_checkpoint,
                "train_baselines": baselines,
                "baseline_state_set": "train_0_to_39",
                "replay_fraction_by_fault": {
                    n: trans_by_fault[n] / max(1, step) for n in names},
            }, logdir / f"ckpt_{step}.pt")

    print(f"\ndone. checkpoints in {logdir}")
    print(json.dumps(renv.faults.stats(), indent=2, default=str))
    print(f"replay fractions: {buf.fault_fractions()}")


if __name__ == "__main__":
    main()
