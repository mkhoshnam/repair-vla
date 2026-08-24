"""
jfcrl_v2_runtime.py -- glue between the curriculum JSON and the trainer.

Three pieces, each usable independently:

  load_curriculum()          per-task joint pools and fault probabilities
  generation_stratum_weights() + sample_weighted()
                             replay sampling at the configured generation
                             distribution instead of equal-share
  install_gate()             swap FiLMActor -> GatedFiLMActor on a built agent

See INTEGRATION.md for the exact edit points in
train_joint_factorized_multitask_sac.py.
"""
import json

import numpy as np
import torch


# --------------------------------------------------------------- curriculum
def load_curriculum(path):
    """-> (tasks, task_probs, per_task_pools, heldout_joint)

    per_task_pools[key] = {"joint_pool": [...], "fault_probs": [...]}
    """
    doc = json.load(open(path))
    if doc.get("schema") != "jfcrl-curriculum-v2":
        raise ValueError(f"{path}: not a jfcrl-curriculum-v2 file")

    tasks = list(doc["tasks"])
    probs = [float(x) for x in doc["task_probs"]]
    if len(probs) != len(tasks):
        raise ValueError("task_probs length must match tasks")
    ps = float(sum(probs))
    if ps <= 0 or abs(ps - 1.0) > 1e-3:
        raise ValueError(f"task_probs sum to {ps}, expected approximately 1.0")
    probs = [x / ps for x in probs]

    pools, heldout = {}, doc["heldout_joint"]
    for k in tasks:
        e = doc["per_task"][k]
        jp, fp = list(e["joint_pool"]), [float(x) for x in e["fault_probs"]]
        if len(jp) != len(fp):
            raise ValueError(f"{k}: joint_pool/fault_probs length mismatch")
        fs = float(sum(fp))
        if fs <= 0 or abs(fs - 1.0) > 1e-3:
            raise ValueError(f"{k}: fault_probs sum to {fs}, expected approximately 1.0")
        fp = [x / fs for x in fp]
        if heldout in jp:
            raise ValueError(
                f"{k}: held-out joint {heldout} present in training pool")
        pools[k] = {"joint_pool": jp, "fault_probs": fp}
    return tasks, probs, pools, heldout


def _canonical_fault_name(token):
    s = str(token)
    if s == "healthy":
        return s
    if s.startswith("j") and s[1:].isdigit():
        return s
    if s.isdigit():
        return f"j{s}"
    return s


# ------------------------------------------------------------ replay sampling
def generation_stratum_weights(stratum_id, tasks, task_probs, pools):
    """Target sampling weight per stratum = p(task) * p(fault | task).

    The shipped sampler (rl/sac.py:515, `stratified=True`) draws an EQUAL
    share from every task x fault stratum. With a 0.45/0.45/0.10 generation
    split that oversamples healthy by ~3.3x per task, and the distortion grows
    with the number of tasks. This restores the configured distribution.
    """
    w = np.zeros(len(stratum_id), dtype=np.float64)
    missing = []
    for tk, tp in zip(tasks, task_probs):
        e = pools[tk]
        for fname, fp in zip(e["joint_pool"], e["fault_probs"]):
            key = (tk, _canonical_fault_name(fname))
            if key not in stratum_id:
                missing.append(key)
                continue
            w[stratum_id[key]] = tp * fp
    if missing:
        raise ValueError(f"curriculum strata missing from replay mapping: {missing[:8]}")
    if w.sum() <= 0:
        raise ValueError("no stratum matched the curriculum; check fault names")
    return w / w.sum()


def sample_weighted(buf, batch_size, device, obs_norm=None, weights=None,
                    mode="generation", rng=None):
    """Replay sample with an explicit per-stratum target distribution.

    mode="generation"  draw stratum counts ~ Multinomial(batch, weights)
    mode="equal"       equal share per non-empty stratum (shipped behaviour)
    mode="uniform"     plain uniform over transitions

    Empty strata have their mass redistributed over non-empty ones, so early
    training (before every cell has data) does not silently shrink the batch.
    """
    if mode == "uniform" or buf.size == 0:
        return buf.sample(batch_size, device, obs_norm=obs_norm,
                          stratified=False)
    if mode == "equal":
        return buf.sample(batch_size, device, obs_norm=obs_norm,
                          stratified=True)
    if mode != "generation":
        raise ValueError(f"unknown sampler mode {mode!r}")
    if weights is None:
        raise ValueError("generation mode needs weights")

    rng = rng or np.random
    ids = buf.fault_id[: buf.size]
    pools = [np.flatnonzero(ids == k) for k in range(len(weights))]
    live = [k for k, p in enumerate(pools) if len(p) > 0]
    if len(live) <= 1:
        return buf.sample(batch_size, device, obs_norm=obs_norm,
                          stratified=False)

    w = np.asarray([weights[k] for k in live], dtype=np.float64)
    w = w / w.sum()
    counts = rng.multinomial(batch_size, w)

    idx = np.concatenate([
        rng.choice(pools[k], size=n, replace=True)
        for k, n in zip(live, counts) if n > 0
    ])
    rng.shuffle(idx)
    return buf.sample_at(idx, device, obs_norm=obs_norm)


def sampled_fraction(buf, idx):
    """Diagnostic: realised minibatch fraction per stratum.

    buffer/frac reports what is STORED. This reports what the optimizer
    actually received, which is the number that reveals whether the sampler
    did what the config said.
    """
    ids = buf.fault_id[idx]
    n = len(idx)
    return {int(k): float((ids == k).mean())
            for k in np.unique(ids)} if n else {}


# ------------------------------------------------------------------- gate
def install_gate(agent, obs_dim, z_dim, act_dim, args):
    """Replace agent.actor with GatedFiLMActor and rebuild its optimizer.

    Call immediately after the agent is constructed, before any update, so
    the target networks and optimizer state stay consistent.
    """
    try:
        from rl.jfcrl_v2_heads import GatedFiLMActor
    except ModuleNotFoundError:  # direct `python rl/jfcrl_v2_runtime.py` self-test
        from jfcrl_v2_heads import GatedFiLMActor

    base_actor_state = agent.actor.state_dict()
    actor = GatedFiLMActor(
        obs_dim, z_dim, act_dim,
        hidden=args.hidden,
        zero_init=True,
        log_std_init=args.log_std_init,
        gate_hidden=getattr(args, "gate_hidden", 64),
        gate_bias_init=getattr(args, "gate_bias_init", 4.0),
        gate_floor=getattr(args, "gate_floor", 0.0),
        logp_jacobian=not getattr(args, "gate_no_jacobian", False),
    ).to(agent.device)
    # Copy every shared FiLMActor parameter so A and B start from the same
    # actor initialization; only the new gate parameters differ.
    incompatible = actor.load_state_dict(base_actor_state, strict=False)
    if incompatible.unexpected_keys or any(not k.startswith("gate_net.") for k in incompatible.missing_keys):
        raise RuntimeError(
            f"gated actor/base actor state mismatch: {incompatible}"
        )

    agent.actor = actor
    # FactorizedSACAgent.update() steps `pi_opt`; replacing any other attribute
    # would leave the new gated actor completely unoptimized.
    agent.pi_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    return agent


@torch.no_grad()
def gate_stats(agent, z_np):
    """Mean/min/max gate over a batch of capability latents, for logging."""
    if not hasattr(agent.actor, "gate"):
        return {}
    z = torch.as_tensor(np.asarray(z_np), dtype=torch.float32,
                        device=agent.device)
    if z.ndim == 1:
        z = z[None]
    g = agent.actor.gate(z).squeeze(-1)
    return {"gate/mean": float(g.mean()),
            "gate/min": float(g.min()),
            "gate/max": float(g.max())}


if __name__ == "__main__":
    import tempfile, os

    doc = {
        "schema": "jfcrl-curriculum-v2",
        "heldout_joint": "2",
        "train_joints": ["0", "4", "5", "6"],
        "tasks": ["libero_spatial:0", "libero_goal:6"],
        "task_probs": [0.5, 0.5],
        "per_task": {
            "libero_spatial:0": {"joint_pool": ["0", "healthy"],
                                 "fault_probs": [0.9, 0.1]},
            "libero_goal:6": {"joint_pool": ["0", "4", "6", "healthy"],
                              "fault_probs": [0.45, 0.11, 0.34, 0.10]},
        },
    }
    p = os.path.join(tempfile.mkdtemp(), "c.json")
    json.dump(doc, open(p, "w"))

    tasks, tp, pools, ho = load_curriculum(p)
    assert tasks and ho == "2" and len(pools) == 2

    sid = {}
    for t in tasks:
        for f in pools[t]["joint_pool"]:
            sid[(t, _canonical_fault_name(f))] = len(sid)
    w = generation_stratum_weights(sid, tasks, tp, pools)
    assert abs(w.sum() - 1.0) < 1e-9
    assert abs(w[sid[("libero_spatial:0", "j0")]] - 0.45) < 1e-9
    assert abs(w[sid[("libero_goal:6", "healthy")]] - 0.05) < 1e-9

    # equal-share would give every stratum 1/6 = 0.167; healthy in
    # spatial:0 is configured at 0.05, so the shipped sampler oversamples
    # it by 3.3x. That factor grows with the task count.
    print("generation weights:", np.round(w, 4))
    print("equal-share weight:", round(1 / len(sid), 4))
    print("all checks passed")
