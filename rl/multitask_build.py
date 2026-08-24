"""Utilities for JFCRL training across arbitrary LIBERO task sets.

Two VLA construction modes are supported:
  1) legacy/default: one official suite-specific frozen OpenVLA-OFT per task;
  2) shared: load ONE frozen multi-suite OpenVLA-OFT stack once and reuse the
     same model / processor / OFT heads for every task while keeping a separate
     task description, action queue, MuJoCo env, and suite-specific unnorm key.

The shared mode is intended for the 5-task -> 10-task experiment. It changes
only frozen-VLA infrastructure; the JFCRL capability encoder, Transformer,
FiLM-SAC actor/critics, replay, losses, and gradient routing are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from rl.build import TASK_MAX_STEPS, VLACfg, build_env, build_vla, split_init_states

COMBINED_CHECKPOINT = (
    "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
)


SUITE_CHECKPOINTS = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10": "moojink/openvla-7b-oft-finetuned-libero-10",
}

SUITE_UNNORM = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_id: int

    @property
    def key(self) -> str:
        return f"{self.suite}:{self.task_id}"

    def to_dict(self):
        return {"suite": self.suite, "task_id": int(self.task_id), "key": self.key}


def parse_task_spec(token: str) -> TaskSpec:
    """Parse e.g. 'libero_spatial:0' or 'libero_goal:6'."""
    try:
        suite, task_id = token.rsplit(":", 1)
        task_id = int(task_id)
    except Exception as exc:
        raise ValueError(
            f"invalid task spec {token!r}; expected e.g. libero_spatial:0"
        ) from exc
    if suite not in SUITE_CHECKPOINTS:
        raise ValueError(
            f"unsupported suite {suite!r}; choose from {sorted(SUITE_CHECKPOINTS)}"
        )
    return TaskSpec(suite=suite, task_id=task_id)


def build_shared_vla_handles(
    checkpoint: str,
    *,
    seed: int = 7,
    representative_suite: str = "libero_spatial",
):
    """Load one frozen OpenVLA-OFT stack for reuse across all task bundles.

    `build_vla` does not need a task environment or language instruction. The
    per-task `VLACfg` and `FrozenOFT` wrappers below still provide the correct
    suite-specific `unnorm_key` and task description at action-query time.
    The returned modules are frozen/eval exactly as in the legacy build path.
    """
    if representative_suite not in SUITE_UNNORM:
        raise ValueError(f"unsupported representative suite {representative_suite!r}")
    cfg = VLACfg(task_suite_name=representative_suite, task_id=0, seed=seed)
    cfg.pretrained_checkpoint = str(checkpoint)
    cfg.unnorm_key = SUITE_UNNORM[representative_suite]
    handles = build_vla(cfg)
    handles = dict(handles)
    handles["_pretrained_checkpoint"] = str(checkpoint)
    return handles


@dataclass
class TaskBundle:
    spec: TaskSpec
    cfg: VLACfg
    renv: object
    train_ids: list
    eval_ids: list
    meta: dict


def build_task_bundle(
    spec: TaskSpec,
    *,
    residual_cfg_factory,
    joint_pool,
    fault_probs,
    fault_block: int,
    n_eval_states: int,
    seed: int,
    collect_images: bool = False,
    pretrained_checkpoint: str | None = None,
    shared_vla_handles: dict | None = None,
):
    """Build one task bundle.

    If `shared_vla_handles` is None, this preserves the original behavior and
    loads the official suite-specific VLA. If handles are supplied, the same
    frozen VLA stack is reused and only the per-task wrapper/env is distinct.
    """
    from faults.multi_fault import MultiFaultManager
    from rl.residual_env import FrozenOFT, ResidualLiberoEnv

    cfg = VLACfg(task_suite_name=spec.suite, task_id=spec.task_id, seed=seed)
    cfg.pretrained_checkpoint = (
        str(pretrained_checkpoint) if pretrained_checkpoint is not None
        else SUITE_CHECKPOINTS[spec.suite]
    )
    cfg.unnorm_key = SUITE_UNNORM[spec.suite]

    env, task_description, initial_states = build_env(cfg)
    if shared_vla_handles is None:
        handles = build_vla(cfg)
        vla_shared = False
    else:
        handles = shared_vla_handles
        loaded = handles.get("_pretrained_checkpoint")
        if loaded is not None and str(loaded) != str(cfg.pretrained_checkpoint):
            raise RuntimeError(
                f"shared VLA checkpoint mismatch: loaded={loaded!r}, "
                f"task cfg={cfg.pretrained_checkpoint!r}"
            )
        vla_shared = True
    vla = FrozenOFT(
        cfg_vla=cfg,
        model=handles["model"],
        resize_size=handles["resize_size"],
        task_description=task_description,
        processor=handles["processor"],
        action_head=handles["action_head"],
        proprio_projector=handles["proprio_projector"],
        noisy_action_projector=handles["noisy_action_projector"],
        use_film=cfg.use_film,
        chunk_len=cfg.num_open_loop_steps,
    )

    fault_mgr = MultiFaultManager(
        joint_pool=tuple(joint_pool),
        fault_probs=tuple(fault_probs),
        include_healthy=False,
        fault_block=fault_block,
        seed=seed,
        env_factory=lambda cfg=cfg: build_env(cfg)[0],
    )
    train_ids, eval_ids = split_init_states(len(initial_states), n_eval_states)
    res_cfg = residual_cfg_factory(TASK_MAX_STEPS[spec.suite], cfg)

    renv = ResidualLiberoEnv(
        env=env,
        vla=vla,
        fault_mgr=fault_mgr,
        initial_states=initial_states,
        init_ids=train_ids,
        cfg=res_cfg,
        collect_images=collect_images,
        context_include_time=False,
    )
    meta = {
        "task_key": spec.key,
        "suite": spec.suite,
        "task_id": int(spec.task_id),
        "task_description": task_description,
        "pretrained_checkpoint": cfg.pretrained_checkpoint,
        "vla_shared_across_tasks": bool(vla_shared),
        "unnorm_key": cfg.unnorm_key,
        "n_init_states": len(initial_states),
        "obs_dim": renv.obs_dim,
        "act_dim": renv.act_dim,
        "chunk_len": cfg.num_open_loop_steps,
        "max_steps": res_cfg.max_steps,
        "fault_pool": fault_mgr.names,
        "joint_ctx_dim": renv.joint_ctx_dim,
        "joint_token_dim": renv.joint_token_dim,
        "jacobian_dim": renv.jacobian_dim,
        "jacobian_in_obs": bool(getattr(res_cfg, "include_jacobian_obs", False)),
    }
    return TaskBundle(spec, cfg, renv, train_ids, eval_ids, meta)


def assert_compatible_bundles(bundles):
    """All tasks must expose the same Panda/JFCRL tensor contract."""
    if not bundles:
        raise ValueError("at least one task is required")
    ref = bundles[0].meta
    keys = ("obs_dim", "act_dim", "joint_ctx_dim", "joint_token_dim", "jacobian_dim")
    for b in bundles[1:]:
        bad = {k: (ref[k], b.meta[k]) for k in keys if ref[k] != b.meta[k]}
        if bad:
            raise RuntimeError(
                f"task {b.spec.key} is incompatible with {bundles[0].spec.key}: {bad}"
            )
    if not all(b.meta.get("jacobian_in_obs", False) for b in bundles):
        raise RuntimeError("every multitask environment must include the live Jacobian")
