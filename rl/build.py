"""
build.py -- one place where LIBERO + OpenVLA-OFT get constructed.

Training and evaluation MUST build the environment identically. If they
drift, "held-out success went up" becomes unfalsifiable: you cannot tell a
learned improvement from a setup difference. So both import from here, and
nothing else is allowed to call `get_model` or `get_libero_env`.

Everything below mirrors upstream `run_libero_eval.py`. The only additions
are the fault manager and the train/eval split of initial states.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Union

import numpy as np

# Upstream default for libero_spatial. Kept here so the value has one home.
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class VLACfg:
    """Mirrors the fields of upstream GenerateConfig that `get_action` reads.

    Values are the validated ones from the handoff (section 8). Changing any
    of them invalidates the 19/20 healthy and 4/20 faulted baselines.
    """

    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, None] = (
        "moojink/openvla-7b-oft-finetuned-libero-spatial"
    )
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: str = "libero_spatial_no_noops"

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_steps_wait: int = 10
    env_img_res: int = 256
    seed: int = 7


def set_headless_env():
    """Handoff section 7. Must run before MuJoCo/OpenGL import."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_vla(cfg: VLACfg):
    """Load the frozen OpenVLA-OFT stack. Returns a dict of handles."""
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_noisy_action_projector,
        get_processor,
        get_proprio_projector,
    )
    from experiments.robot.robot_utils import get_image_resize_size, get_model

    model = get_model(cfg)
    resize_size = get_image_resize_size(cfg)

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Belt and braces: the whole claim rests on this staying frozen.
    for m in (model, action_head, proprio_projector, noisy_action_projector):
        if m is None:
            continue
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    return {
        "model": model,
        "resize_size": resize_size,
        "processor": processor,
        "action_head": action_head,
        "proprio_projector": proprio_projector,
        "noisy_action_projector": noisy_action_projector,
    }


def build_env(cfg: VLACfg):
    """Returns (env, task_description, initial_states)."""
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import get_libero_env

    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = suite.get_task(cfg.task_id)
    initial_states = suite.get_task_init_states(cfg.task_id)
    env, task_description = get_libero_env(
        task, cfg.model_family, resolution=cfg.env_img_res
    )
    return env, task_description, initial_states


def split_init_states(n_total: int, n_eval: int = 10, seed: int = 0):
    """Train / held-out split over LIBERO initial states.

    Held-out states are taken from the TAIL, not sampled at random, so the
    split is reproducible across machines without carrying a seed file --
    and so the first 20 (the indices your paired screening used) stay in
    the training pool where they belong.
    """
    if n_eval >= n_total:
        raise ValueError(f"n_eval={n_eval} must be < n_total={n_total}")
    train_ids = list(range(0, n_total - n_eval))
    eval_ids = list(range(n_total - n_eval, n_total))
    return train_ids, eval_ids


def build_all(cfg: VLACfg, res_cfg, joint_idx: int = 0, fault_enabled: bool = True,
              n_eval_states: int = 10, collect_images: bool = False):
    """Full construction. Returns (renv, train_ids, eval_ids, meta)."""
    from faults.joint_lock import JointLockManager
    from rl.residual_env import FrozenOFT, ResidualLiberoEnv

    env, task_description, initial_states = build_env(cfg)
    handles = build_vla(cfg)

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

    fault_mgr = JointLockManager(joint_idx=joint_idx, enabled=fault_enabled)

    train_ids, eval_ids = split_init_states(len(initial_states), n_eval_states)

    renv = ResidualLiberoEnv(
        env=env,
        vla=vla,
        fault_mgr=fault_mgr,
        initial_states=initial_states,
        init_ids=train_ids,
        cfg=res_cfg,
        collect_images=collect_images,
    )

    meta = {
        "task_description": task_description,
        "n_init_states": len(initial_states),
        "obs_dim": renv.obs_dim,
        "act_dim": renv.act_dim,
        "chunk_len": cfg.num_open_loop_steps,
        "max_steps": res_cfg.max_steps,
    }
    return renv, train_ids, eval_ids, meta


def build_shared(cfg: VLACfg, res_cfg, joint_pool=(0, 6), fault_probs=None,
                 include_healthy: bool = False, fault_block: int = 1,
                 n_eval_states: int = 10, seed: int = 7,
                 collect_images: bool = False,
                 context_include_time: bool = False):
    """Shared-policy construction: one env, one VLA, a POOL of faults.

    Deliberately built through the same `build_env` / `build_vla` path as the
    single-fault case. If the shared runs and the per-joint runs constructed
    the environment differently, "one policy handles both faults" would be
    confounded with a setup difference and there would be no way to tell.
    """
    from faults.multi_fault import MultiFaultManager
    from rl.residual_env import FrozenOFT, ResidualLiberoEnv

    env, task_description, initial_states = build_env(cfg)
    handles = build_vla(cfg)

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

    # FIX 2: a FACTORY, not the single env. Switching faults on a REUSED
    # robosuite env does not reliably recompile the model -- processors run
    # only inside _initialize_sim(), which reset() can skip -- so a fault
    # change must build a fresh env. Costs a few seconds per switch against
    # ~20 s per episode, and `set_xml_processor` appends rather than
    # replaces, so reuse also leaks processors.
    fault_mgr = MultiFaultManager(
        joint_pool=tuple(joint_pool),
        fault_probs=fault_probs,
        include_healthy=include_healthy,
        fault_block=fault_block,
        seed=seed,
        env_factory=lambda: build_env(cfg)[0],
    )

    train_ids, eval_ids = split_init_states(len(initial_states), n_eval_states)

    renv = ResidualLiberoEnv(
        env=env, vla=vla, fault_mgr=fault_mgr,
        initial_states=initial_states, init_ids=train_ids,
        cfg=res_cfg, collect_images=collect_images,
        context_include_time=context_include_time,
    )

    meta = {
        "task_description": task_description,
        "n_init_states": len(initial_states),
        "obs_dim": renv.obs_dim,
        "act_dim": renv.act_dim,
        "chunk_len": cfg.num_open_loop_steps,
        "max_steps": res_cfg.max_steps,
        "fault_pool": fault_mgr.names,
        "ctx_dim": renv.ctx_dim,
        "dyn_dim": renv.dyn_dim,
        "joint_ctx_dim": getattr(renv, "joint_ctx_dim", 0),
        "joint_token_dim": getattr(renv, "joint_token_dim", 0),
        "jacobian_dim": getattr(renv, "jacobian_dim", 0),
        "jacobian_in_obs": bool(getattr(res_cfg, "include_jacobian_obs", False)),
    }
    return renv, train_ids, eval_ids, meta
