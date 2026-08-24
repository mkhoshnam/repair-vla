"""
residual_env.py -- the core of the method.

    a_base  = OpenVLA-OFT(images, language)      [7-D, FROZEN, chunked x8]
    a_final = a_base;  a_final[:6] += scale * tanh(delta_RL)
    a_final = clip(a_final, -1, 1)               [gripper dim 6 untouched]

--------------------------------------------------------------------------
DESIGN DECISIONS AND WHY
--------------------------------------------------------------------------

* THE CONTROL LOOP IS A LINE-BY-LINE MIRROR OF
  `openvla-oft/experiments/robot/libero/run_libero_eval.py::run_episode`.
  Same `num_steps_wait` dummy prefix, same `deque` chunk queue requeried
  when empty, same `process_action` gripper handling, same `success =
  bool(done)` convention. That is deliberate: with `residual_scale = 0`
  this file must reproduce your validated 4/20 faulted baseline exactly.
  If it does not, the wrapper is wrong and every number after it is
  meaningless. `eval_residual.py --policy zero` is that check.

* THE RESIDUAL NEVER SEES IMAGES. Pixels go only to the VLA. The residual's
  view of "what the scene wants" is a_base itself plus proprioception. This
  keeps the actor a small MLP, keeps the replay buffer in RAM, and keeps the
  per-step cost near zero next to a 7B forward pass.

* RESIDUAL ON ARM DIMS ONLY (0..5). The gripper is binarized to exactly +-1
  by `process_action`; a continuous residual on it would either be a no-op
  or would flip grasps at random. Handoff section 12.1.

* A SCALAR `residual_scale` IS CORRECT HERE. All six arm dims are in the
  same normalized [-1, 1] OSC action space, so one scale is dimensionally
  coherent. (This is the opposite of the G1/WBC case, where the action
  mixed rad, m/s and m and a scalar was incoherent. Different robot,
  different answer.)

* THE FAULT LABEL IS NOT IN THE OBSERVATION. What the residual gets is a
  history of (commanded arm action, realized end-effector delta). A locked
  j0 shows up as a persistent, direction-specific gap between what was
  commanded and what the body did. That is closed-loop execution mismatch,
  observable from onboard sensing alone -- no privileged simulator state,
  and no fault flag. It is also what makes the later "unseen fault"
  experiment possible without changing the observation space, so it is in
  from day one even though j0 is always locked in the first run.

* NO REFERENCE TRAJECTORY. Reward is the LIBERO task outcome. There is no
  term rewarding similarity to a healthy rollout, and none may be added
  later "just to help it train" -- that single line collapses the claim
  that separates this from J-PARC, which builds its teacher targets from
  reference end-effector positions. `w_residual` defaults to 0.0 so the
  headline condition really is task-reward-only (handoff section 12.3).

--------------------------------------------------------------------------
FIVE FACTS TO CONFIRM ON DENVER BEFORE TRAINING (run probe_interfaces.py)
--------------------------------------------------------------------------
1. `robot0_joint_pos` / `robot0_joint_vel` are present in the LIBERO obs.
2. Arm action dims land in [-1, 1] (they should: OSC_POSE normalized).
3. `env.step` returns `done=True` only on success, not on horizon.
4. Wall-clock: env-steps/s with the VLA in the loop.
5. `NUM_ACTIONS_CHUNK` == 8 in your `prismatic.vla.constants`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# quaternion helpers (kept local so the offline tests need no robosuite)
# --------------------------------------------------------------------------


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """q = [x, y, z, w] (robosuite convention)."""
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x0, y0, z0, w0 = q0
    return np.array(
        [
            w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
            w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
            w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
            w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
        ]
    )


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """[x, y, z, w] -> 3-vector whose norm is the angle. Matches robosuite."""
    q = np.asarray(q, dtype=np.float64)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-8:
        return np.zeros(3)
    return (q[:3] / den) * (2.0 * np.arccos(w))


def rot_delta_axisangle(q_prev: np.ndarray, q_now: np.ndarray) -> np.ndarray:
    """Realized rotation from q_prev to q_now, as an axis-angle 3-vector.

    Differencing axis-angle vectors directly is wrong near the +-pi wrap;
    composing with the conjugate is not.
    """
    dq = quat_multiply(q_now, quat_conjugate(q_prev))
    if dq[3] < 0:  # canonicalize to the short way round
        dq = -dq
    return quat2axisangle(dq)


# --------------------------------------------------------------------------


@dataclass
class ResidualCfg:
    # --- residual authority ------------------------------------------------
    residual_scale: float = 0.2      # in normalized OSC units; 1.0 would let
                                     # RL fully overwrite the VLA
    arm_dims: int = 6                # dims 0..5; dim 6 (gripper) untouched

    # --- observation -------------------------------------------------------
    history_len: int = 8             # == chunk length, so the window always
                                     # spans at least one full open-loop chunk
    # New joint-factorized method only. Default False keeps every validated
    # legacy observation bit-identical. When True, append the current 6x7
    # Panda geometric Jacobian (column-major by joint) to the low-dimensional
    # observation. This is kinematic state, not a fault label.
    include_jacobian_obs: bool = False

    # --- reward ------------------------------------------------------------
    # Handoff 12.3: the primary condition is TASK REWARD ONLY. Both knobs
    # below default off. Turning either on makes the run an ablation that
    # must be reported separately -- it is no longer the headline claim.
    w_residual: float = 0.0          # penalty on ||delta||^2
    w_shaping: float = 0.0           # potential-based, object-geometry only;
                                     # requires privileged state, so this is
                                     # a DIAGNOSTIC, not the main method
    gamma: float = 0.99              # must match the trainer for shaping to
                                     # be policy-invariant (Ng et al. 1999)

    # --- episode -----------------------------------------------------------
    max_steps: int = 220             # TASK_MAX_STEPS[libero_spatial] upstream
    num_steps_wait: int = 10         # upstream default: let objects settle

    seed: int = 0


class FrozenOFT:
    """Chunk-queue wrapper around the frozen OpenVLA-OFT policy.

    Mirrors the upstream requery-when-empty pattern exactly. The queue is the
    only place chunk state lives; nothing else may cache or index into it.
    """

    def __init__(self, cfg_vla, model, resize_size, task_description,
                 processor=None, action_head=None, proprio_projector=None,
                 noisy_action_projector=None, use_film=False,
                 chunk_len: int = 8):
        self.cfg = cfg_vla
        self.model = model
        self.resize_size = resize_size
        self.task_description = task_description
        self.processor = processor
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.noisy_action_projector = noisy_action_projector
        self.use_film = use_film
        self.chunk_len = chunk_len

        self._queue: deque = deque(maxlen=chunk_len)
        self._consumed = 0
        self.n_queries = 0

        # freeze, defensively -- the scientific claim depends on it
        for p in getattr(model, "parameters", lambda: [])():
            p.requires_grad_(False)
        if hasattr(model, "eval"):
            model.eval()

    def reset(self):
        self._queue.clear()
        self._consumed = 0

    @property
    def chunk_phase(self) -> float:
        """Age of the action most recently returned, in [0, 1).

        0.0 = first action of a fresh chunk, 7/8 = last. Read AFTER
        `next_action`, which is where `_build_obs` reads it.

        This is deliberately an explicit counter rather than
        `1 - len(queue)/chunk_len`. That expression looks equivalent but
        aliases: after the 8th action is popped the queue is empty and the
        expression returns 0.0 -- the same value it gives for the FIRST
        action of a chunk. The residual would then be unable to distinguish
        "this command is fresh" from "this command is 8 steps stale and the
        fault has had a whole chunk to push us off course", which is exactly
        the distinction the feature exists to provide.

        Why it matters at all: within a chunk the VLA is open-loop. It
        cannot see what the locked joint did to the trajectory since the
        chunk was emitted, so error accumulates with phase and the right
        correction late in a chunk is larger than early in it.
        """
        if self._consumed == 0:
            return 0.0
        return (self._consumed - 1) / float(self.chunk_len)

    def next_action(self, observation) -> np.ndarray:
        """Returns the processed 7-D base action for this env step."""
        if len(self._queue) == 0:
            from experiments.robot.robot_utils import get_action

            actions = get_action(
                self.cfg,
                self.model,
                observation,
                self.task_description,
                processor=self.processor,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
                use_film=self.use_film,
            )
            self._queue.extend(actions)
            self._consumed = 0
            self.n_queries += 1

        from experiments.robot.robot_utils import (
            invert_gripper_action,
            normalize_gripper_action,
        )

        a = self._queue.popleft()
        self._consumed += 1
        a = normalize_gripper_action(np.asarray(a, dtype=np.float64), binarize=True)
        if self.cfg.model_family == "openvla":
            a = invert_gripper_action(a)
        return np.asarray(a, dtype=np.float32)


class ResidualLiberoEnv:
    """Step-level MDP over a faulted LIBERO task with a frozen VLA base."""

    def __init__(self, env, vla: FrozenOFT, fault_mgr, initial_states,
                 init_ids, cfg: ResidualCfg, prepare_observation=None,
                 collect_images: bool = False,
                 context_include_time: bool = False):
        self.env = env
        self.vla = vla
        self.faults = fault_mgr
        self.initial_states = initial_states
        self.init_ids = list(init_ids)
        self.cfg = cfg
        self.collect_images = collect_images
        self.rng = np.random.default_rng(cfg.seed)

        if prepare_observation is None:
            from experiments.robot.libero.run_libero_eval import (
                prepare_observation as _prep,
            )
            prepare_observation = _prep
        self._prepare_observation = prepare_observation

        self.act_dim = cfg.arm_dims
        H = cfg.history_len
        # 8 proprio + 7 qpos + 7 qvel + 7 base action + phase + time + H*12
        self.base_obs_dim = 8 + 7 + 7 + 7 + 1 + 1 + H * 12
        self.jacobian_dim = 6 * 7
        self.obs_dim = self.base_obs_dim + (
            self.jacobian_dim if cfg.include_jacobian_obs else 0
        )

        self._hist: deque = deque(maxlen=H)
        self.replay_images: list = []
        self.episode_init_id: int | None = None

        # ---- CONTEXT FEATURE (spec 3.1) ---------------------------------
        # Per-step token for the capability encoder. Deployment-observable
        # signals ONLY: no object pose, no fault identity, no lock angle.
        #   q(7) qdot(7) a_vla_arm(6) a_final_arm(6) dq(7) d_eef_pos(3)
        #   d_eef_rot(3) chunk_phase(1) [+ time(1) if enabled]
        self.context_include_time = bool(context_include_time)
        self.ctx_dim = 7 + 7 + 6 + 6 + 7 + 3 + 3 + 1 + (
            1 if self.context_include_time else 0)
        # Realized motion of the step just taken: [dq(7), d_pos(3), d_rot(3)].
        # This is the L_dyn target and it is what the encoder must NOT be
        # allowed to see for the current step.
        self.dyn_dim = 13
        self.last_context = np.zeros(self.ctx_dim, dtype=np.float32)
        self.last_dyn_target = np.zeros(self.dyn_dim, dtype=np.float32)

        # Joint-factorized context. Each of the seven Panda joints gets the
        # SAME 28-D token; no joint/fault one-hot is ever included:
        #   q_i, qdot_i, dq_i, J_i(6), a_vla(6), a_final(6),
        #   d_eef(6), chunk_phase(1)
        # The shared temporal encoder is therefore forced to learn a reusable
        # actuator-response rule rather than a separate detector per joint.
        self.n_arm_joints = 7
        self.joint_token_dim = 28
        self.joint_ctx_dim = self.n_arm_joints * self.joint_token_dim
        self.last_joint_context = np.zeros(self.joint_ctx_dim, dtype=np.float32)
        self.last_joint_dyn_target = np.zeros(self.dyn_dim, dtype=np.float32)
        self._last_jacobian = np.zeros((6, 7), dtype=np.float32)
        self.episode_id = -1

    # ------------------------------------------------------------------ #
    # observation construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _proprio(obs) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
                quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
            ]
        )

    def _arm_joint_dof_indices(self):
        """Return MuJoCo dof indices for Panda arm joints 0..6."""
        from faults.joint_lock import panda_joint_name

        model = self.env.env.sim.model
        out = []
        for i in range(7):
            jid = int(model.joint_name2id(panda_joint_name(i)))
            out.append(int(model.jnt_dofadr[jid]))
        return out

    def _eef_site_name_and_id(self):
        """Resolve the robot end-effector site without hard-coding one build.

        robosuite versions expose this through slightly different attributes.
        We try the robot/gripper metadata first, then only common *grip_site*
        names. Failure is loud: this method's scientific premise depends on
        the Jacobian, so silently returning zeros would invalidate the run.
        """
        model = self.env.env.sim.model
        robots = getattr(self.env.env, "robots", None)
        robot = robots[0] if robots else None

        # Some robosuite SingleArm versions expose an integer site id.
        if robot is not None:
            sid = getattr(robot, "eef_site_id", None)
            if sid is not None:
                try:
                    sid = int(sid)
                    # Recover the name where possible; modern and legacy
                    # wrappers differ here.
                    name = None
                    if hasattr(model, "site_id2name"):
                        name = model.site_id2name(sid)
                    return name, sid
                except Exception:
                    pass

            gripper = getattr(robot, "gripper", None)
            for attr in ("important_sites", "visualization_sites", "contact_geoms"):
                d = getattr(gripper, attr, None) if gripper is not None else None
                if isinstance(d, dict):
                    for key in ("grip_site", "grip_site_cylinder", "eef_site"):
                        name = d.get(key)
                        if isinstance(name, str):
                            try:
                                return name, int(model.site_name2id(name))
                            except Exception:
                                pass

        for name in ("robot0_grip_site", "gripper0_grip_site",
                     "robot0_eef_site", "grip_site"):
            try:
                return name, int(model.site_name2id(name))
            except Exception:
                pass

        # Last resort: inspect site names and choose an explicit grip/eef site.
        names = getattr(model, "site_names", None)
        if names is not None:
            for name in names:
                text = name.decode() if isinstance(name, bytes) else str(name)
                low = text.lower()
                if "grip_site" in low or "eef_site" in low:
                    try:
                        return text, int(model.site_name2id(text))
                    except Exception:
                        pass
        raise RuntimeError(
            "joint-factorized method could not resolve the Panda end-effector "
            "site. Refusing to train with a fake/zero Jacobian. Inspect "
            "env.env.robots[0].gripper.important_sites and add the site name."
        )

    def arm_jacobian(self) -> np.ndarray:
        """Current 6x7 geometric Jacobian [linear; angular] for the Panda arm."""
        sim = self.env.env.sim
        model, data = sim.model, sim.data
        site_name, site_id = self._eef_site_name_and_id()
        dofs = self._arm_joint_dof_indices()

        # Legacy mujoco-py / robosuite wrappers.
        if site_name is not None and hasattr(data, "get_site_jacp"):
            jp = np.asarray(data.get_site_jacp(site_name), dtype=np.float64).reshape(3, -1)
            jr = np.asarray(data.get_site_jacr(site_name), dtype=np.float64).reshape(3, -1)
            J = np.vstack([jp[:, dofs], jr[:, dofs]])
        else:
            # Native mujoco Python bindings. Reach raw structs when robosuite
            # wraps them.
            import mujoco
            raw_m = getattr(model, "_model", model)
            raw_d = getattr(data, "_data", data)
            nv = int(raw_m.nv)
            jp = np.zeros((3, nv), dtype=np.float64)
            jr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacSite(raw_m, raw_d, jp, jr, int(site_id))
            J = np.vstack([jp[:, dofs], jr[:, dofs]])

        if J.shape != (6, 7) or not np.isfinite(J).all():
            raise RuntimeError(f"invalid Panda Jacobian shape/values: {J.shape}")
        return J.astype(np.float32)

    def _build_obs(self, env_obs, a_base: np.ndarray) -> np.ndarray:
        """Fault-agnostic by construction.

        Everything below comes from the env observation, the base action, or
        the chunk/time counters. `self.faults.active` is deliberately not
        referenced here and must never be: the headline claim is that the
        policy infers the fault from closed-loop mismatch. An offline test
        asserts this vector is bit-identical under two different active
        faults given identical dynamics.
        """
        H = self.cfg.history_len
        hist = list(self._hist)
        while len(hist) < H:
            hist.insert(0, np.zeros(12, dtype=np.float32))

        parts = [
            self._proprio(env_obs),
            np.asarray(env_obs["robot0_joint_pos"], dtype=np.float32).reshape(7),
            np.asarray(env_obs["robot0_joint_vel"], dtype=np.float32).reshape(7),
            a_base.astype(np.float32),
            np.array([self.vla.chunk_phase], dtype=np.float32),
            np.array([self.t / float(self.cfg.max_steps)], dtype=np.float32),
            np.concatenate(hist).astype(np.float32),
        ]
        if self.cfg.include_jacobian_obs:
            J = self.arm_jacobian()
            self._last_jacobian = J.copy()
            # Flatten as joint-major columns: [J_0(6), ..., J_6(6)].
            parts.append(J.T.reshape(-1).astype(np.float32))
        v = np.concatenate(parts)
        assert v.shape == (self.obs_dim,), (v.shape, self.obs_dim)
        return v

    # ------------------------------------------------------------------ #

    def reset(self, init_id: int | None = None, force_fault=None) -> np.ndarray:
        """force_fault: a FaultSpec, to pin the condition for evaluation.

        Training leaves this None so the manager samples. Evaluation pins it
        so each condition is measured on the same initial states.
        """
        if init_id is None:
            init_id = int(self.rng.choice(self.init_ids))
        self.episode_init_id = int(init_id)

        kw = {"init_id": init_id}
        if force_fault is not None:
            kw["force"] = force_fault
        env_obs = self.faults.reset_episode(
            self.env, self.initial_states[init_id], **kw
        )
        # FIX 2 (caller side): with an env_factory the manager may have CLOSED
        # our env and built a fresh one. Adopt it here, BEFORE the dummy-action
        # loop below -- that loop calls self.env.step(), so a stale handle
        # would step a closed simulator on the very first action.
        new_env = getattr(self.faults, "env", None)
        if new_env is not None and new_env is not self.env:
            self.env = new_env

        # Upstream: dummy actions first so objects settle. These are NOT RL
        # steps -- no transition is stored, no residual is applied.
        from experiments.robot.libero.libero_utils import get_libero_dummy_action

        for _ in range(self.cfg.num_steps_wait):
            env_obs, _, _, _ = self.env.step(
                get_libero_dummy_action(self.vla.cfg.model_family)
            )

        self.vla.reset()
        self._hist.clear()
        self.replay_images = []
        self.t = 0
        self.n_success_steps = 0

        # A NEW episode id every reset. The replay buffer refuses to build a
        # history window across an id change, which is what stops a context
        # from spanning an episode boundary, a fault switch, or an env
        # rebuild -- all three end an episode, so one check covers them.
        self.episode_id += 1
        self.last_context = np.zeros(self.ctx_dim, dtype=np.float32)
        self.last_dyn_target = np.zeros(self.dyn_dim, dtype=np.float32)
        self.last_joint_context = np.zeros(self.joint_ctx_dim, dtype=np.float32)
        self.last_joint_dyn_target = np.zeros(self.dyn_dim, dtype=np.float32)
        self._last_jacobian = (self.arm_jacobian() if self.cfg.include_jacobian_obs
                               else np.zeros((6, 7), dtype=np.float32))
        self._q_prev = np.asarray(env_obs["robot0_joint_pos"],
                                  dtype=np.float64).copy()

        self._env_obs = env_obs
        self._a_base = self._query_base(env_obs)
        return self._build_obs(env_obs, self._a_base)

    def _query_base(self, env_obs) -> np.ndarray:
        observation, img = self._prepare_observation(env_obs, self.vla.resize_size)
        if self.collect_images:
            self.replay_images.append(img)
        return self.vla.next_action(observation)

    def step(self, residual: np.ndarray, gate: float = 1.0):
        """Advance one env step.

        `residual` is (6,) and ALREADY BOUNDED TO [-1, 1] -- SAC's tanh-Gaussian
        actor squashes it. The env only scales. Do not add a second tanh here:
        tanh(tanh(x)) is monotonic but compresses the range, so the effective
        authority would be ~0.76x of the configured `residual_scale` and the
        number you report in the paper would not be the number you ran.
        """
        a_base = self._a_base
        # `gate` in [0, 1] scales the residual's authority this step. It is
        # a deterministic function of the observation (via the capability
        # latent's novelty), so from the MDP's point of view it is part of
        # the environment, exactly like residual_scale -- no importance
        # correction is needed and the stored action stays the raw residual.
        # gate = 1.0 is the ungated policy, bit-identical to before.
        gate = float(np.clip(gate, 0.0, 1.0))
        u = np.asarray(residual, dtype=np.float64).reshape(self.act_dim)
        if np.abs(u).max() > 1.0 + 1e-4:
            raise ValueError(
                f"residual must be bounded to [-1, 1]; got max |u| = "
                f"{np.abs(u).max():.4f}. The actor is responsible for squashing."
            )
        delta = self.cfg.residual_scale * gate * np.clip(u, -1.0, 1.0)

        a_final = np.array(a_base, dtype=np.float64, copy=True)
        a_final[: self.act_dim] = np.clip(
            a_final[: self.act_dim] + delta, -1.0, 1.0
        )
        # dim 6 (gripper) is left exactly as process_action produced it.

        q_before = np.asarray(self._env_obs["robot0_joint_pos"],
                              dtype=np.float64).copy()
        qd_before = np.asarray(self._env_obs["robot0_joint_vel"],
                               dtype=np.float64).copy()
        eef_pos_before = np.asarray(
            self._env_obs["robot0_eef_pos"], dtype=np.float64
        ).copy()
        eef_quat_before = np.asarray(
            self._env_obs["robot0_eef_quat"], dtype=np.float64
        ).copy()
        J_before = (self.arm_jacobian() if self.cfg.include_jacobian_obs
                    else self._last_jacobian.copy())

        env_obs, reward, done, info = self.env.step(a_final.tolist())
        self.t += 1

        drift = self.faults.step_record(self.env)

        # closed-loop execution mismatch: what was commanded vs what moved
        d_pos = (
            np.asarray(env_obs["robot0_eef_pos"], dtype=np.float64) - eef_pos_before
        )
        d_rot = rot_delta_axisangle(
            eef_quat_before, np.asarray(env_obs["robot0_eef_quat"], dtype=np.float64)
        )
        self._hist.append(
            np.concatenate(
                [a_final[: self.act_dim], d_pos, d_rot]
            ).astype(np.float32)
        )

        # ---- context feature + dynamics target for this step -------------
        d_q = (np.asarray(env_obs["robot0_joint_pos"], dtype=np.float64)
               - q_before)
        parts = [
            q_before, qd_before,
            np.asarray(a_base[: self.act_dim], dtype=np.float64),
            a_final[: self.act_dim],
            d_q, d_pos, d_rot,
            np.array([self.vla.chunk_phase]),
        ]
        if self.context_include_time:
            parts.append(np.array([self.t / float(self.cfg.max_steps)]))
        self.last_context = np.concatenate(parts).astype(np.float32)
        self.last_dyn_target = np.concatenate(
            [d_q, d_pos, d_rot]).astype(np.float32)

        if self.cfg.include_jacobian_obs:
            deef = np.concatenate([d_pos, d_rot]).astype(np.float64)
            toks = []
            for i in range(7):
                tok = np.concatenate([
                    np.array([q_before[i], qd_before[i], d_q[i]], dtype=np.float64),
                    J_before[:, i].astype(np.float64),
                    np.asarray(a_base[: self.act_dim], dtype=np.float64),
                    a_final[: self.act_dim],
                    deef,
                    np.array([self.vla.chunk_phase], dtype=np.float64),
                ])
                assert tok.shape == (self.joint_token_dim,), tok.shape
                toks.append(tok)
            self.last_joint_context = np.stack(toks).astype(np.float32).reshape(-1)
            self.last_joint_dyn_target = self.last_dyn_target.copy()

        # ---- reward. Upstream convention: done == success. --------------
        success = bool(done)
        r = 1.0 if success else 0.0
        r_task = r
        if self.cfg.w_residual > 0.0:
            r -= self.cfg.w_residual * float(np.sum(delta**2))

        terminated = success
        truncated = (not success) and (self.t >= self.cfg.max_steps)

        self._env_obs = env_obs
        if terminated:
            # Success. `term=1` masks the target's next-state value entirely,
            # so next_obs is never read -- skip the VLA query and save a 7B
            # forward pass on every successful episode.
            next_obs = self._build_obs(env_obs, self._a_base)
        else:
            # Both the ordinary case AND truncation. Truncation BOOTSTRAPS,
            # so its next_obs is read by the critic and must be a real state
            # -- including the base action that actually belongs to it.
            # Reusing the stale `_a_base` here would feed the critic a state
            # whose action feature came from the previous timestep, biasing
            # the value of every horizon-length episode. Since almost all
            # episodes truncate at this success rate, that bias would land
            # on almost all of the data.
            self._a_base = self._query_base(env_obs)
            next_obs = self._build_obs(env_obs, self._a_base)

        info = {
            "success": success,
            # Logging / per-fault metrics / replay tagging only. Never read
            # by _build_obs -- the residual must infer the fault from
            # command-vs-realization mismatch, not from a label.
            "fault": getattr(getattr(self.faults, "active", None), "name", None),
            "r_task": r_task,
            "residual_norm": float(np.linalg.norm(delta)),
            "gate": gate,
            "base_action_norm": float(np.linalg.norm(a_base[: self.act_dim])),
            "final_action_norm": float(np.linalg.norm(a_final[: self.act_dim])),
            "clipped_frac": float(
                np.mean(np.abs(a_final[: self.act_dim]) >= 0.999)
            ),
            "lock_drift": drift,
            "t": self.t,
            "init_id": self.episode_init_id,
            "env_reward": float(reward),
            "jacobian_fro_norm": float(np.linalg.norm(J_before))
            if self.cfg.include_jacobian_obs else float("nan"),
        }
        return next_obs, r, terminated, truncated, info
