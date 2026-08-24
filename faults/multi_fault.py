"""
multi_fault.py -- episode-level fault sampling over a pool of joint locks
plus an optional healthy condition.

==========================================================================
FIX 1 -- THE CONSTRAINT CHECKER MUST NEVER FAIL SILENTLY
==========================================================================
The earlier version called `sim.model.equality_id2name(i)` inside a bare
`except: pass`. That method does not exist on robosuite's model wrapper, so
the exception fired on every constraint and the function returned `[]` --
indistinguishable from "no fault compiled". That is what produced

    expected exactly [fault_lock_robot0_joint1], found []

during screening, and it sent the debugging in exactly the wrong direction:
the fault WAS compiled, the checker just could not see it.

This version uses the native MuJoCo lookup,
`mujoco.mj_id2name(model, mjOBJ_EQUALITY, i)`, on the raw MjModel struct,
and RAISES if it cannot reach that struct. A verification routine that
cannot verify must say so; it must never return an empty list that reads as
a clean answer. It also checks `eq_active`, because a constraint that is
compiled but disabled is not a fault.

==========================================================================
FIX 2 -- SWITCHING FAULTS NEEDS A FRESH ENV
==========================================================================
robosuite applies `_xml_processors` only inside `_initialize_sim()`, and
`reset()` calls that only when
`(sim is None) or (hard_reset and not deterministic_reset)`. On a REUSED env
that condition can be false, the processor never runs, and the episode
executes HEALTHY while being logged as faulted -- the worst failure mode
available, because the numbers look plausible.

Additionally `set_xml_processor` APPENDS to a list rather than replacing it,
so reusing an env grows that list every episode and every rebuild re-runs
every processor ever registered.

`env_factory` closes the old env and builds a fresh one on a fault CHANGE,
registering the processor before its first reset. That is precisely what the
validated single-fault evaluator did, which is why that path always worked.
No rebuild happens when the fault is unchanged.

==========================================================================
FAULT IDENTITY NEVER REACHES THE ACTOR
==========================================================================
`active` exists for logging, per-fault metrics, and stratified replay. It is
never read by the observation builder. An offline test asserts the
observation vector is bit-identical under j0, j6 and healthy given identical
dynamics.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from faults.joint_lock import LockMonitor, panda_joint_name

FAULT_PREFIX = "fault_lock_"
HEALTHY = "healthy"


# ==========================================================================
# XML processing
# ==========================================================================

def make_multi_fault_processor(locks: dict, damping: dict | None = None,
                               baseline_damping: dict | None = None):
    """locks: {joint_name: lock_value}. Empty dict (and no damping) == healthy.

    Always strips every pre-existing `fault_lock_*` equality joint and
    restores baseline damping before applying the requested fault, so
    switching between conditions cannot leave a residue.
    """
    damping = damping or {}
    baseline_damping = baseline_damping or {}

    def processor(xml_string: str) -> str:
        root = ET.fromstring(xml_string)

        body = root.find("worldbody")
        if body is not None:
            for jname, base in baseline_damping.items():
                el = body.find(f".//joint[@name='{jname}']")
                if el is None:
                    continue
                if base is None:
                    el.attrib.pop("damping", None)
                else:
                    el.set("damping", str(base))
            for jname, value in damping.items():
                el = body.find(f".//joint[@name='{jname}']")
                if el is None:
                    raise KeyError(
                        f"joint '{jname}' not in worldbody; cannot damp it")
                el.set("damping", f"{float(value):.6f}")

        equality = root.find("equality")
        if equality is None:
            if not locks:
                return ET.tostring(root, encoding="unicode")
            equality = ET.SubElement(root, "equality")

        # strip -- the line that makes switching safe
        for child in list(equality.findall("joint")):
            if (child.get("name") or "").startswith(FAULT_PREFIX):
                equality.remove(child)

        for joint_name, lock_value in locks.items():
            el = ET.SubElement(equality, "joint")
            el.set("name", f"{FAULT_PREFIX}{joint_name}")
            el.set("joint1", joint_name)
            # polycoef [c0..c4] with only c0 set => q == c0, a hard lock
            el.set("polycoef", f"{float(lock_value):.10f} 0 0 0 0")
            el.set("active", "true")
            el.set("solref", "0.0002 1")
            el.set("solimp", "0.9999 0.9999 0.0001 0.5 2")

        return ET.tostring(root, encoding="unicode")

    processor.locks = dict(locks)
    processor.damping = dict(damping)
    return processor


# ==========================================================================
# FIX 1: native compiled-model inspection
# ==========================================================================

class ModelInspectionError(RuntimeError):
    """Raised when the compiled model cannot be inspected.

    Deliberately an error and not an empty result. The previous
    implementation swallowed this case and returned `[]`, which is
    identical to "no fault is active" -- so a broken checker looked exactly
    like a broken fault.
    """


def _raw_mj_model(env):
    """Reach the underlying `mujoco.MjModel` behind robosuite's wrapper."""
    sim = getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        raise ModelInspectionError(
            "env.env.sim is missing; cannot inspect the compiled model")
    model = getattr(sim, "model", None)
    if model is None:
        raise ModelInspectionError("env.env.sim.model is missing")

    # robosuite wraps MjModel; the raw struct is usually at `_model`.
    for attr in ("_model", "model"):
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "neq"):
            return inner
    if hasattr(model, "neq"):
        return model
    raise ModelInspectionError(
        f"could not find a MuJoCo model with `neq` on {type(model).__name__}; "
        f"tried ._model and .model"
    )


def read_compiled_damping(env, joint_name: str) -> float:
    """Damping of `joint_name` in the COMPILED model, not in the XML we sent.

    Same principle as `count_active_fault_locks`: verify what MuJoCo built.
    Without this, a damping processor that silently failed to find the joint
    would produce HEALTHY episodes labelled `j0_damp50`, and no drift check
    would catch it -- a damped joint has no lock target to drift from.
    """
    sim = getattr(getattr(env, "env", None), "sim", None)
    if sim is None:
        raise ModelInspectionError("env.env.sim missing")
    model = sim.model
    try:
        jid = model.joint_name2id(joint_name)
        dof = int(model.jnt_dofadr[jid])
        return float(model.dof_damping[dof])
    except Exception as exc:
        raise ModelInspectionError(
            f"cannot read dof_damping for '{joint_name}': {exc!r}") from exc


def count_active_fault_locks(env) -> list:
    """Names of ACTIVE `fault_lock_*` equality constraints, read from the
    compiled model.

    Uses `mujoco.mj_id2name(model, mjOBJ_EQUALITY, i)` -- the native call --
    rather than a robosuite convenience method that may not exist. Raises
    `ModelInspectionError` instead of returning an empty list if anything
    prevents inspection.
    """
    # Offline-test hook: a fake model may expose names directly. Real
    # MjModel never has this attribute, so this cannot mask a live failure.
    direct = getattr(getattr(getattr(env, "env", None), "sim", None), "model", None)
    direct_names = getattr(direct, "_fault_names", None)
    if direct_names is not None:
        return [n for n in direct_names if str(n).startswith(FAULT_PREFIX)]

    try:
        import mujoco
    except ImportError as e:  # pragma: no cover
        raise ModelInspectionError(f"mujoco not importable: {e!r}") from e

    model = _raw_mj_model(env)
    n_eq = int(getattr(model, "neq", 0))
    if n_eq == 0:
        return []

    # eq_active is the runtime enable flag; eq_active0 is its compiled
    # default. A compiled-but-disabled constraint is not a fault.
    active_flags = None
    for attr in ("eq_active", "eq_active0"):
        arr = getattr(model, attr, None)
        if arr is not None and len(arr) >= n_eq:
            active_flags = arr
            break
    data = getattr(getattr(getattr(env, "env", None), "sim", None), "data", None)
    for attr in ("_data", "data"):
        inner = getattr(data, attr, None) if data is not None else None
        if inner is not None and getattr(inner, "eq_active", None) is not None:
            if len(inner.eq_active) >= n_eq:
                active_flags = inner.eq_active
            break

    names = []
    for i in range(n_eq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
        if not nm or not str(nm).startswith(FAULT_PREFIX):
            continue
        if active_flags is not None and not bool(active_flags[i]):
            continue
        names.append(str(nm))
    return names


# ==========================================================================

@dataclass
class FaultSpec:
    """One condition in the pool.

    kind="lock"     equality constraint pinning the joint. `offset` shifts
                    the target away from the joint's initial angle, turning
                    a binary fault into a severity continuum.
    kind="damping"  joint still moves but is heavily resisted.
    joint_idx=None  healthy.
    """

    joint_idx: int | None = None
    prefix: str = "robot0_"
    kind: str = "lock"
    offset: float = 0.0
    damping: float = 50.0

    @property
    def is_healthy(self) -> bool:
        return self.joint_idx is None

    @property
    def name(self) -> str:
        if self.is_healthy:
            return HEALTHY
        if self.kind == "damping":
            return f"j{self.joint_idx}_damp{self.damping:g}"
        if abs(self.offset) > 1e-9:
            return f"j{self.joint_idx}_off{self.offset:+g}"
        return f"j{self.joint_idx}"

    @property
    def joint_name(self) -> str | None:
        if self.is_healthy:
            return None
        return panda_joint_name(self.joint_idx, self.prefix)

    @staticmethod
    def parse(token, prefix: str = "robot0_"):
        """'0' | '6' | 'healthy' | '2:off=0.3' | '2:damp=50' -> FaultSpec."""
        if isinstance(token, FaultSpec):
            return token
        token = str(token).strip()
        if token.lower() in ("healthy", "none"):
            return FaultSpec(None, prefix)
        head, _, tail = token.partition(":")
        if head.lower().startswith("j"):
            head = head[1:]
        spec = FaultSpec(int(head), prefix)
        if tail:
            key, _, val = tail.partition("=")
            key = key.strip().lower()
            if key in ("off", "offset"):
                spec.offset = float(val)
            elif key in ("damp", "damping"):
                spec.kind, spec.damping = "damping", float(val)
            else:
                raise ValueError(
                    f"unknown fault modifier '{key}' in '{token}'; "
                    f"use off=<rad> or damp=<value>")
        return spec


def joint_indices(tokens) -> set:
    """Joint indices touched by a list of condition tokens.

    Needed because a held-out JOINT claim is not protected by excluding a
    condition NAME: `j2` and `j2_off+0.2` and `j2_damp50` are three
    different names and all three are joint 2. Comparing names would let a
    j2 variant into training and quietly invalidate the entire
    unseen-joint result.
    """
    out = set()
    for t in tokens or ():
        sp = FaultSpec.parse(t)
        if sp.joint_idx is not None:
            out.add(int(sp.joint_idx))
    return out


@dataclass
class MultiFaultManager:
    """Samples one condition per EPISODE and guarantees it is the only one."""

    joint_pool: tuple = (0, 6)
    fault_probs: tuple | None = None
    include_healthy: bool = False
    prefix: str = "robot0_"
    drift_tol: float = 1e-2
    seed: int = 0
    fault_block: int = 1
    env_factory: object = None
    """Zero-arg callable returning a brand-new LIBERO env.

    REQUIRED whenever the fault can change during a run -- see FIX 2 in the
    module docstring. Without it, a fault switch on a reused env may compile
    nothing at all."""

    specs: list = field(init=False)
    probs: np.ndarray = field(init=False)
    active: FaultSpec | None = field(default=None, init=False)
    monitor: LockMonitor | None = field(default=None, init=False)
    env: object = field(default=None, init=False)

    _rng: np.random.Generator = field(init=False)
    _addr: dict = field(default_factory=dict, init=False)
    _targets: dict = field(default_factory=dict, init=False)
    _baseline_damping: dict = field(default_factory=dict, init=False)
    _applied: tuple | None = field(default=None, init=False)
    _rebuilds: int = field(default=0, init=False)
    _env_rebuilds: int = field(default=0, init=False)
    _episodes: int = field(default=0, init=False)
    _counts: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.specs = [FaultSpec.parse(j, self.prefix) for j in self.joint_pool]
        if self.include_healthy:
            self.specs.append(FaultSpec(None, self.prefix))

        if self.fault_probs is None:
            p = np.ones(len(self.specs))
        else:
            p = np.asarray(self.fault_probs, dtype=np.float64)
            if len(p) != len(self.specs):
                raise ValueError(
                    f"fault_probs has {len(p)} entries but the pool has "
                    f"{len(self.specs)} conditions "
                    f"({[s.name for s in self.specs]}). With "
                    f"include_healthy, healthy needs its own probability.")
        self.probs = p / p.sum()
        self._rng = np.random.default_rng(self.seed)
        self._counts = {s.name: 0 for s in self.specs}

    # ------------------------------------------------------------------ #

    @property
    def names(self) -> list:
        return [s.name for s in self.specs]

    def index_of(self, name: str) -> int:
        """Stable integer id per condition, for replay tagging."""
        return self.names.index(name)

    def _qpos_addr(self, env, joint_name: str) -> int:
        if joint_name not in self._addr:
            sim = env.env.sim
            jid = sim.model.joint_name2id(joint_name)
            self._addr[joint_name] = int(sim.model.jnt_qposadr[jid])
        return self._addr[joint_name]

    def read_q(self, env, joint_name: str) -> float:
        return float(env.env.sim.data.qpos[self._qpos_addr(env, joint_name)])

    def sample(self) -> FaultSpec:
        return self.specs[int(self._rng.choice(len(self.specs), p=self.probs))]

    # ------------------------------------------------------------------ #

    def _install(self, env, processor, initial_state):
        """Apply `processor` so that it is GUARANTEED to be compiled."""
        if self.env_factory is not None:
            try:
                env.close()
            except Exception:
                pass
            env = self.env_factory()
            self._env_rebuilds += 1
        env.env.set_xml_processor(processor)
        self._addr.clear()          # model recompiled; qpos addresses move
        env.reset()
        obs = env.set_init_state(initial_state)
        self.env = env
        return env, obs

    def _capture_baseline_damping(self, env):
        for spec in self.specs:
            jn = spec.joint_name
            if jn is None or jn in self._baseline_damping:
                continue
            try:
                sim = env.env.sim
                jid = sim.model.joint_name2id(jn)
                dof = int(sim.model.jnt_dofadr[jid])
                self._baseline_damping[jn] = float(sim.model.dof_damping[dof])
            except Exception:
                self._baseline_damping[jn] = None

    def reset_episode(self, env, initial_state, init_id: int | None = None,
                      force: FaultSpec | None = None):
        """Reset with exactly one (or zero) active fault. Returns obs."""
        if force is not None:
            self.active = force
        elif self.active is None or self._episodes % max(1, self.fault_block) == 0:
            self.active = self.sample()
        self._episodes += 1
        self._counts[self.active.name] = self._counts.get(self.active.name, 0) + 1

        self.env = env
        env.reset()
        obs = env.set_init_state(initial_state)
        self._capture_baseline_damping(env)

        if self.active.is_healthy:
            key = (HEALTHY,)
            if self._applied != key:
                env, obs = self._install(env, make_multi_fault_processor(
                    {}, {}, self._baseline_damping), initial_state)
                self._applied = key
                self._rebuilds += 1
            self.monitor = None
            return obs

        jname = self.active.joint_name

        if self.active.kind == "damping":
            key = (jname, "damp", round(self.active.damping, 9))
            if self._applied != key:
                env, obs = self._install(env, make_multi_fault_processor(
                    {}, {jname: self.active.damping},
                    self._baseline_damping), initial_state)
                self._applied = key
                self._rebuilds += 1
            self.monitor = None    # a damped joint still moves; no lock target
            return obs

        # qpos here is the init configuration -- constraints act during
        # stepping, not at set_init_state -- so this reads the true target
        # even while a DIFFERENT joint is currently locked.
        tkey = (self.active.joint_idx, init_id)
        if tkey in self._targets:
            q0 = self._targets[tkey]
        else:
            q0 = self.read_q(env, jname)
            if init_id is not None:
                self._targets[tkey] = q0
        lock_value = q0 + self.active.offset

        key = (jname, "lock", round(lock_value, 12))
        if self._applied != key:
            env, obs = self._install(env, make_multi_fault_processor(
                {jname: lock_value}, {}, self._baseline_damping),
                initial_state)
            self._applied = key
            self._rebuilds += 1

        self.monitor = LockMonitor(
            joint_name=jname, lock_value=lock_value, drift_tol=self.drift_tol)
        return obs

    def step_record(self, env) -> float:
        if self.monitor is None:
            return 0.0
        return self.monitor.record(self.read_q(env, self.monitor.joint_name))

    # ------------------------------------------------------------------ #

    def assert_exactly_one_lock(self, env):
        """Verify against the COMPILED model, naming the real failure.

        Three distinct outcomes, three distinct messages. Conflating them is
        what cost a debugging cycle last time.
        """
        found = count_active_fault_locks(env)   # raises if uninspectable

        if self.active.is_healthy or self.active.kind == "damping":
            if found:
                raise AssertionError(
                    f"condition '{self.active.name}' expects no equality lock "
                    f"but the compiled model has {found}. A previous fault "
                    f"was not removed.")
            if self.active.kind == "damping":
                # "No lock present" is NOT proof the damping fault exists.
                # Check the compiled dof_damping equals what was requested,
                # otherwise a broken processor yields healthy episodes
                # labelled as damped -- and nothing else would notice,
                # because a damped joint has no lock target to drift from.
                got = read_compiled_damping(env, self.active.joint_name)
                want = float(self.active.damping)
                if abs(got - want) > max(1e-6, 1e-3 * want):
                    raise AssertionError(
                        f"condition '{self.active.name}' expects "
                        f"dof_damping={want:g} on {self.active.joint_name} "
                        f"but the COMPILED model has {got:g}. The damping "
                        f"processor did not take effect: these episodes are "
                        f"effectively HEALTHY while logged as damped.")
            return

        want = f"{FAULT_PREFIX}{self.active.joint_name}"
        if found == [want]:
            return

        if not found:
            raise AssertionError(
                f"NO fault constraint compiled: expected [{want}], found []. "
                f"The XML processor did not run -- this is NOT a stale lock. "
                f"robosuite applies processors only inside _initialize_sim(), "
                f"which reset() skips on a reused env. Every episode here "
                f"would be silently HEALTHY while logged as "
                f"'{self.active.name}'. Pass an `env_factory`.")

        raise AssertionError(
            f"expected exactly [{want}], found {found}. A lock from a "
            f"previous fault survived: this episode is a MULTI-joint fault "
            f"mislabelled as '{self.active.name}'.")

    def stats(self) -> dict:
        return {
            "fault/pool": self.names,
            "fault/probs": [round(float(p), 4) for p in self.probs],
            "fault/n_rebuilds": self._rebuilds,
            "fault/n_env_rebuilds": self._env_rebuilds,
            "fault/env_factory": self.env_factory is not None,
            "fault/n_episodes": self._episodes,
            "fault/episodes_per_condition": dict(self._counts),
            "fault/rebuilds_per_episode": round(
                self._rebuilds / max(1, self._episodes), 3),
            "fault/active": self.active.name if self.active else None,
        }
