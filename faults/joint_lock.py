"""
joint_lock.py -- reusable MuJoCo joint-lock fault, factored out of the
validated `eval_joint_fault_gap.py` so that training and evaluation use the
*same* mechanism (handoff section 13.2).

The fault is a physics constraint, not an action-space trick. A MuJoCo
<equality><joint> element pins the joint to a constant angle; the solver
fights any torque that tries to move it, exactly as a seized actuator would.

--------------------------------------------------------------------------
THE RESET ORDERING PROBLEM -- read this before changing anything
--------------------------------------------------------------------------
The lock target is "that episode's own initial joint angle". That creates a
chicken-and-egg: you cannot write the constraint into the XML until you know
the angle, and you cannot know the angle until the sim is built and the
initial state is applied. The correct sequence is therefore:

    env.reset()                      # build clean
    obs = env.set_init_state(s)      # arm at the init configuration
    q0  = read_joint_qpos(env, j)    # <-- the lock target
    env.env.set_xml_processor(make_joint_lock_processor(name, q0))
    env.reset()                      # REBUILD, constraint now compiled in
    obs = env.set_init_state(s)      # same init state, joint now locked

Two sim rebuilds per episode. For a 20-episode screening run that is
irrelevant; for a 200k-step RL run it is not, so `JointLockManager` caches
the lock target per init-state id and, if the target is unchanged, skips the
rebuild entirely. On LIBERO the robot's home configuration is usually
identical across init states (only object poses vary), in which case the
whole training run needs exactly ONE rebuild. `stats()` reports whether that
was the case so you can confirm rather than assume.

--------------------------------------------------------------------------
VERIFICATION IS NOT OPTIONAL
--------------------------------------------------------------------------
A constraint that silently fails to compile looks identical to a healthy
run. `LockMonitor` records max |q(t) - q_lock| over the episode. Your
screening run measured 1.25e-4 rad. If a training episode ever reports
drift above `drift_tol` the episode is not a faulted episode and the number
it produced is not evidence. Log it, and treat a rising drift as a bug.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# Panda arm joint names as they appear in the robosuite MJCF, in order j0..j6.
# robosuite prefixes model joints with "robot0_"; both spellings are searched
# because the prefix depends on the robosuite version and naming scheme.
PANDA_JOINT_BASENAMES = [f"joint{i + 1}" for i in range(7)]  # joint1..joint7


def panda_joint_name(joint_idx: int, prefix: str = "robot0_") -> str:
    """j0 -> 'robot0_joint1'. The handoff's j0 is Panda joint 1."""
    if not 0 <= joint_idx <= 6:
        raise ValueError(f"Panda has 7 arm joints; got joint_idx={joint_idx}")
    return f"{prefix}{PANDA_JOINT_BASENAMES[joint_idx]}"


def make_joint_lock_processor(joint_name: str, lock_value: float):
    """Return an xml_processor(xml_string) -> xml_string that locks `joint_name`.

    Idempotent: re-applying it to an already-processed XML updates the target
    rather than appending a second, conflicting constraint.
    """

    def processor(xml_string: str) -> str:
        root = ET.fromstring(xml_string)

        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")

        eq_name = f"fault_lock_{joint_name}"
        existing = None
        for child in equality.findall("joint"):
            if child.get("name") == eq_name:
                existing = child
                break

        if existing is None:
            existing = ET.SubElement(equality, "joint")
            existing.set("name", eq_name)

        existing.set("joint1", joint_name)
        # polycoef = [c0, c1, c2, c3, c4]; with only c0 non-zero the
        # constraint is q == c0, i.e. a hard lock at `lock_value`.
        existing.set("polycoef", f"{float(lock_value):.10f} 0 0 0 0")
        existing.set("active", "true")
        # A stiff solref makes the constraint behave like a rigid stop
        # rather than a soft spring. Without it the joint creeps.
        existing.set("solref", "0.0002 1")
        existing.set("solimp", "0.9999 0.9999 0.0001 0.5 2")

        return ET.tostring(root, encoding="unicode")

    processor.joint_name = joint_name
    processor.lock_value = float(lock_value)
    return processor


def find_joint_qpos_addr(env, joint_name: str) -> int:
    """Index into sim.data.qpos for a 1-DOF hinge joint."""
    sim = env.env.sim
    try:
        jid = sim.model.joint_name2id(joint_name)
    except Exception as exc:  # pragma: no cover - depends on live sim
        names = list(getattr(sim.model, "joint_names", []))
        raise KeyError(
            f"joint '{joint_name}' not in model. Available: {names}"
        ) from exc
    return int(sim.model.jnt_qposadr[jid])


def read_joint_qpos(env, joint_name: str) -> float:
    return float(env.env.sim.data.qpos[find_joint_qpos_addr(env, joint_name)])


@dataclass
class LockMonitor:
    """Per-episode telemetry proving the joint actually stayed locked."""

    joint_name: str
    lock_value: float
    drift_tol: float = 1e-2

    max_drift: float = 0.0
    n_samples: int = 0

    def record(self, q: float) -> float:
        d = abs(float(q) - self.lock_value)
        self.max_drift = max(self.max_drift, d)
        self.n_samples += 1
        return d

    @property
    def ok(self) -> bool:
        return self.max_drift <= self.drift_tol

    def summary(self) -> dict:
        return {
            "fault/joint": self.joint_name,
            "fault/lock_value": self.lock_value,
            "fault/max_drift_rad": self.max_drift,
            "fault/lock_ok": self.ok,
            "fault/n_samples": self.n_samples,
        }


@dataclass
class JointLockManager:
    """Applies the lock across episodes with as few sim rebuilds as possible.

    Usage inside an episode reset:

        obs = mgr.reset_episode(env, initial_state)   # returns faulted obs
        ...
        mgr.monitor.record(mgr.read_q(env))           # every step
    """

    joint_idx: int = 0
    prefix: str = "robot0_"
    enabled: bool = True
    drift_tol: float = 1e-2

    joint_name: str = field(init=False)
    _qpos_addr: int | None = field(default=None, init=False)
    _applied_lock: float | None = field(default=None, init=False)
    _targets: dict = field(default_factory=dict, init=False)
    _rebuilds: int = field(default=0, init=False)
    monitor: LockMonitor | None = field(default=None, init=False)

    def __post_init__(self):
        self.joint_name = panda_joint_name(self.joint_idx, self.prefix)

    # ------------------------------------------------------------------ #

    def read_q(self, env) -> float:
        if self._qpos_addr is None:
            self._qpos_addr = find_joint_qpos_addr(env, self.joint_name)
        return float(env.env.sim.data.qpos[self._qpos_addr])

    def reset_episode(self, env, initial_state, init_id: int | None = None):
        """Reset to `initial_state` with the fault active. Returns obs."""
        env.reset()
        obs = env.set_init_state(initial_state)

        if not self.enabled:
            self.monitor = None
            return obs

        # Determine the lock target for this init state (cached).
        if init_id is not None and init_id in self._targets:
            lock_value = self._targets[init_id]
        else:
            lock_value = self.read_q(env)
            if init_id is not None:
                self._targets[init_id] = lock_value

        # Only rebuild if the compiled constraint does not already match.
        if self._applied_lock is None or abs(self._applied_lock - lock_value) > 1e-9:
            env.env.set_xml_processor(
                make_joint_lock_processor(self.joint_name, lock_value)
            )
            self._applied_lock = lock_value
            self._rebuilds += 1
            self._qpos_addr = None  # model recompiled; address may move
            env.reset()
            obs = env.set_init_state(initial_state)

        self.monitor = LockMonitor(
            joint_name=self.joint_name,
            lock_value=lock_value,
            drift_tol=self.drift_tol,
        )
        return obs

    def step_record(self, env) -> float:
        if self.monitor is None:
            return 0.0
        return self.monitor.record(self.read_q(env))

    def stats(self) -> dict:
        vals = list(self._targets.values())
        return {
            "fault/enabled": self.enabled,
            "fault/joint": self.joint_name,
            "fault/n_rebuilds": self._rebuilds,
            "fault/n_init_states_seen": len(vals),
            "fault/lock_target_spread": (
                float(np.ptp(vals)) if len(vals) > 1 else 0.0
            ),
        }
