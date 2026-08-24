# Reproducibility protocol

## Reference environment

The current validated stack uses:

- Python 3.10.12
- PyTorch 2.2.0 with CUDA 11.8
- NumPy 1.26.4
- MuJoCo 3.3.0
- robosuite 1.4.1
- OpenVLA-OFT commit `e4287e94541f459edc4feabc4e181f537cd569a8`
- LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- OpenVLA-OFT combined checkpoint `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`

The requirements file pins the small REPAIR-side dependency surface. Install the full simulator and VLA environment from the upstream projects.

## Experimental invariants

These conditions are part of the experiment, not implementation details:

1. **Frozen base policy.** OpenVLA-OFT and its action/proprioception heads remain in evaluation mode with gradients disabled.
2. **One construction path.** Training and evaluation both use `rl/build.py` and `rl/multitask_build.py`.
3. **Physical fault.** A MuJoCo equality constraint locks a Panda arm joint; the action vector is not masked to imitate a fault.
4. **Per-state lock target.** The locked angle is read after applying the episode's initial state, preventing an artificial arm teleport.
5. **Verified fault.** The compiled equality name and locked-joint drift are checked. A condition that cannot be verified is an error, not a healthy fallback.
6. **Unmodified gripper.** The residual controls six arm dimensions only. The base VLA retains the seventh gripper command.
7. **Held-out initial states.** Evaluation uses the tail of the LIBERO initial-state list; training never samples those indices.
8. **Globally held-out actuator.** The j2 condition is absent from training in the main protocol.
9. **Held-out-blind selection.** Curriculum selection uses healthy and seen-joint screening data only. Held-out values are stored separately for later reporting.
10. **Matched evaluation.** Baseline and residual policies use the same initial states, task horizon, VLA checkpoint, and fault implementation.

## Before a training run

```bash
python -m compileall -q rl faults scripts tests
python tests/test_joint_factorized.py
python tests/test_offline.py
bash -n scripts/*.sh
```

Then run a live smoke test on the intended machine. Confirm that the log records the expected fault equality and a maximum drift below the configured tolerance before committing GPU time.

## Curriculum construction

`rl/screen_to_curriculum.py` parses the frozen-VLA screening tables and retains a task only when:

- healthy success reaches the configured threshold; and
- the required number of seen training joints has nonzero signal and recovery headroom.

The selector does not receive held-out performance. It writes the training curriculum and held-out reference as different files. The checked-in main curriculum contains 28 tasks and 46 usable seen-joint cells; the main13 curriculum requires two usable cells and contains 13 tasks.

## Run metadata

For every reported run, retain:

- Git commit and dirty-state flag;
- training seed and physical GPU mapping;
- curriculum file checksum;
- OpenVLA-OFT and LIBERO commits;
- VLA checkpoint identifier;
- total environment steps and per-task exposure;
- residual scale, discount, n-step horizon, update-to-data ratio, and actor arm;
- per-episode task, condition, initial-state index, success, horizon, and fault-verification telemetry.

Logs, checkpoints, and rollouts are ignored by Git. Archive them in experiment storage and publish stable artifacts separately for a paper release.

## Statistical reporting

Report raw counts together with rates. For paired policies evaluated on the same initial states, use paired analyses. Do not treat multiple joint conditions on the same scene as independent observations. Most importantly, episode-level uncertainty does not replace variation across independently trained seeds.
