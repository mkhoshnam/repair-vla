# Project status

Last updated: 24 August 2026

REPAIR is an active research project. This document separates implemented and validated components from preliminary evidence and planned experiments.

## Implemented

- A frozen OpenVLA-OFT wrapper with the same action construction for training and evaluation.
- Bounded residual control over the six arm-action dimensions; gripper commands remain unchanged.
- MuJoCo equality-constraint joint locks applied at each episode's own initial joint angle.
- Runtime verification of compiled constraints and maximum locked-joint drift.
- Residual SAC, shared-context SAC, joint-factorized capability SAC, and multitask JFCRL.
- A shared temporal encoder, live Jacobian grounding, cross-joint attention, and self-supervised capability objectives.
- Capability-gated actor support with neutral initialization and checkpoint reconstruction.
- Resumable 40-task/fault screening and held-out-blind curriculum construction.
- Deterministic train/evaluation splits over LIBERO initial states.
- CPU-only regression tests for the algorithmic and fault-injection invariants.

## Preliminary evidence

In the current held-out-j2 pilot, the frozen VLA succeeds in 2/10 episodes and JFCRL succeeds in 6/10 without adapting on j2. This result uses one trained policy and only ten held-out initial states. It motivates the expanded evaluation but does not measure training-seed variability and should not be treated as a final significance claim.

The global-context baseline fits seen fault conditions strongly but did not show comparable transfer in the pilot. JFCRL's factorization and kinematic grounding are the working hypotheses for the difference.

## Current experiment sequence

| Stage | Curriculum | Budget | Purpose | Status |
|---|---:|---:|---|---|
| Diagnostic | 6 tasks / 11 seen cells | 240k steps per arm | A/B test the capability gate | Ready to reproduce |
| Intermediate | 13 tasks / 31 seen cells | 600k steps per arm | Require at least two usable seen faults per task | Ready |
| Main | 28 tasks / 46 seen cells | 1.4M steps per arm | Broad held-out-j2 evaluation | Ready |

Each v2 stage compares `arm=none` against `arm=gate`, preserves a globally unseen j2 condition, and evaluates healthy control alongside the faulted condition.

## Next evidence required

- Repeat the selected architecture across at least three independent training seeds.
- Report per-task and aggregate healthy/faulted performance with confidence intervals.
- Retain random-residual controls where they are needed to separate learned recovery from beneficial perturbation.
- Expand held-out evaluation beyond one actuator when screening support makes the split defensible.
- Publish final checkpoints, machine-readable episode outcomes, and the paper identifier when available.

## Known limitations

- The full stack has been validated on one Linux/CUDA/MuJoCo environment; portability beyond it still needs testing.
- Training depends on external OpenVLA-OFT and LIBERO checkouts and their model weights.
- Sparse task reward makes cells with a zero base success rate difficult to learn from; screening reduces but does not eliminate this issue.
- The current public repository does not include large checkpoints, raw rollouts, or the full screening log.
- The project page's result is preliminary and is intentionally labeled as such in the repository documentation.
