# Contributing

REPAIR is active research code. Issues and focused pull requests are welcome, especially for reproducibility, portability, tests, documentation, and independently confirmed fault-recovery results.

Before submitting a change:

```bash
python -m compileall -q rl faults scripts tests
python tests/test_joint_factorized.py
python tests/test_offline.py
bash -n scripts/*.sh
```

Changes to environment construction, action composition, fault injection, initial-state splitting, or held-out selection can change the scientific question. Describe such changes explicitly, add a regression test where possible, and do not compare results across the change as if the protocol were identical.

Do not commit model weights, datasets, rollout videos, raw run directories, credentials, machine-specific paths, or Weights & Biases state. Small project-page media already tracked at the repository root are the exception.

For result contributions, include the code commit, upstream dependency commits, seeds, curriculum checksum, model identifier, per-episode outcomes, and fault-verification telemetry.
