# REPAIR

**Reinforcement-Enabled Policy Adaptation for Impaired Robots with Vision-Language-Action Models**

[Project page](https://mkhoshnam.github.io/repair-vla/) · [Source code](https://github.com/mkhoshnam/repair-vla) · Paper forthcoming

REPAIR studies how a robot can continue a task when a persistent actuator fault changes the relationship between a Vision-Language-Action (VLA) policy's command and the motion the robot actually produces. The repository contains our current Joint-Factorized Capability Reinforcement Learning (JFCRL) implementation, residual Soft Actor-Critic (SAC) baselines, physical-fault simulation, screened curricula, evaluation tools, and offline regression tests.

> **Research status:** this is active research code. The 20% to 60% held-out-fault result shown below is a preliminary pilot result from 10 evaluation episodes and one training seed. It is evidence motivating the larger study, not yet a multi-seed final result. See [Project status](docs/PROJECT_STATUS.md) for the completed and planned experiments.

## Method

The pretrained OpenVLA-OFT policy remains frozen and produces the task-directed base action. REPAIR adds a bounded six-dimensional residual correction to the arm command; the VLA retains control of the gripper.

JFCRL builds its correction from execution rather than a symbolic fault label:

1. Each Panda arm joint contributes the same 28-dimensional history token containing commanded and realized motion.
2. A shared temporal encoder converts each joint history into a capability token—there is no learned joint-ID embedding.
3. The current manipulator Jacobian grounds every token in that joint's live kinematic role.
4. Cross-joint attention produces a global capability latent.
5. A FiLM-conditioned SAC actor and twin critics learn a residual action from task reward.
6. Joint-motion, end-effector-motion, and kinematic-consistency auxiliary losses train the capability representation from execution.

The physical impairment is a MuJoCo equality constraint that locks a joint at its own episode-initial angle. Runtime monitoring checks that the constraint was compiled and that joint drift stays within tolerance.

## Preliminary result

| Policy | Globally unseen actuator fault | Success |
|---|---:|---:|
| Frozen VLA | j2 | 2/10 (20%) |
| JFCRL residual policy | j2 | 6/10 (60%) |

The JFCRL policy was evaluated without additional adaptation on the held-out joint. The small sample and single training seed are important limitations; the repository includes the broader curricula and A/B protocol being used to test whether the effect survives more tasks and seeds.

## Repository layout

```text
repair-vla/
├── rl/                 # JFCRL, SAC, environments, training, and evaluation
├── faults/             # verified MuJoCo joint-lock fault implementations
├── scripts/            # screening, train, eval, and result-analysis entry points
├── configs/            # screened task pools and held-out-blind curricula
├── tests/              # CPU-only offline regression suites
├── docs/               # status and reproducibility notes
├── index.html           # GitHub Pages project website
├── *.mp4                # small website demonstration assets
├── requirements.txt
└── README.md
```

Large model weights, datasets, checkpoints, rollouts, logs, and Weights & Biases files are intentionally not versioned.

## Installation

The training and simulator entry points are an overlay on [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). They import OpenVLA-OFT's `experiments` package and LIBERO at runtime; those large upstream repositories are not vendored here.

The validated environment uses Linux, Python 3.10, PyTorch 2.2, CUDA 11.8, MuJoCo 3.3, robosuite 1.4.1, OpenVLA-OFT commit `e4287e9`, and LIBERO commit `8f1084e`. A CUDA-capable GPU and EGL rendering are required for full training and evaluation. The offline tests need only NumPy and PyTorch and run on CPU.

1. Install OpenVLA-OFT and LIBERO using the upstream OpenVLA-OFT setup guide. For exact reproduction, check out the commits above.
2. Clone REPAIR and expose both repository roots on `PYTHONPATH`:

```bash
git clone https://github.com/moojink/openvla-oft.git
git clone https://github.com/mkhoshnam/repair-vla.git

export OPENVLA_OFT_ROOT=/path/to/openvla-oft
export REPAIR_ROOT=/path/to/repair-vla
export PYTHONPATH="$REPAIR_ROOT:$OPENVLA_OFT_ROOT:${PYTHONPATH:-}"

cd "$REPAIR_ROOT"
python -m pip install -r requirements.txt
```

Install the PyTorch wheel appropriate for your CUDA version if it differs from the validated environment. The requirements file describes the REPAIR-side dependencies; the full VLA/simulator dependency stack comes from OpenVLA-OFT and LIBERO.

## Verify the installation

Run the two offline suites before starting a simulator or GPU job:

```bash
python tests/test_joint_factorized.py
python tests/test_offline.py
```

They cover capability-encoder invariances, SAC updates and serialization, residual arithmetic, gripper isolation, n-step replay, timeout handling, fault parsing, XML constraint behavior, curriculum sampling, and gated-actor checkpoint round trips. They do not replace a live MuJoCo smoke test.

## Training

The simplest screened-task workflow lists the tracked task pool, validates a configuration without constructing the full environment, and then launches training:

```bash
bash scripts/run_jfcrl_screened_tasks.sh list
bash scripts/run_jfcrl_screened_tasks.sh 5 --dry-run
GPU=0 SEED=7 bash scripts/run_jfcrl_screened_tasks.sh 5
```

The current held-out-blind v2 protocol compares the base JFCRL policy with a capability-gated actor. It uses the tracked curriculum unless `SCREEN` points to a raw screening log:

```bash
# Six-task diagnostic, two GPUs
GPU_A=0 GPU_B=1 bash scripts/run_jfcrl_v2.sh diag

# 28-task main curriculum
GPU_A=0 GPU_B=1 bash scripts/run_jfcrl_v2.sh main
```

Scripts assume the correct environment is already active. Alternatively, set `VENV=/path/to/venv`; it may name either the environment directory or its `bin/activate` file. Other useful overrides include `STEPS`, `SEED`, `VLA`, `RUNNAME`, and `RUNDIR`.

## Evaluation

Evaluate a multitask checkpoint on the globally unseen j2 condition and the same held-out LIBERO initial states:

```bash
bash scripts/eval_jfcrl_unseen_j2.sh runs/<run>/ckpt_<step>.pt 20
```

For direct control over conditions and policies:

```bash
python rl/eval_joint_factorized_multitask.py \
  --ckpt runs/<run>/ckpt_<step>.pt \
  --curriculum configs/curriculum_main.json \
  --conditions healthy 2 \
  --policies zero ckpt \
  --held_out --n_episodes 20
```

`zero` is the frozen-VLA baseline and `ckpt` is the learned residual policy. Evaluation writes per-episode records and summaries beneath the requested output directory.

## Screening and curricula

Screening identifies task/fault cells that retain enough sparse-reward signal to train a recovery policy:

```bash
GPU=0 N=5 bash scripts/run_screening_40_combined.sh
```

To regenerate a held-out-blind curriculum from a completed screening log:

```bash
python rl/screen_to_curriculum.py \
  --log /path/to/40_task_screening_combinedvla.log \
  --train_joints 0 4 5 6 --heldout 2 \
  --out configs/curriculum_main.json
```

Task selection uses only healthy performance and seen training joints. Held-out j2 values are written to a separate `.heldout_reference` file for reporting and cannot influence selection. See [Reproducibility](docs/REPRODUCIBILITY.md) for the protocol invariants and reference versions.

## Configuration guide

| File | Purpose |
|---|---|
| `curriculum_diag6.json` | Six-task capability-gate diagnostic |
| `curriculum_main13.json` | Thirteen tasks with at least two usable seen-fault cells |
| `curriculum_main.json` | Main 28-task, 46-cell held-out-blind curriculum |
| `curriculum_main_epbalanced.json` | Episode-balanced variant of the main curriculum |
| `jfcrl_screened_task_pool.tsv` | Ordered screened-task expansion pool |
| `jfcrl_unseen_j0_3suite.tsv` | Three-suite experiment holding out j0 |

## Citation

The paper is in preparation. Until a paper identifier is available, cite the software metadata in [`CITATION.cff`](CITATION.cff):

```bibtex
@software{repair2026,
  title  = {REPAIR: Reinforcement-Enabled Policy Adaptation for Impaired Robots with Vision-Language-Action Models},
  author = {Khoshnazar, Mohammad and Tezerjani, Mohammad Dehghani and Yang, Qing and Beetz, Michael},
  year   = {2026},
  url    = {https://github.com/mkhoshnam/repair-vla}
}
```

## Acknowledgments and license

REPAIR builds on OpenVLA-OFT, LIBERO, robosuite, and MuJoCo. Please cite those projects when using the corresponding components. Source code is released under the MIT License; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for attribution. Website videos remain part of the REPAIR project materials.
