# PC-MAPPO: Phase-Conditioned Rollout Buffering for Multi-Robot Collaborative Motion Planning

This repository contains the PC-MAPPO implementation associated with the manuscript *PC-MAPPO: Phase-Conditioned Rollout Buffering for Multi-Robot Collaborative Motion Planning*.

PC-MAPPO retains the MAPPO optimizer and organizes on-policy transitions in separate formation and navigation buffers. At an update, the phase datasets are concatenated and assigned phase-specific per-transition loss weights. The implementation also uses separate phase-wise GAE and advantage normalization, scenario-specific fixed-graph GAT encoding, and APF-inspired directional guidance for obstacle-aware execution.

## Scope and interpretation

The phase coefficients in this implementation are per-transition loss weights, not exact phase-sampling proportions. Once formation has been reached, updates use effective weights of 0.5/0.5; therefore, any observed performance difference cannot be attributed to phase weighting alone. The phase-buffering ablation changes buffer organization, phase-wise GAE, advantage normalization, and loss weighting together, so it is a combined-implementation ablation rather than an isolated estimate of any single component.

The GAT topology is scenario-specific and fixed during an episode; it is not dynamically rebuilt at each step. The current implementation may concatenate formation segments across episodes when the formation buffer does not receive a terminal flag, and it uses a zero next-state value at non-terminal batch truncation. These implementation details may affect advantage estimation near rollout boundaries.

## Repository layout

- `obstacle-free/`: three-agent obstacle-free scenario.
- `three-obstacles/`, `five obstacles/`, and `seven obstacles/`: three-agent static-obstacle scenarios.
- `dynamic obstacles/`: moving-obstacle scenario.
- `square-formation/`: four-agent square-formation scenarios.

Each scenario directory contains:

- `train_formation.py` for training.
- `evaluate_formation.py` for evaluation.
- `formation_env.py` for the environment definition.

## Installation

```bash
git clone https://github.com/hhqq233/PC-MAPPO-Phase-Conditioned-Rollout-Buffering.git
cd PC-MAPPO-Phase-Conditioned-Rollout-Buffering
conda create -n pc-mappo python=3.9
conda activate pc-mappo
```

This repository does not currently include a `requirements.txt`. Before running an experiment, install versions of the packages used by the selected scenario, including PyTorch, NumPy, Gymnasium, TensorBoard, and PyTorch Geometric, in an environment compatible with the local hardware and CUDA configuration.

## Usage

Run commands from the relevant scenario directory:

```bash
cd "three-obstacles"
python train_formation.py
python evaluate_formation.py
```

For the obstacle-free scenario:

```bash
cd "obstacle-free"
python train_formation.py
python evaluate_formation.py
```

The training entry points are configured for 500,000 environment steps. The available evaluation entry points run 50 evaluation episodes. Evaluation requires a compatible trained checkpoint at the path expected by the selected scenario implementation; if model loading fails, do not interpret the output as an evaluation of a trained policy.

## Materials still required from the authors

The repository currently does not provide all materials required to reproduce every comparison reported in the manuscript. The following materials remain to be supplied:

- Checkpoints corresponding to the reported evaluations, together with their paths and checksums.
- Baseline implementations and configurations for MASAC, MADDPG, standard MAPPO, and all reported ablations.
- Raw evaluation results, TensorBoard logs, and plotting scripts used to produce the manuscript figures and tables.
- A versioned dependency file, such as `requirements.txt`, and a fixed release or tag corresponding to the submitted manuscript.

Accordingly, this repository documents the PC-MAPPO implementation and scenario configurations, but does not currently provide a complete reproduction package for all manuscript results.
