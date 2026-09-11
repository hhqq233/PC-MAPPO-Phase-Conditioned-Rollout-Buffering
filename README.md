# PC-MAPPO-Phased-Curriculum-Learning-for-Multi-Robot-Collaborative-Motion-Planning
This repository contains the official source code for the paper **"PC-MAPPO: Phased Curriculum Learning for Multi-Robot Collaborative Motion Planning"**[cite: 2]. This project proposes a multi-agent reinforcement learning (MARL) framework equipped with a Phased Curriculum Episodic Rollout (PCER) mechanism, designed to solve decentralized collaborative formation and navigation tasks in environments containing static and dynamic obstacles.
## Core Features
* **PCER Mechanism:** Employs a dual-buffer architecture ($D_{form}$ and $D_{nav}$) to decouple conflicting phase objectives and utilizes phase-balanced sampling to mitigate catastrophic forgetting.
* **Topological Awareness:** Integrates Graph Attention Networks (GATConv) for the real-time encoding of dynamically changing neighborhood topologies and obstacle risks.
* **Residual Safety Guidance:** Incorporates an Artificial Potential Fields (APF) module that provides heuristic collision-avoidance guidance during both training and decentralized execution.

* ## Repository Structure
The codebase is organized into different simulation scenarios based on the experimental design in the paper:
* (Obstacle-Free): Baseline formation convergence configuration for a 3-agent system[cite: 2].
*  (Static Obstacles): Configuration for high-density environments with 3 static obstacles[cite: 2].
* (Dynamic Obstacles): Non-stationary environment configuration featuring moving obstacles[cite: 2].
*  (Square Formation): Code validating algorithmic scalability using a 4-agent square topology[cite: 2].

## Requirements
We recommend using Python 3.8 or higher. The primary dependencies include:
* [Gymnasium](https://gymnasium.farama.org/) (for constructing the 2D continuous simulation space)[cite: 2]
* PyTorch (core algorithm framework for GAT and MAPPO implementation)
* NumPy / Pandas / SciPy (for trajectory data processing and state matrix operations)
* Matplotlib (for visualizing learning curves and 2D motion trajectories)

## Quick Start
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/hhqq233/PC-MAPPO-Phased-Curriculum-Learning-for-Multi-Robot-Collaborative-Motion-Planning.git](https://github.com/hhqq233/PC-MAPPO-Phased-Curriculum-Learning-for-Multi-Robot-Collaborative-Motion-Planning.git)
   cd PC-MAPPO-Phased-Curriculum-Learning-for-Multi-Robot-Collaborative-Motion-Planning
