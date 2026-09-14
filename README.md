PC-MAPPO: Phase-Conditioned Rollout Buffering for Multi-Robot Collaborative Motion Planning
This repository contains the official implementation of the paper: "PC-MAPPO: Implementation and Empirical Evaluation of Phase-Conditioned Rollout Buffering for Multi-Robot Collaborative Motion Planning".
Multi-robot collaborative tasks—like formation maintenance and safe navigation—often consist of sequential phases that create non-stationary task distributions. Traditional on-policy methods (like MAPPO) can suffer from distribution shifts when the rollout buffer is flooded with high-variance obstacle-avoidance data, degrading previously learned geometric formation behaviors.
PC-MAPPO (Phase-Conditioned Multi-Agent Proximal Policy Optimization) addresses this practical tension. Instead of a single pooled buffer, this implementation uses separate phase-conditioned rollout buffers (Formation Establishment and Navigation/Avoidance). It performs batch-triggered updates using scheduled, phase-specific per-transition weights to coordinate learning without requiring exact phase-balanced sampling.
Key Architecture Components
Phase-Conditioned Rollout Buffering: Maintains separate buffers for formation and navigation data, assigning dynamic phase-dependent weights during PPO loss aggregation to mitigate gradient interference.

Graph Attention Networks (GAT): Encodes scenario-specific interaction graphs (e.g., full connectivity or bidirectional nearest-neighbor) to capture dynamic relational representations among agents.

APF-Inspired Directional Guidance: A deterministic execution layer that smoothly blends target and repulsive directions, ensuring obstacle-aware navigation while the learned policy scales the displacement magnitude.
Environments & Scenarios
The framework utilizes a custom 2D continuous Multi-Agent Gymnasium environment ($1000 \times 1000$ simulation units). The state and reward structures are explicitly designed for Decentralized Partially Observable Markov Decision Processes (Dec-POMDPs) under a Centralized Training with Decentralized Execution (CTDE) paradigm.
We provide multi-robot configurations (3-agent triangular and 4-agent square formations) across four increasing levels of complexity:

Obstacle-Free Formation: Baseline environment focusing on rapid swarm convergence to the target topology.

Static Obstacle Fields: Includes 3, 5, or 7 static circular obstacles to introduce multi-objective conflict (formation vs. safety).

Dynamic Environments: Non-stationary scenarios featuring multiple moving obstacles with distinct movement speeds and areas.

Narrow Passage Constraints: Complex topological constraints using rotated rectangular obstacles that introduce opposing repulsive directions.
nstallation

Clone the repository and install the required dependencies:
git clone https://github.com/hhqq233/PC-MAPPO-Phase-Conditioned-Rollout-Buffering.git
cd PC-MAPPO-Phase-Conditioned-Rollout-Buffering

# Create a virtual environment (optional but recommended)
conda create -n pc-mappo python=3.9
conda activate pc-mappo

# Install dependencies
pip install -r requirements.txt
Usage

Training

To train the PC-MAPPO agents in a specific scenario, run the training script. The training process runs for 500,000 environment steps with updates triggered every 2048 transitions.
# Example command (adjust based on your actual entrypoint script)
python train.py --scenario static --num_agents 3 --num_obstacles 5
Evaluation

To evaluate the trained policies and render the formation trajectories (averaging over 50 episodes):
python eval.py --scenario dynamic --num_agents 3 --load_model /path/to/model
