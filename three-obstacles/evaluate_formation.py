# evaluate_formation.py
import numpy as np
import torch
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from formation_env import MultiUAVGymEnv
from train_formation import GNNAgent, evaluate

# --- Configuration Flags ---
RENDER_IMAGES = True  # Set to True to save step-by-step images and trajectory plots
LOG_TO_TENSORBOARD = True
EVAL_EPISODES = 10
SEED = 42
# ---------------------------

# Detect device
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} available GPU(s)")
    device = torch.device("cuda:0")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Initialize TensorBoard Writer
    writer = None
    if LOG_TO_TENSORBOARD:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/eval_GNN_{now}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging enabled: {log_dir}")

    env = MultiUAVGymEnv()
    agent = GNNAgent(env, lr=3e-5, batch_size=2048)

    # Load Models
    try:
        agent.load_models()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)

    print("\nStarting evaluation...")
    print(f"Render Images: {RENDER_IMAGES}")
    print(f"Center point: {env.center_point}")
    print(f"Ideal positions: {env.ideal_positions}")
    print(f"Obstacle positions: {env.obstacle_positions}")
    print(f"Obstacle radius: {env.obstacle_radius}")

    # Run Evaluation
    # Note: We pass global_step=0 since this is a standalone evaluation run
    avg_reward, avg_length, avg_collisions, avg_deviation = evaluate(
        env,
        agent,
        num_episodes=EVAL_EPISODES,
        seed=SEED,
        save_step_images=RENDER_IMAGES,
        writer=writer,
        global_step=0
    )

    print(f"\nFinal Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.1f}")
    print(f"Average Collisions: {avg_collisions:.1f}")
    print(f"Average Formation Deviation: {avg_deviation:.2f}")

    if writer:
        writer.close()

if __name__ == "__main__":
    main()