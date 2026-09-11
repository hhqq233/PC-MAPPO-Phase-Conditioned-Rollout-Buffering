# evaluate_formation.py
import numpy as np
import torch
import os
import datetime
from formation_env import MultiUAVGymEnv
from train_formation import GNNAgent, evaluate
from torch.utils.tensorboard import SummaryWriter


# ... (Previous imports/code)

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MultiUAVGymEnv()
    agent = GNNAgent(env, lr=3e-5, batch_size=2048)

    # Change to True to see the trajectory images
    render_evaluation_images = True

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_writer = SummaryWriter(f"runs/GNN_MAPPO_eval_{now}")

    try:
        agent.load_models()  # Make sure path is correct
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        eval_writer.close()
        exit(1)

    print("\nStarting evaluation...")

    avg_reward, avg_length, avg_collisions = evaluate(
        env,
        agent,
        num_episodes=10,  # Reduce episodes for faster visual check
        seed=seed,
        save_evaluation_images=render_evaluation_images,
        writer=eval_writer
    )

    eval_writer.close()

    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.1f}")
    print(f"Average Collisions: {avg_collisions:.1f}")


if __name__ == "__main__":
    main()