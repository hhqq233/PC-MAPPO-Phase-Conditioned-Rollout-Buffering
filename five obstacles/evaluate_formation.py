# evaluate_formation.py
import numpy as np
import torch
import os
import datetime
from formation_env import MultiUAVGymEnv
from train_formation import GNNAgent, evaluate  # 导入 evaluate 函数
from torch.utils.tensorboard import SummaryWriter  # 新增导入


# ... (保持原有设备检测不变) ...

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MultiUAVGymEnv()
    agent = GNNAgent(env, lr=3e-5, batch_size=2048)

    # 【新增/修改】控制是否渲染图片的 Flag
    # 将此设置为 True/False 来控制是否生成图像
    render_evaluation_images = True

    # 初始化 TensorBoard Writer 用于评估记录
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_writer = SummaryWriter(f"runs/GNN_MAPPO_eval_{now}")  # 新增

    # 加载模型
    try:
        # 【注意】请确保您的模型路径 'models-gnn-mp-7355-5个障碍物' 存在
        agent.load_models()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # 在遇到错误时关闭 writer
        eval_writer.close()
        exit(1)

    print("\nStarting evaluation...")
    print(f"Center point: {env.center_point}")
    print(f"Ideal positions: {env.ideal_positions}")
    print(f"Obstacle positions: {env.obstacle_positions}")
    print(f"Obstacle radius: {env.obstacle_radius}")

    # 将控制图片渲染的 flag 和 writer 传递给 evaluate 函数
    avg_reward, avg_length, avg_collisions = evaluate(
        env,
        agent,
        num_episodes=50,
        seed=seed,
        save_evaluation_images=render_evaluation_images,
        writer=eval_writer  # 传递 writer
    )

    # 关闭 TensorBoard Writer
    eval_writer.close()

    print(f"\nEvaluation Results (Averages over 10 episodes):")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.1f}")
    print(f"Average Collisions: {avg_collisions:.1f}")


if __name__ == "__main__":
    main()