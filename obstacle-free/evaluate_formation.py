# evaluate_formation.py
import numpy as np
import torch
from formation_env import MultiUAVGymEnv
from train_formation import GNNAgent, evaluate

# 检测可用设备
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个可用的 GPU")
    device = torch.device("cuda:0")
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("没有可用的 GPU，使用 CPU")


def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MultiUAVGymEnv()
    agent = GNNAgent(env, lr=3e-5, batch_size=2048)

    # 加载模型
    try:
        agent.load_models()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        exit(1)

    print("\nStarting evaluation...")
    print(f"Center point: {env.center_point}")
    print(f"Ideal positions: {env.ideal_positions}")

    # 控制是否渲染图片的flag
    RENDER_IMAGES = True

    # 评估并记录数据到 TensorBoard (日志位于 runs/GNN_MAPPO_...)
    evaluate(env, agent, num_episodes=50, seed=seed, render_images=RENDER_IMAGES)

    # 关闭 TensorBoard writer
    if hasattr(agent, 'writer'):
        agent.writer.close()


if __name__ == "__main__":
    main()