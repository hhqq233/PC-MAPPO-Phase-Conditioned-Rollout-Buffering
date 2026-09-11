# formation_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
from collections import deque
import os
import math

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 全局字体设置，确保风格统一
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18  # 全局默认字体增大
plt.rcParams['axes.unicode_minus'] = False


class MultiUAVGymEnv(gym.Env):
    def __init__(self, world_size=1000, step_size=15.0, max_steps=600,
                 sensor_range=200.0):
        super().__init__()
        self.world_size = world_size
        self.step_size = step_size
        self.max_steps = max_steps
        self.sensor_range = sensor_range
        self.num_uavs = 3  # 3架无人机
        self.formation_distance = 30.0  # 等边三角形边长
        self.center_point = np.array([200, 200], dtype=np.float32)  # 红色固定点
        self.center_velocity = np.array([5.0, 5.0], dtype=np.float32)  # 中心点移动速度 (东北方向)
        self.target_center = np.array([801.03906, 801.03906], dtype=np.float32)  # 目标中心点
        self.initial_formation_reached = False  # 是否到达初始理想位置标志
        self.formation_reached_step = None  # 记录队形形成的时间步

        # 移除障碍物相关代码
        # 计算理想位置
        self.ideal_positions = self._calculate_ideal_positions()

        # 轨迹记录
        self.trajectories = None
        self.full_trajectories = None  # 新增：记录完整轨迹
        self.previous_actions = None  # 记录上一步的动作

        # 修改动作空间：每个无人机的速度向量 (2维)，但限制幅度
        self.action_space = spaces.Box(low=-0.3, high=0.3, shape=(2,), dtype=np.float32)  # 缩小动作范围

        # 观测空间：自身位置(2) + 理想位置相对偏差(2) + 无人机索引(1) = 5维 (移除了障碍物相关维度)
        self.observation_space = spaces.Box(
            low=-world_size, high=world_size,
            shape=(5,), dtype=np.float32
        )

        # 全局状态空间：所有无人机位置 + 理想位置 (移除了障碍物)
        self.state_space = spaces.Box(
            low=-world_size, high=world_size,
            shape=(self.num_uavs * 2 + 2,), dtype=np.float32
        )

        # 初始化无人机位置
        self.uav_positions = None
        self.current_step = 0

    def _calculate_ideal_positions(self):
        """修改理想位置：0号在东北方向，1和2号形成等边三角形"""
        # 0号无人机在东北方向（45度）
        angle0 = math.pi / 4
        x0 = self.center_point[0] + 100 * math.cos(angle0)
        y0 = self.center_point[1] + 100 * math.sin(angle0)

        # 1号和2号无人机形成等边三角形
        angle1 = angle0 + 2 * math.pi / 3  # 120度间隔
        angle2 = angle0 + 4 * math.pi / 3  # 240度间隔

        x1 = self.center_point[0] + 100 * math.cos(angle1)
        y1 = self.center_point[1] + 100 * math.sin(angle1)
        x2 = self.center_point[0] + 100 * math.cos(angle2)
        y2 = self.center_point[1] + 100 * math.sin(angle2)

        return np.array([[x0, y0], [x1, y1], [x2, y2]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random_state = np.random.RandomState(seed)

        # 重置中心点和标志
        self.center_point = np.array([200, 200], dtype=np.float32)
        self.initial_formation_reached = False
        self.formation_reached_step = None  # 重置队形形成时间步
        self.previous_actions = np.zeros((self.num_uavs, 2), dtype=np.float32)  # 初始化动作历史

        # 在以中心点为圆心，半径100的圆内随机生成初始位置
        self.uav_positions = np.zeros((self.num_uavs, 2), dtype=np.float32)
        for i in range(self.num_uavs):
            angle = random_state.uniform(0, 2 * math.pi)
            radius = random_state.uniform(0, 100)
            self.uav_positions[i] = self.center_point + radius * np.array([math.cos(angle), math.sin(angle)])

        # 更新理想位置
        self.ideal_positions = self._calculate_ideal_positions()

        # 初始化轨迹记录
        self.trajectories = [deque(maxlen=100) for _ in range(self.num_uavs)]
        self.full_trajectories = [[] for _ in range(self.num_uavs)]  # 重置全轨迹
        for i, pos in enumerate(self.uav_positions):
            self.trajectories[i].append(pos.copy())
            self.full_trajectories[i].append(pos.copy())  # 记录初始位置

        self.current_step = 0
        return self._get_obs(), self._get_state()

    def step(self, actions):
        self.current_step += 1

        # 检查是否到达初始理想位置
        if not self.initial_formation_reached:
            threshold = 5.0
            if all(
                    np.linalg.norm(self.uav_positions[i] - self.ideal_positions[i]) < threshold
                    for i in range(self.num_uavs)
            ):
                self.initial_formation_reached = True
                self.formation_reached_step = self.current_step  # 记录队形形成的时间步

        # 如果已到达初始理想位置，移动中心点
        if self.initial_formation_reached:
            # 计算移动方向
            direction = self.target_center - self.center_point
            norm_direction = direction / (np.linalg.norm(direction) + 1e-8)

            # 应用中心点移动
            self.center_point += self.center_velocity * norm_direction
            self.center_point = np.clip(self.center_point, 0, self.world_size)

            # 更新理想位置
            self.ideal_positions = self._calculate_ideal_positions()

        # 移动无人机 - 修改动作执行逻辑
        for i in range(self.num_uavs):
            # 计算朝向理想位置的方向向量
            direction_to_target = self.ideal_positions[i] - self.uav_positions[i]
            norm_direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-8)

            # 无障碍物威胁，直接朝向目标
            norm_mixed_direction = norm_direction_to_target

            # 应用动作平滑 - 与上一步动作进行加权平均
            smooth_factor = 0.6  # 平滑因子，越大越平滑
            smoothed_action = (smooth_factor * self.previous_actions[i] +
                               (1 - smooth_factor) * actions[i])

            # 应用动作（在混合方向上移动）
            velocity = smoothed_action * self.step_size
            self.uav_positions[i] += velocity * norm_mixed_direction
            self.uav_positions[i] = np.clip(self.uav_positions[i], 0, self.world_size)

            # 保存当前动作用于下一步平滑
            self.previous_actions[i] = smoothed_action.copy()

        # 更新轨迹
        for i in range(self.num_uavs):
            self.trajectories[i].append(self.uav_positions[i].copy())
            self.full_trajectories[i].append(self.uav_positions[i].copy())

        # 计算奖励（惩罚） - 简化奖励函数，移除避障相关
        rewards = self._calculate_rewards()

        # 移除碰撞检测
        collisions = np.zeros(self.num_uavs)

        # 终止条件：中心点到达目标位置 或者 步数超过最大步数
        center_reached = np.linalg.norm(self.center_point - self.target_center) < 1.0
        terminated = center_reached
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), self._get_state(), rewards, terminated, truncated, {
            "avoidance_rewards": np.zeros(self.num_uavs), "collisions": collisions}

    def _get_obs(self):
        """获取每个无人机的局部观测 - 移除了障碍物相关观测"""
        obs = []
        for i, pos in enumerate(self.uav_positions):
            # 自身位置 + 理想位置相对偏差 + 无人机索引
            single_obs = list(pos)
            single_obs.extend(pos - self.ideal_positions[i])
            single_obs.append(i)  # 无人机索引

            # 移除了障碍物相对位置

            obs.append(np.array(single_obs, dtype=np.float32))
        return obs

    def _get_state(self):
        """获取全局状态 - 移除了障碍物信息"""
        state = []
        # 修正：直接遍历无人机位置，不要使用enumerate
        for pos in self.uav_positions:
            state.extend(pos)
        state.extend(self.center_point)
        # 移除了障碍物位置
        return np.array(state, dtype=np.float32)

    def _calculate_rewards(self):
        """计算奖励（惩罚） - 简化版本，只保留编队保持奖励"""
        rewards = np.zeros(self.num_uavs)

        for i in range(self.num_uavs):
            # 计算与理想位置的距离偏差
            dist = np.linalg.norm(self.uav_positions[i] - self.ideal_positions[i])
            # 惩罚与理想位置的偏差
            rewards[i] = -dist * 0.1  # 缩放因子

            # 移除所有障碍物相关的奖励计算

        return rewards

    # 其余方法保持不变...
    def get_trajectories(self):
        """获取当前轨迹"""
        return self.trajectories

    def _smooth_trajectory(self, trajectory, window_size=10):
        """使用滑动平均平滑轨迹"""
        if len(trajectory) < window_size:
            return trajectory

        smoothed = []
        for i in range(len(trajectory)):
            start = max(0, i - window_size // 2)
            end = min(len(trajectory), i + window_size // 2 + 1)
            segment = trajectory[start:end]
            avg_x = sum(p[0] for p in segment) / len(segment)
            avg_y = sum(p[1] for p in segment) / len(segment)
            smoothed.append([avg_x, avg_y])
        return smoothed

    def save_trajectory_image(self, episode, steps):
        """保存轨迹图像 - 移除障碍物显示"""

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)

        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=18)

        # 绘制目标中心点
        target_center_plot = ax.plot(self.target_center[0], self.target_center[1], 'go', ms=10, label='Target')

        # 绘制无人机（不同颜色）
        colors = ['blue', 'green', 'purple']
        labels = ['Agent 0', 'Agent 1', 'Agent 2']
        uav_plots = []
        for i in range(self.num_uavs):
            uav_plot = ax.plot(self.uav_positions[i][0], self.uav_positions[i][1],
                               color=colors[i], marker='o', ms=8, label=labels[i])
            uav_plots.append(uav_plot[0])

        # 绘制无人机之间的连线（形成三角形）
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                ax.plot(
                    [self.uav_positions[i][0], self.uav_positions[j][0]],
                    [self.uav_positions[i][1], self.uav_positions[j][1]],
                    'b-', linewidth=2, alpha=0.5
                )

        # 绘制无人机轨迹（从队形形成后开始）
        if self.formation_reached_step is not None:
            for i in range(self.num_uavs):
                if len(self.full_trajectories[i]) > self.formation_reached_step:
                    # 只取队形形成后的轨迹点
                    formation_trajectory = self.full_trajectories[i][self.formation_reached_step:]
                    if len(formation_trajectory) > 1:
                        # 应用平滑处理
                        smooth_traj = self._smooth_trajectory(formation_trajectory)
                        traj = np.array(smooth_traj)
                        ax.plot(traj[:, 0], traj[:, 1],
                                color=colors[i], linewidth=2, alpha=0.8)

        # 显示图例 - 增大字体
        all_legend_items = [target_center_plot[0]] + uav_plots
        all_legend_labels = ['Target'] + labels
        ax.legend(all_legend_items, all_legend_labels, loc='lower right', fontsize=18)

        ax.grid(True)

        # 保存图像
        os.makedirs("image/evaluation", exist_ok=True)
        plt.savefig(f"image/evaluation/min_step_episode_{episode}.jpg", dpi=300, bbox_inches='tight')
        plt.close()

    def render(self, mode='human', filename=None):
        """可视化 - 移除障碍物显示"""

        # 创建一个新的图形对象
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)

        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=18)

        # 绘制目标中心点
        target_center_plot = ax.plot(self.target_center[0], self.target_center[1], 'go', ms=10, label='Target')

        # 绘制无人机（不同颜色）
        colors = ['blue', 'green', 'purple']
        labels = ['Agent 0', 'Agent 1', 'Agent 2']
        uav_plots = []
        for i in range(self.num_uavs):
            uav_plot = ax.plot(self.uav_positions[i][0], self.uav_positions[i][1],
                               color=colors[i], marker='o', ms=8, label=labels[i])
            uav_plots.append(uav_plot[0])

        # 绘制无人机之间的连线（形成三角形）
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                ax.plot(
                    [self.uav_positions[i][0], self.uav_positions[j][0]],
                    [self.uav_positions[i][1], self.uav_positions[j][1]],
                    'b-', linewidth=2, alpha=0.5
                )

        # 绘制无人机轨迹（从队形形成后开始）
        if self.formation_reached_step is not None:
            for i in range(self.num_uavs):
                if len(self.full_trajectories[i]) > self.formation_reached_step:
                    # 只取队形形成后的轨迹点
                    formation_trajectory = self.full_trajectories[i][self.formation_reached_step:]
                    if len(formation_trajectory) > 1:
                        # 应用平滑处理
                        smooth_traj = self._smooth_trajectory(formation_trajectory)
                        traj = np.array(smooth_traj)
                        ax.plot(traj[:, 0], traj[:, 1],
                                color=colors[i], linewidth=2, alpha=0.8)

        # 显示图例 - 增大字体
        all_legend_items = [target_center_plot[0]] + uav_plots
        all_legend_labels = ['Target'] + labels
        ax.legend(all_legend_items, all_legend_labels, loc='lower right', fontsize=18)

        # 设置标题 - 增大字体
        #ax.set_title(
            #f'Step {self.current_step}/{self.max_steps}, Formation: {"Reached" if self.initial_formation_reached else "Not Reached"}',
            #fontsize=20
        #)
        ax.grid(True)

        # 保存或显示图像
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 关闭图像以避免内存泄漏
        elif mode == 'human':
            plt.show()