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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18


class MultiUAVGymEnv(gym.Env):
    def __init__(self, world_size=1000, step_size=15.0, max_steps=600,
                 sensor_range=200.0):
        super().__init__()
        self.world_size = world_size
        self.step_size = step_size
        self.max_steps = max_steps
        self.sensor_range = sensor_range
        self.num_uavs = 4  # 改为4架无人机
        self.formation_distance = 30.0  # 方形边长
        self.center_point = np.array([200, 200], dtype=np.float32)  # 红色固定点
        self.center_velocity = np.array([5.0, 5.0], dtype=np.float32)  # 中心点移动速度 (东北方向)
        self.target_center = np.array([801.03906, 801.03906], dtype=np.float32)  # 目标中心点
        self.initial_formation_reached = False  # 是否到达初始理想位置标志
        self.formation_reached_step = None  # 记录队形形成的时间步

        # 添加多个障碍物 - 现在有5个障碍物
        self.obstacle_positions = np.array([
            [600, 450],  # 中心障碍物
            [230, 400],  # 新增障碍物1
            [630, 630],  # 新增障碍物2
            [400, 400],
            [450, 600]

        ], dtype=np.float32)
        self.obstacle_radius = 20.0  # 障碍物半径
        self.obstacle_penalty_coef = 1.0  # 障碍物惩罚系数
        self.safe_distance = 30.0  # 安全距离

        # 计算理想位置
        self.ideal_positions = self._calculate_ideal_positions()

        # 轨迹记录
        self.trajectories = None
        self.full_trajectories = None  # 新增：记录完整轨迹
        self.prev_dist_to_obstacle = None  # 记录上一步到最近障碍物的距离
        self.previous_actions = None  # 记录上一步的动作

        # 修改动作空间：每个无人机的速度向量 (2维)，但限制幅度
        self.action_space = spaces.Box(low=-0.3, high=0.3, shape=(2,), dtype=np.float32)  # 缩小动作范围

        # 观测空间：自身位置(2) + 理想位置相对偏差(2) + 无人机索引(1) + 最近障碍物相对位置(2) = 7维
        self.observation_space = spaces.Box(
            low=-world_size, high=world_size,
            shape=(7,), dtype=np.float32
        )

        # 全局状态空间：所有无人机位置 + 理想位置 + 所有障碍物位置
        self.state_space = spaces.Box(
            low=-world_size, high=world_size,
            shape=(self.num_uavs * 2 + 2 + 2 * len(self.obstacle_positions),), dtype=np.float32
        )

        # 初始化无人机位置
        self.uav_positions = None
        self.current_step = 0

    def _calculate_ideal_positions(self):
        """修改理想位置：方形编队"""
        # 方形边长
        side_length = 100.0

        # 0号无人机在东北方向（45度）
        angle0 = math.pi / 4
        x0 = self.center_point[0] + side_length * math.cos(angle0)
        y0 = self.center_point[1] + side_length * math.sin(angle0)

        # 1号无人机在东南方向（135度）
        angle1 = 3 * math.pi / 4
        x1 = self.center_point[0] + side_length * math.cos(angle1)
        y1 = self.center_point[1] + side_length * math.sin(angle1)

        # 2号无人机在西南方向（225度）
        angle2 = 5 * math.pi / 4
        x2 = self.center_point[0] + side_length * math.cos(angle2)
        y2 = self.center_point[1] + side_length * math.sin(angle2)

        # 3号无人机在西北方向（315度）
        angle3 = 7 * math.pi / 4
        x3 = self.center_point[0] + side_length * math.cos(angle3)
        y3 = self.center_point[1] + side_length * math.sin(angle3)

        return np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)

    def _get_nearest_obstacle_info(self, uav_position):
        """获取距离无人机最近的障碍物信息"""
        min_dist = float('inf')
        nearest_obstacle = None

        for obstacle_pos in self.obstacle_positions:
            dist = np.linalg.norm(uav_position - obstacle_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_obstacle = obstacle_pos

        return nearest_obstacle, min_dist

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

        # 初始化到最近障碍物的距离记录
        self.prev_dist_to_obstacle = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            _, min_dist = self._get_nearest_obstacle_info(self.uav_positions[i])
            self.prev_dist_to_obstacle[i] = min_dist

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

            # 获取最近障碍物信息
            nearest_obstacle, obstacle_dist = self._get_nearest_obstacle_info(self.uav_positions[i])

            # 计算避障方向（远离障碍物）
            obstacle_direction = self.uav_positions[i] - nearest_obstacle
            norm_obstacle_direction = obstacle_direction / (np.linalg.norm(obstacle_direction) + 1e-8)

            # 动态混合目标方向和避障方向
            if obstacle_dist < (self.obstacle_radius + self.safe_distance * 2):
                # 根据障碍物距离调整混合权重
                obstacle_weight = max(0, 1.0 - obstacle_dist / (self.obstacle_radius + self.safe_distance * 2))
                target_weight = 1.0 - obstacle_weight

                # 混合方向
                mixed_direction = (target_weight * norm_direction_to_target +
                                   obstacle_weight * norm_obstacle_direction)
                norm_mixed_direction = mixed_direction / (np.linalg.norm(mixed_direction) + 1e-8)
            else:
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

        # 计算奖励（惩罚） - 添加避障相关奖励
        rewards, avoidance_rewards = self._calculate_rewards()

        # 检测碰撞
        collisions = np.zeros(self.num_uavs)
        for i in range(self.num_uavs):
            _, min_dist = self._get_nearest_obstacle_info(self.uav_positions[i])
            if min_dist < self.obstacle_radius:
                collisions[i] = 1

        # 终止条件：中心点到达目标位置 或者 步数超过最大步数
        center_reached = np.linalg.norm(self.center_point - self.target_center) < 1.0
        terminated = center_reached
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), self._get_state(), rewards, terminated, truncated, {
            "avoidance_rewards": avoidance_rewards, "collisions": collisions}

    def _get_obs(self):
        """获取每个无人机的局部观测 - 添加障碍物相关观测"""
        obs = []
        for i, pos in enumerate(self.uav_positions):
            # 自身位置 + 理想位置相对偏差 + 无人机索引
            single_obs = list(pos)
            single_obs.extend(pos - self.ideal_positions[i])
            single_obs.append(i)  # 无人机索引

            # 添加最近障碍物相对位置
            nearest_obstacle, _ = self._get_nearest_obstacle_info(pos)
            obstacle_rel_pos = pos - nearest_obstacle
            single_obs.extend(obstacle_rel_pos)

            obs.append(np.array(single_obs, dtype=np.float32))
        return obs

    def _get_state(self):
        """获取全局状态 - 添加障碍物信息"""
        state = []
        # 修正：直接遍历无人机位置，不要使用enumerate
        for pos in self.uav_positions:
            state.extend(pos)
        state.extend(self.center_point)
        for obstacle_pos in self.obstacle_positions:
            state.extend(obstacle_pos)  # 添加所有障碍物位置
        return np.array(state, dtype=np.float32)

    def _calculate_rewards(self):
        """计算奖励（惩罚） - 添加避障相关奖励计算"""
        rewards = np.zeros(self.num_uavs)
        avoidance_rewards = np.zeros(self.num_uavs)  # 单独记录避障奖励

        for i in range(self.num_uavs):
            # 计算与理想位置的距离偏差
            dist = np.linalg.norm(self.uav_positions[i] - self.ideal_positions[i])
            # 惩罚与理想位置的偏差
            rewards[i] = -dist * 0.1  # 缩放因子

            # 获取最近障碍物信息
            _, curr_dist_to_obstacle = self._get_nearest_obstacle_info(self.uav_positions[i])
            prev_dist = self.prev_dist_to_obstacle[i]

            # 检查碰撞
            if curr_dist_to_obstacle < self.obstacle_radius:
                # 碰撞惩罚
                collision_penalty = -100.0
                rewards[i] += collision_penalty
                avoidance_rewards[i] = collision_penalty  # 记录碰撞惩罚值
                # 更新距离记录并跳过其他避障计算
                self.prev_dist_to_obstacle[i] = curr_dist_to_obstacle
                continue  # 发生碰撞时跳过后续避障计算

            # 基于绝对距离的惩罚（连续惩罚）- 单障碍物思路
            if curr_dist_to_obstacle < (self.obstacle_radius + self.safe_distance):
                # 距离越近惩罚越大（指数增长）
                penalty = np.exp(-curr_dist_to_obstacle / 10) * self.obstacle_penalty_coef
                rewards[i] -= penalty
                avoidance_rewards[i] = -penalty

            # 基于距离变化的奖励/惩罚 - 单障碍物思路
            if curr_dist_to_obstacle < (self.obstacle_radius + self.safe_distance * 1.5):  # 扩大感知范围
                if curr_dist_to_obstacle < prev_dist:
                    # 靠近障碍物，给予惩罚
                    penalty = (prev_dist - curr_dist_to_obstacle) * self.obstacle_penalty_coef
                    rewards[i] -= penalty
                    avoidance_rewards[i] = -penalty

            # 更新上一步的距离记录
            self.prev_dist_to_obstacle[i] = curr_dist_to_obstacle

        return rewards, avoidance_rewards

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
        """保存轨迹图像 - 添加障碍物显示"""
        # 设置字体为Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.tick_params(axis='both', which='major', labelsize=18)

        # 绘制目标中心点
        target_center_plot = ax.plot(self.target_center[0], self.target_center[1], 'go', ms=10, label='目标中心点')

        # 绘制所有障碍物 - 现在有5个障碍物
        obstacle_colors = ['red', 'orange', 'brown', 'gray', 'pink']
        first_obstacle_handle = None
        for idx, obstacle_pos in enumerate(self.obstacle_positions):
            obstacle_circle = plt.Circle(
                (obstacle_pos[0], obstacle_pos[1]),
                self.obstacle_radius,
                color=obstacle_colors[idx], alpha=0.3
            )
            ax.add_patch(obstacle_circle)
            obstacle_plot = ax.plot(obstacle_pos[0], obstacle_pos[1], 'ro', ms=5, alpha=0.5)
            if idx == 0:
                first_obstacle_handle = obstacle_plot[0]

        # 绘制无人机（不同颜色）
        colors = ['blue', 'green', 'purple', 'orange']  # 增加橙色
        labels = ['Agent 0', 'Agent 1', 'Agent 2', 'Agent 3']  # 增加UAV3
        uav_plots = []
        for i in range(self.num_uavs):
            uav_plot = ax.plot(self.uav_positions[i][0], self.uav_positions[i][1],
                               color=colors[i], marker='o', ms=8, label=labels[i])
            uav_plots.append(uav_plot[0])

        # 绘制无人机之间的连线（形成方形）
        # 连接相邻的无人机形成方形
        connections = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in connections:
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

        # 显示图例
        all_legend_items = [target_center_plot[0], first_obstacle_handle] + uav_plots
        all_legend_labels = ['Target', 'Obstacles'] + labels
        ax.legend(all_legend_items, all_legend_labels, loc='lower right', fontsize=18)

        ax.grid(True)

        # 保存图像
        os.makedirs("image/evaluation", exist_ok=True)
        plt.savefig(f"image/evaluation/min_step_episode_{episode}.jpg", dpi=300)
        plt.close()

    def render(self, mode='human', filename=None):
        """可视化 - 添加障碍物显示"""
        # 设置字体为Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'

        # 创建一个新的图形对象
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.tick_params(axis='both', which='major', labelsize=18)

        # 绘制目标中心点
        target_center_plot = ax.plot(self.target_center[0], self.target_center[1], 'go', ms=10, label='目标中心点')

        # 绘制所有障碍物 - 现在有5个障碍物
        obstacle_colors = ['red', 'orange', 'brown', 'gray', 'pink']
        first_obstacle_handle = None
        for idx, obstacle_pos in enumerate(self.obstacle_positions):
            obstacle_circle = plt.Circle(
                (obstacle_pos[0], obstacle_pos[1]),
                self.obstacle_radius,
                color=obstacle_colors[idx], alpha=0.3
            )
            ax.add_patch(obstacle_circle)
            obstacle_plot = ax.plot(obstacle_pos[0], obstacle_pos[1], 'ro', ms=5, alpha=0.5)
            if idx == 0:
                first_obstacle_handle = obstacle_plot[0]

        # 绘制无人机（不同颜色）
        colors = ['blue', 'green', 'purple', 'orange']  # 增加橙色
        labels = ['Agent 0', 'Agent 1', 'Agent 2', 'Agent 3']  # 增加UAV3
        uav_plots = []
        for i in range(self.num_uavs):
            uav_plot = ax.plot(self.uav_positions[i][0], self.uav_positions[i][1],
                               color=colors[i], marker='o', ms=8, label=labels[i])
            uav_plots.append(uav_plot[0])

        # 绘制无人机之间的连线（形成方形）
        # 连接相邻的无人机形成方形
        connections = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in connections:
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

        # 显示图例
        all_legend_items = [target_center_plot[0], first_obstacle_handle] + uav_plots
        all_legend_labels = ['Target', 'Obstacles'] + labels
        ax.legend(all_legend_items, all_legend_labels, loc='lower right', fontsize=18)

        #ax.set_title(
            #f'Step {self.current_step}/{self.max_steps}, Formation: {"Reached" if self.initial_formation_reached else "Not Reached"}')
        ax.grid(True)

        # 保存或显示图像
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300)
            plt.close(fig)  # 关闭图像以避免内存泄漏
        elif mode == 'human':
            plt.show()