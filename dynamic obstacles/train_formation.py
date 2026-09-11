# train_formation.py
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
from torch.utils.tensorboard import SummaryWriter
from formation_env import MultiUAVGymEnv
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch
from torch_geometric.data import Data

# 检测可用设备
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个可用的 GPU")
    device = torch.device("cuda:0")
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("没有可用的 GPU，使用 CPU")


class GNNActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=1024, num_heads=8, num_layers=4):
        super().__init__()
        # 共享的GNN特征提取器（处理通信交互）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),  # 使用LayerNorm替代BatchNorm
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )
        self.gat_convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_size if i == 0 else hidden_size * num_heads
            self.gat_convs.append(GATConv(in_channels, hidden_size, heads=num_heads, dropout=0.1))

        # 每个无人机独立的策略头（输出动作）
        self.num_uavs = 3  # 假设3架无人机
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * num_heads, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, action_dim)
            ) for _ in range(self.num_uavs)
        ])
        self.log_std = nn.Parameter(torch.zeros(self.num_uavs, action_dim))  # 每个无人机独立的标准差

    def forward(self, data, node_types):
        x = self.obs_encoder(data.x)
        for conv in self.gat_convs:
            x = F.gelu(conv(x, data.edge_index))

        num_nodes = x.size(0)
        batch_size = num_nodes // self.num_uavs  # 计算批次大小

        # 为每个节点选择对应的策略头
        mu_list = []
        for node_idx in range(num_nodes):
            uav_type = node_types[node_idx % self.num_uavs]
            mu = torch.tanh(self.policy_heads[uav_type](x[node_idx]))
            mu_list.append(mu)
        mu = torch.stack(mu_list)  # [num_nodes, action_dim]

        # 扩展标准差以匹配批次大小
        std = torch.exp(self.log_std)  # [num_uavs, action_dim]
        std = std.repeat(batch_size, 1)  # [batch_size * num_uavs, action_dim]

        return mu, std


class MLPCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),  # 使用LayerNorm替代BatchNorm
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.net(state)


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# -------- 轨迹级缓冲区（按阶段）--------
# 修改Transition以包含阶段信息
Transition = namedtuple('Transition',
                        ['obs', 'state', 'action', 'reward', 'next_obs', 'next_state', 'logp', 'done', 'phase'])


class PhaseRolloutBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)
        self.episode_boundaries = deque(maxlen=capacity)  # 逐transition是否为episode结尾

    def add(self, *args, is_terminal: bool):
        self.data.append(Transition(*args))
        self.episode_boundaries.append(bool(is_terminal))

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()
        self.episode_boundaries.clear()

    def as_trajectories(self):
        trajs, curr = [], []
        for t, is_terminal in zip(self.data, self.episode_boundaries):
            curr.append(t)
            if is_terminal:
                trajs.append(curr)
                curr = []
        if len(curr) > 0:
            trajs.append(curr)
        return trajs


class GNNAgent:
    def __init__(self, env, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, lr=3e-5, train_iters=10,
                 batch_size=2048, hidden_size=512):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.entropy_coef = 0.05
        self.entropy_decay = 0.9995
        self.max_grad_norm = 0.5
        self.min_entropy_coef = 0.005

        # 初始阶段权重（7:3）
        self.phase1_weight = 0.7
        self.phase2_weight = 0.3
        self.phase_weights = [self.phase1_weight, self.phase2_weight]

        # 第二阶段权重（5:5）
        self.phase1_weight_after_formation = 0.5
        self.phase2_weight_after_formation = 0.5

        # 权重过渡参数
        self.weight_transition_rate = 0.01  # 每次权重调整的步长
        self.weight_transition_steps = 0  # 记录权重过渡步数
        self.max_weight_transition_steps = 100  # 最大过渡步数

        # 网络参数 - 观测维度改为7（只包含最近障碍物信息）
        obs_dim = 7  # 修改后的观测维度：自身位置(2) + 理想位置相对偏差(2) + 无人机索引(1) + 最近障碍物相对位置(2)
        action_dim = env.action_space.shape[0]
        state_dim = env.state_space.shape[0]  # 状态维度会自动适应

        # 创建边索引
        self.base_edge_index = self._create_fully_connected_edges(env.num_uavs)
        self.edge_index = self.base_edge_index.to(device)

        # 节点类型标识 - 现在将在_prepare_graph_data中动态生成
        self.node_types = None

        self.actor = GNNActorNetwork(obs_dim, action_dim, hidden_size).to(device)
        self.critic = MLPCriticNetwork(state_dim, hidden_size).to(device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 标准化器
        self.reward_normalizer = RunningMeanStd(shape=())
        self.state_normalizer = RunningMeanStd(shape=(state_dim,))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, 'min', factor=0.5, patience=5)

        # 经验缓冲区 - 分为两个阶段（轨迹级）
        self.buffer_phase1 = PhaseRolloutBuffer(capacity=200000)  # 形成编队阶段
        self.buffer_phase2 = PhaseRolloutBuffer(capacity=200000)  # 保持编队移动阶段

        # TensorBoard
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"runs/GNN_MAPPO_{now}")
        self.episode_count = 0
        self.step_count = 0
        self.last_formation_log_step = 0
        self.formation_reached_in_episode = False  # 记录当前episode是否已形成编队
        self.current_episode_avoidance_reward = 0.0  # 记录当前episode的避障奖励总和
        self.current_episode_collisions = 0  # 记录当前episode的碰撞次数

    def update_normalizers(self):
        # 合并两个阶段的数据进行归一化
        if len(self.buffer_phase1) > 0:
            all_rewards = np.array([t.reward for t in self.buffer_phase1.data])
            all_states = np.array([t.state for t in self.buffer_phase1.data])

            if len(self.buffer_phase2) > 0:
                all_rewards = np.concatenate([all_rewards, np.array([t.reward for t in self.buffer_phase2.data])])
                all_states = np.concatenate([all_states, np.array([t.state for t in self.buffer_phase2.data])])

            self.reward_normalizer.update(all_rewards)
            self.state_normalizer.update(all_states)

    def _create_fully_connected_edges(self, num_nodes):
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def _prepare_graph_data(self, obs):
        """将观测数据转换为图数据，动态生成节点类型"""
        num_uavs = len(obs)
        x = torch.FloatTensor(np.array(obs)).to(device)

        # 动态生成节点类型标识
        node_types = torch.arange(num_uavs).to(device)

        # 创建批次图数据
        data = Data(x=x, edge_index=self.edge_index)
        return data, node_types

    def get_action(self, obs):
        graph_data, node_types = self._prepare_graph_data(obs)

        with torch.no_grad():
            mu, std = self.actor(graph_data, node_types)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

        actions = action.cpu().numpy()
        log_probs = log_prob.cpu().numpy()
        stds = std.mean().item()

        return actions, log_probs, stds

    def compute_gae(self, rewards, values, dones):
        batch_size = len(rewards)
        advantages = np.zeros(batch_size)
        last_advantage = 0

        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        return advantages, returns

    def _compute_gae_on_traj(self, traj):
        # 输入：单条trajectory（保持时间顺序）
        states = torch.FloatTensor(np.array([t.state for t in traj], dtype=np.float32)).to(device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in traj], dtype=np.float32)).to(device)
        rewards = np.array([np.mean(t.reward) for t in traj], dtype=np.float32)  # 使用平均奖励
        dones = np.array([t.done for t in traj], dtype=np.float32)
        phases = np.array([t.phase for t in traj], dtype=np.int32)  # 获取阶段信息

        # 归一化state
        mean = torch.FloatTensor(self.state_normalizer.mean).to(device)
        var = torch.FloatTensor(self.state_normalizer.var).to(device)
        norm_states = (states - mean) / (torch.sqrt(var) + 1e-8)
        norm_next_states = (next_states - mean) / (torch.sqrt(var) + 1e-8)

        with torch.no_grad():
            values = self.critic(norm_states).squeeze(-1)
            next_values = self.critic(norm_next_states).squeeze(-1)

        # GAE
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(traj))):
            nv = next_values[t].item() if t < len(traj) - 1 else 0.0
            delta = rewards[t] + self.gamma * nv * (1.0 - dones[t]) - values[t].item()
            adv[t] = lastgaelam = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * lastgaelam
        ret = values.cpu().numpy() + adv
        return torch.FloatTensor(adv).to(device), torch.FloatTensor(ret).to(device), values, phases

    def update(self):
        # 更新归一化器
        self.update_normalizers()

        # 至少需要 batch_size 条 transition
        total_buffer_size = len(self.buffer_phase1) + len(self.buffer_phase2)
        if total_buffer_size < self.batch_size:
            return None

        # 将缓冲区按轨迹取出
        trajs1 = self.buffer_phase1.as_trajectories()
        trajs2 = self.buffer_phase2.as_trajectories()

        # 根据是否已形成编队选择权重
        if self.formation_reached_in_episode:
            phase1_weight = self.phase1_weight_after_formation
            phase2_weight = self.phase2_weight_after_formation
        else:
            phase1_weight = self.phase1_weight
            phase2_weight = self.phase2_weight

        # 构建分阶段数据集
        def build_phase_dataset(trajs, phase_weight):
            if len(trajs) == 0:
                return None
            obs_list, act_list, logp_list, state_list, phase_list = [], [], [], [], []
            adv_list, ret_list = [], []
            for traj in trajs:
                adv, ret, values, phases = self._compute_gae_on_traj(traj)
                adv_list.append(adv)
                ret_list.append(ret)
                for t in traj:
                    obs_list.append(t.obs)
                    act_list.append(t.action)
                    logp_list.append(t.logp)
                    state_list.append(t.state)
                    phase_list.append(t.phase)  # 保留阶段信息
            # 阶段内优势归一化
            adv = torch.cat(adv_list, dim=0)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # 为每个样本添加阶段权重
            weights = torch.ones(len(obs_list)) * phase_weight

            return {
                'obs': obs_list,  # 留作构图
                'act': torch.FloatTensor(np.array(act_list)).to(device),
                'logp': torch.FloatTensor(np.array(logp_list)).to(device),
                'state': torch.FloatTensor(np.array(state_list)).to(device),
                'phase': torch.LongTensor(np.array(phase_list)).to(device),  # 添加阶段信息
                'adv': adv,
                'ret': torch.cat(ret_list, dim=0).to(device),
                'weights': weights.to(device)  # 添加权重
            }

        ds1 = build_phase_dataset(trajs1, phase1_weight)
        ds2 = build_phase_dataset(trajs2, phase2_weight)
        parts = [d for d in [ds1, ds2] if d is not None]
        if len(parts) == 0:
            return None

        # 合并两阶段样本（按条目数简单拼接；如需"最低配比"可在此加权采样）
        def cat_tensor(key):
            return torch.cat([p[key] for p in parts], dim=0)

        obs_batch = sum([p['obs'] for p in parts], [])  # 列表合并
        act_batch = cat_tensor('act')
        old_logp_batch = cat_tensor('logp').view(-1)  # 展开为[样本*无人机]
        state_batch = cat_tensor('state')
        phase_batch = cat_tensor('phase')  # 阶段信息
        adv_batch = cat_tensor('adv')
        ret_batch = cat_tensor('ret')
        weights_batch = cat_tensor('weights')  # 获取权重

        # 计算解释方差（基于当前critic）
        with torch.no_grad():
            mean = torch.FloatTensor(self.state_normalizer.mean).to(device)
            var = torch.FloatTensor(self.state_normalizer.var).to(device)
            norm_state = (state_batch - mean) / (torch.sqrt(var) + 1e-8)
            values_now = self.critic(norm_state).squeeze(-1)
            explained_var = 1 - torch.var(ret_batch - values_now) / (torch.var(ret_batch) + 1e-8)

        stats = {
            'explained_var': explained_var.item(),
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'entropy_loss': 0.0,
            'approx_kl': 0.0,
            'clip_frac': 0.0,
            'std': float(self.actor.log_std.exp().mean().detach().cpu().item())
        }

        # 准备图数据（每条样本一张图）
        graph_data_list = []
        node_types_list = []
        for obs in obs_batch:
            graph_data, node_types = self._prepare_graph_data(obs)
            graph_data_list.append(graph_data)
            node_types_list.append(node_types)
        batched_graph = Batch.from_data_list(graph_data_list)
        batched_node_types = torch.cat(node_types_list, dim=0)  # 合并节点类型

        # 将动作/对数概率展平到[样本*无人机, act_dim]
        actions_tensor = act_batch.view(-1, self.env.action_space.shape[0]).to(device)
        old_log_probs_tensor = old_logp_batch.view(-1).to(device)

        # 训练循环：minibatch + 目标KL早停
        target_kl = 0.015
        minibatch = min(4096, actions_tensor.size(0))
        idx_all = torch.randperm(actions_tensor.size(0), device=device)
        actor_loss_total = 0.0
        entropy_total = 0.0
        critic_loss_total = 0.0

        for epoch in range(self.train_iters):
            for start in range(0, idx_all.numel(), minibatch):
                mb = idx_all[start:start + minibatch]

                # Actor 前向
                mu, std = self.actor(batched_graph, batched_node_types)  # 使用动态节点类型
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(actions_tensor).sum(-1)

                ratio = (logp[mb] - old_log_probs_tensor[mb]).exp()
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                # 将阶段优势扩展到节点级
                num_uavs = self.env.num_uavs
                adv_expanded = adv_batch.repeat_interleave(num_uavs).to(device)

                # 将权重扩展到节点级
                weights_expanded = weights_batch.repeat_interleave(num_uavs).to(device)

                # 使用权重计算策略损失
                pg_loss_unweighted = torch.min(ratio * adv_expanded[mb], clipped_ratio * adv_expanded[mb])
                pg_loss = -(pg_loss_unweighted * weights_expanded[mb]).mean()

                entropy = dist.entropy().mean()
                loss_pi = pg_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                actor_loss_total += loss_pi.item()
                entropy_total += entropy.item()

                with torch.no_grad():
                    approx_kl = (old_log_probs_tensor[mb] - logp[mb]).mean().item()
                if approx_kl > 1.5 * target_kl:
                    stats['approx_kl'] = approx_kl
                    break

                # Critic 更新
                mean = torch.FloatTensor(self.state_normalizer.mean).to(device)
                var = torch.FloatTensor(self.state_normalizer.var).to(device)
                # 将 mb 的节点索引映射回样本索引（每个样本对应 num_uavs 个节点）
                sample_indices = (mb // num_uavs).unique()  # 获取唯一样本索引
                norm_state_mb = (state_batch[sample_indices] - mean) / (torch.sqrt(var) + 1e-8)
                v_pred = self.critic(norm_state_mb).squeeze(-1)
                v_target = ret_batch[sample_indices]

                # 使用权重计算价值损失
                v_loss_unweighted = F.mse_loss(v_pred, v_target, reduction='none')
                v_loss = (v_loss_unweighted * weights_batch[sample_indices]).mean()

                self.critic_optimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                critic_loss_total += v_loss.item()
            else:
                pass
            if stats['approx_kl'] > 0 and stats['approx_kl'] > 1.5 * target_kl:
                break

        denom = max(1, (idx_all.numel() // minibatch) * max(1, self.train_iters))
        stats['value_loss'] = critic_loss_total / denom
        stats['policy_loss'] = actor_loss_total / denom
        stats['entropy_loss'] = entropy_total / denom

        # 学习率调度器针对critic
        self.scheduler.step(stats['value_loss'])

        # 熵系数衰减
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)

        # 清空缓冲
        self.buffer_phase1.clear()
        self.buffer_phase2.clear()
        return stats

    def _update_phase_weights(self):
        """平滑更新阶段权重"""
        if self.formation_reached_in_episode and self.weight_transition_steps < self.max_weight_transition_steps:
            # 逐步过渡到第二阶段权重
            self.phase1_weight = np.clip(
                self.phase1_weight - self.weight_transition_rate,
                self.phase1_weight_after_formation,
                self.phase1_weight
            )
            self.phase2_weight = 1.0 - self.phase1_weight
            self.weight_transition_steps += 1

            # 记录权重变化
            self.writer.add_scalar('train/phase1_weight', self.phase1_weight, self.step_count)
            self.writer.add_scalar('train/phase2_weight', self.phase2_weight, self.step_count)

    def train(self, total_steps=500000):
        obs, state = self.env.reset()
        episode_rewards = []
        episode_lengths = []
        episode_collisions = []  # 记录每个episode的碰撞次数
        current_episode_rewards = []
        current_episode_avoidance_rewards = []  # 记录当前episode的避障奖励
        current_episode_length = 0

        for step in range(total_steps):
            self.step_count += 1
            current_episode_length += 1

            actions, log_probs, stds = self.get_action(obs)
            next_obs, next_state, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated

            if len(current_episode_rewards) == 0:
                current_episode_rewards = rewards.copy()
            else:
                current_episode_rewards = [cr + r for cr, r in zip(current_episode_rewards, rewards)]

            # 记录避障奖励
            avoidance_rewards = info.get("avoidance_rewards", np.zeros(self.env.num_uavs))
            current_episode_avoidance_rewards.append(np.mean(avoidance_rewards))

            # 记录碰撞次数
            collisions = info.get("collisions", np.zeros(self.env.num_uavs))
            self.current_episode_collisions += np.sum(collisions)

            # 检查是否形成编队
            if not self.formation_reached_in_episode and self.env.initial_formation_reached:
                self.formation_reached_in_episode = True
                print(f"Episode {self.episode_count + 1}: Formation reached at step {self.step_count}")
                self.writer.add_scalar('train/formation_reached_step', self.step_count, self.episode_count)

            # 更新阶段权重（平滑过渡）
            self._update_phase_weights()

            # 确定当前阶段
            current_phase = 2 if self.env.initial_formation_reached else 1

            # 根据当前阶段将经验存入不同的缓冲区（轨迹级）
            if self.env.initial_formation_reached:
                # 保持编队移动阶段
                self.buffer_phase2.add(obs, state, actions, rewards, next_obs, next_state, log_probs, done,
                                       current_phase,
                                       is_terminal=done)
            else:
                # 形成编队阶段
                self.buffer_phase1.add(obs, state, actions, rewards, next_obs, next_state, log_probs, done,
                                       current_phase,
                                       is_terminal=done)

            # 新增：计算编队偏移量并记录（降低频率）
            if self.env.initial_formation_reached and self.step_count - self.last_formation_log_step >= 100:
                formation_offset = 0.0
                for i in range(self.env.num_uavs):
                    dist = np.linalg.norm(
                        self.env.uav_positions[i] - self.env.ideal_positions[i]
                    )
                    formation_offset += dist
                formation_offset /= self.env.num_uavs
                self.writer.add_scalar('train/formation_offset',
                                       formation_offset, self.step_count)
                self.last_formation_log_step = self.step_count

            if done:
                # 计算平均避障奖励
                avg_avoidance_reward = np.mean(
                    current_episode_avoidance_rewards) if current_episode_avoidance_rewards else 0.0
                self.writer.add_scalar('train/avoidance_reward', avg_avoidance_reward, self.episode_count)

                # 记录碰撞次数
                self.writer.add_scalar('train/collisions', self.current_episode_collisions, self.episode_count)
                episode_collisions.append(self.current_episode_collisions)

                self.episode_count += 1
                episode_rewards.append(np.mean(current_episode_rewards))
                episode_lengths.append(current_episode_length)

                # 重置episode状态
                current_episode_rewards = []
                current_episode_avoidance_rewards = []
                current_episode_length = 0
                self.formation_reached_in_episode = False
                self.current_episode_collisions = 0  # 重置碰撞计数
                obs, state = self.env.reset()

                if len(episode_rewards) > 0:
                    mean_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(
                        episode_rewards)
                    mean_length = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(
                        episode_lengths)
                    mean_collisions = np.mean(episode_collisions[-10:]) if len(episode_collisions) >= 10 else np.mean(
                        episode_collisions)

                    # 记录当前使用的权重
                    current_phase1_weight = self.phase1_weight
                    current_phase2_weight = self.phase2_weight

                    print(f"Episode {self.episode_count}, "
                          f"Avg Reward: {mean_reward:.2f}, "
                          f"Avg Length: {mean_length:.1f}, "
                          f"Avg Avoidance Reward: {avg_avoidance_reward:.4f}, "
                          f"Collisions: {mean_collisions:.0f}, "
                          f"Center: {self.env.center_point}, "
                          f"Weights: {current_phase1_weight:.1f}:{current_phase2_weight:.1f}")

                    self.writer.add_scalar('train/ep_rew_mean', mean_reward, self.episode_count)
                    self.writer.add_scalar('train/ep_len_mean', mean_length, self.episode_count)
                    self.writer.add_scalar('train/ep_collisions_mean', mean_collisions, self.episode_count)
                    self.writer.add_scalar('train/center_x', self.env.center_point[0], self.episode_count)
                    self.writer.add_scalar('train/center_y', self.env.center_point[1], self.episode_count)
                    self.writer.add_scalar('train/phase1_weight', current_phase1_weight, self.episode_count)
                    self.writer.add_scalar('train/phase2_weight', current_phase2_weight, self.episode_count)
            else:
                obs, state = next_obs, next_state

            # 检查总缓冲区大小是否足够更新（rollout风格）
            total_buffer_size = len(self.buffer_phase1) + len(self.buffer_phase2)
            if total_buffer_size >= self.batch_size:
                stats = self.update()
                if stats:
                    self.writer.add_scalar('train/approx_kl', stats.get('approx_kl', 0), self.step_count)
                    self.writer.add_scalar('train/clip_fraction', stats.get('clip_frac', 0), self.step_count)
                    self.writer.add_scalar('train/entropy_loss', stats['entropy_loss'], self.step_count)
                    self.writer.add_scalar('train/explained_variance', stats['explained_var'], self.step_count)
                    self.writer.add_scalar('train/learning_rate', self.actor_optimizer.param_groups[0]['lr'],
                                           self.step_count)
                    self.writer.add_scalar('train/loss', stats['policy_loss'] + stats['value_loss'],
                                           self.step_count)
                    self.writer.add_scalar('train/policy_gradient_loss', stats['policy_loss'], self.step_count)
                    self.writer.add_scalar('train/value_loss', stats['value_loss'], self.step_count)
                    self.writer.add_scalar('train/std', stats['std'], self.step_count)

            # 定期保存模型
            if step % 10000 == 0 and step > 0:
                self.save_models()

        self.writer.close()

    def save_models(self):
        os.makedirs("models", exist_ok=True)
        torch.save(self.actor.state_dict(), "models/gnn_actor.pth")
        torch.save(self.critic.state_dict(), "models/gnn_critic.pth")

    def load_models(self):
        model_dir = "models-3个动态障碍物"
        try:
            self.actor.load_state_dict(torch.load(os.path.join(model_dir, "gnn_actor.pth"), map_location=device))
            self.critic.load_state_dict(torch.load(os.path.join(model_dir, "gnn_critic.pth"), map_location=device))
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")


def evaluate(env, agent, num_episodes=10, seed=42, save_evaluation_images=True, writer=None):
    """
    评估模型性能并记录关键指标。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    episode_rewards = []
    episode_lengths = []
    episode_collisions = []
    episode_formation_offsets = []  # 新增：记录编队偏差

    # 获取评估用的 TensorBoard Writer
    if writer is None:
        # 如果没有传入 writer，创建一个新的用于评估的 writer
        from torch.utils.tensorboard import SummaryWriter
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_writer = SummaryWriter(f"runs/GNN_MAPPO_eval_{now}")
    else:
        eval_writer = writer

    # 只有在需要保存图像时才创建目录
    if save_evaluation_images:
        os.makedirs("image/evaluation", exist_ok=True)
        # 为每一步渲染的图像创建一个单独的目录
        step_images_dir = "image/evaluation_steps"
        os.makedirs(step_images_dir, exist_ok=True)

    print("\n--- Starting Evaluation ---")

    for ep in range(num_episodes):
        obs, state = env.reset(seed=seed + ep)
        episode_reward = np.zeros(env.num_uavs)
        episode_length = 0
        episode_collision = 0

        current_episode_formation_offset = 0.0
        formation_offset_count = 0

        while True:
            actions, _, _ = agent.get_action(obs)
            next_obs, next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # 1. 记录碰撞次数
            collisions = info.get("collisions", np.zeros(env.num_uavs))
            episode_collision += np.sum(collisions)

            # 2. 计算编队偏移量 (仅在编队形成后计算和记录)
            if env.initial_formation_reached:
                formation_offset = 0.0
                for i in range(env.num_uavs):
                    dist = np.linalg.norm(
                        env.uav_positions[i] - env.ideal_positions[i]
                    )
                    formation_offset += dist
                formation_offset /= env.num_uavs
                current_episode_formation_offset += formation_offset
                formation_offset_count += 1

            episode_reward += rewards
            episode_length += 1

            # 3. 保存每一步的渲染图像
            if save_evaluation_images:
                step_image_path = f"{step_images_dir}/ep_{ep + 1}_step_{episode_length}.jpg"
                env.render(filename=step_image_path)

            if done:
                # 4. 保存最终轨迹图
                if save_evaluation_images:
                    env.save_trajectory_image(ep + 1, episode_length)

                # 5. 记录评估结果
                mean_episode_reward = np.mean(episode_reward)
                avg_formation_offset = (
                            current_episode_formation_offset / formation_offset_count) if formation_offset_count > 0 else 0.0

                episode_rewards.append(mean_episode_reward)
                episode_lengths.append(episode_length)
                episode_collisions.append(episode_collision)
                episode_formation_offsets.append(avg_formation_offset)

                # 6. 写入 TensorBoard
                eval_writer.add_scalar('eval/ep_reward', mean_episode_reward, ep + 1)
                eval_writer.add_scalar('eval/ep_length', episode_length, ep + 1)
                eval_writer.add_scalar('eval/ep_collisions', episode_collision, ep + 1)
                eval_writer.add_scalar('eval/ep_formation_offset', avg_formation_offset, ep + 1)

                print(f"Episode {ep + 1}: steps={episode_length}, reward={mean_episode_reward:.2f}, "
                      f"collisions={episode_collision}, offset={avg_formation_offset:.2f}, "
                      f"final_center={env.center_point}")
                break

            obs, state = next_obs, next_state

    # 7. 写入平均评估结果
    mean_rewards = np.mean(episode_rewards)
    mean_lengths = np.mean(episode_lengths)
    mean_collisions = np.mean(episode_collisions)
    mean_offsets = np.mean(episode_formation_offsets)

    # 使用 Step 0 记录全局平均值，方便对比
    eval_writer.add_scalar('eval/avg_reward', mean_rewards, 0)
    eval_writer.add_scalar('eval/avg_length', mean_lengths, 0)
    eval_writer.add_scalar('eval/avg_collisions', mean_collisions, 0)
    eval_writer.add_scalar('eval/avg_formation_offset', mean_offsets, 0)

    if writer is None:
        eval_writer.close()

    print("--- Evaluation Finished ---")
    return mean_rewards, mean_lengths, mean_collisions



if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MultiUAVGymEnv()
    agent = GNNAgent(env, lr=3e-5, batch_size=2048)

    # 训练模型
    agent.train(total_steps=500000)
    agent.save_models()
    print("\nTraining completed.")