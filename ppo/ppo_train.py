import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rewardCalculator import RewardCalculator
from advantageCalculator import AdvantageCalculator
from critic_loss import CriticLossCalculator
from ppo_actor_loss import PPOLossCalculator

# 设备配置（支持5080 16G GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 神经网络模型定义 =====================
class ActorCritic(nn.Module):
    """PPO策略网络和价值网络"""
    def __init__(self, input_dim=4, hidden_dim=64, action_dim=2):
        super().__init__()
        # 共享特征提取层
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # 策略网络（Actor）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.LogSoftmax(dim=-1)  # 输出动作的对数概率
        )
        # 价值网络（Critic）
        self.critic = nn.Linear(hidden_dim, 1)  # 输出状态价值
        
    def forward(self, x):
        shared = self.shared_layer(x)
        log_probs = self.actor(shared)
        values = self.critic(shared).squeeze(-1)
        return log_probs, values

# ===================== 模拟数据集生成 =====================
class PPODataset(Dataset):
    """生成简单的PPO训练数据"""
    def __init__(self, num_samples=1000, seq_len=5, input_dim=4):
        self.states = torch.randn(num_samples, seq_len, input_dim)
        self.actions = torch.randint(0, 2, (num_samples, seq_len))
        self.old_logprobs = torch.randn(num_samples, seq_len)
        self.masks = torch.ones(num_samples, seq_len)  # 全有效掩码
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (
            self.states[idx].to(device),
            self.actions[idx].to(device),
            self.old_logprobs[idx].to(device),
            self.masks[idx].to(device)
        )

# ===================== 训练主函数 =====================
def train_ppo():
    # 超参数配置
    config = {
        'batch_size': 64,      # 适合16G GPU的批次大小
        'num_epochs': 10,      # 训练轮次
        'lr': 3e-4,            # 学习率
        'gamma': 0.99,         # 折扣因子
        'lam': 0.95,           # GAE系数
        'kl_coef': 0.1,        # KL惩罚系数
        'clip_ratio': 0.2       # PPO截断范围
    }
    
    # 初始化模型和优化器
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # 工具类实例化
    reward_calculator = RewardCalculator(kl_coef=config['kl_coef'])
    advantage_calculator = AdvantageCalculator(gamma=config['gamma'], lam=config['lam'])
    loss_calculator = PPOLossCalculator(cliprange=config['clip_ratio'])
    
    # 创建数据加载器
    dataset = PPODataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        epoch_loss = 0.0
        
        for batch in dataloader:
            states, actions, old_logprobs, masks = batch
            
            # 前向传播获取新策略的logits和值
            logits, values = model(states)
            
            # 计算新策略的对数概率（注意：实际场景需根据动作选择）
            # 这里简化处理，直接使用旧策略的对数概率生成模拟数据
            new_logprobs = old_logprobs + 0.1 * torch.randn_like(old_logprobs)
            
            # 模拟偏好得分（真实场景需要从环境中获取）
            scores = torch.randn(states.size(0))  # 随机生成得分
            
            # ===== 计算奖励和优势 =====
            # 计算KL惩罚后的奖励（使用模拟的参考策略对数概率）
            rewards, _, _ = reward_calculator.compute_rewards(
                scores, new_logprobs, old_logprobs, masks
            )
            rewards = torch.stack(rewards)  # 转换为张量
            
            # 计算优势函数和总收益
            _, advantages, returns = advantage_calculator.compute_advantages(
                values.detach(), rewards, masks
            )
            
            # ===== 计算损失并更新参数 =====
            optimizer.zero_grad()
            
            loss_dict = loss_calculator.compute_loss(
                old_logprobs=old_logprobs,
                values=values.detach(),
                logits=values,  # 这里简化处理，实际Critic应有独立输出
                logprobs=new_logprobs,
                mask=masks,
                advantages=advantages,
                returns=returns
            )
            
            # 反向传播
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            epoch_loss += loss_dict['total_loss'].item()
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {epoch_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train_ppo()