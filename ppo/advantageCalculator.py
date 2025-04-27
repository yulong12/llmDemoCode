import torch

def masked_whiten(values, mask, shift_mean=True):
    """对 masked 数据进行标准化"""
    mean = (values * mask).sum() / mask.sum()
    var = ((values - mean)**2 * mask).sum() / mask.sum()
    whitened = (values - mean) / (var.sqrt() + 1e-8)
    return whitened if shift_mean else values

class AdvantageCalculator:
    def __init__(self, gamma=0.99, lam=0.95, whiten_rewards=True):
        self.config = type('', (), {
            'gamma': gamma,
            'lam': lam,
            'whiten_rewards': whiten_rewards
        })()
        
    def compute_advantages(self, values, rewards, mask):
        """
        计算优势函数和总收益
        参数:
            values: 状态价值预测 [batch_size, seq_len]
            rewards: 即时奖励 [batch_size, seq_len]
            mask: 有效标记 [batch_size, seq_len]
        返回:
            values: 原始状态价值
            advantages: 优势函数
            returns: 总收益
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]
        
        # 应用掩码
        values = values * mask
        rewards = rewards * mask
        
        # 奖励白化（可选）
        if self.config.whiten_rewards:
            rewards = masked_whiten(rewards, mask, shift_mean=False)
        
        # 逆序计算每个时间步
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t+1] if t < gen_len-1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        
        # 反转并拼接结果
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        returns = advantages + values
        
        # 优势函数白化
        advantages = masked_whiten(advantages, mask)
        return values, advantages.detach(), returns

if __name__ == "__main__":
    # 测试用例
    calculator = AdvantageCalculator(gamma=0.9, lam=0.8)
    
    # 构造数据（batch_size=1, seq_len=3）
    values = torch.tensor([[0.5, 0.6, 0.4]]).float()
    rewards = torch.tensor([[1.0, 0.5, 0.0]]).float()
    mask = torch.tensor([[1, 1, 0]])  # 第三个位置无效
    
    # 执行计算
    values, advantages, returns = calculator.compute_advantages(values, rewards, mask)
    
    print("测试结果:")
    print(f"原始价值:\n{values.detach().numpy().round(3)}")
    print(f"优势函数:\n{advantages.detach().numpy().round(3)}")
    print(f"总收益:\n{returns.detach().numpy().round(3)}")