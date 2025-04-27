import torch

def clip_by_value(tensor, min_val, max_val):
    """数值截断函数"""
    return torch.clamp(tensor, min=min_val, max=max_val)

def masked_mean(tensor, mask):
    """带掩码的均值计算"""
    return (tensor * mask).sum() / mask.sum()

class PPOLossCalculator:
    def __init__(self, cliprange=0.2, cliprange_value=0.2, vf_coef=0.5):
        self.config = type('', (), {
            'cliprange': cliprange,
            'cliprange_value': cliprange_value,
            'vf_coef': vf_coef
        })()

    def compute_loss(
        self,
        old_logprobs: torch.Tensor,  # 旧策略对数概率 [batch, seq]
        values: torch.Tensor,        # 旧价值预测 [batch, seq]
        logits: torch.Tensor,        # 新策略logits
        logprobs: torch.Tensor,      # 新策略对数概率 [batch, seq]
        mask: torch.Tensor,          # 有效标记 [batch, seq]
        advantages: torch.Tensor,    # 优势函数 [batch, seq]
        returns: torch.Tensor        # 总收益 [batch, seq]
    ):
        # ============== Critic损失计算 ==============
        vpredclipped = clip_by_value(
            logits, 
            values - self.config.cliprange_value,
            values + self.config.cliprange_value
        )
        vf_losses1 = (logits - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)

        # ============== Actor损失计算 ==============
        ratio = torch.exp(logprobs - old_logprobs)
        
        # 原始策略损失
        pg_losses = -advantages * ratio
        
        # 截断后的策略损失
        pg_losses2 = -advantages * torch.clamp(
            ratio, 
            1.0 - self.config.cliprange,
            1.0 + self.config.cliprange
        )
        
        # 取最大损失
        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)

        # ============== 总损失 ==============
        total_loss = pg_loss + self.config.vf_coef * vf_loss
        
        return {
            'total_loss': total_loss,
            'pg_loss': pg_loss,
            'vf_loss': vf_loss
        }

if __name__ == "__main__":
    # 测试用例
    calculator = PPOLossCalculator(cliprange=0.2, vf_coef=0.5)
    
    # 构造数据（batch_size=1, seq_len=3）
    old_logprobs = torch.tensor([[-1.0, -0.5, -1.2]])
    values = torch.tensor([[0.8, 0.9, 0.7]])
    logits = torch.tensor([[0.6, 1.1, 0.5]])
    logprobs = torch.tensor([[-1.2, -0.4, -1.5]])
    mask = torch.tensor([[1, 1, 0]])          # 第三个位置无效
    advantages = torch.tensor([[0.5, -0.3, 0.0]])
    returns = torch.tensor([[1.3, 0.8, 0.0]])
    
    # 执行计算
    loss_dict = calculator.compute_loss(
        old_logprobs, values, logits, logprobs, mask, advantages, returns
    )
    
    print("测试结果:")
    print(f"总损失: {loss_dict['total_loss'].item():.4f}")
    print(f"策略损失: {loss_dict['pg_loss'].item():.4f}")
    print(f"价值损失: {loss_dict['vf_loss'].item():.4f}")