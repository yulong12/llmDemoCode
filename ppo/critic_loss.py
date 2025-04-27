import torch

def clip_by_value(tensor, min_val, max_val):
    """数值截断函数"""
    return torch.clamp(tensor, min=min_val, max=max_val)

def masked_mean(tensor, mask):
    """带掩码的均值计算"""
    return (tensor * mask).sum() / mask.sum()

class CriticLossCalculator:
    def __init__(self, cliprange_value=0.2):
        self.config = type('', (), {'cliprange_value': cliprange_value})()
        
    def compute_vf_loss(self, values, vpreds, returns, mask):
        """
        计算Critic损失
        参数:
            values: 旧价值预测 [batch_size, seq_len]
            vpreds: 新价值预测 [batch_size, seq_len]
            returns: 实际收益 [batch_size, seq_len]
            mask: 有效标记 [batch_size, seq_len]
        返回:
            vf_loss: 价值损失值
            vf_clipfrac: 截断触发比例
        """
        # 截断新预测值
        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value
        )
        
        # 计算两种损失
        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        
        # 取最大损失求平均
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        
        # 计算截断触发比例
        vf_clipfrac = masked_mean(
            (vf_losses2 > vf_losses1).float(), mask
        )
        
        return vf_loss, vf_clipfrac

if __name__ == "__main__":
    # 测试用例
    calculator = CriticLossCalculator(cliprange_value=0.2)
    
    # 构造数据（batch_size=1, seq_len=3）
    values = torch.tensor([[0.8, 0.9, 0.7]])    # 旧价值预测
    vpreds = torch.tensor([[1.0, 0.5, 0.6]])    # 新价值预测
    returns = torch.tensor([[0.9, 0.8, 0.0]])    # 实际收益
    mask = torch.tensor([[1, 1, 0]])             # 掩码
    
    # 执行计算
    vf_loss, clipfrac = calculator.compute_vf_loss(values, vpreds, returns, mask)
    
    print("测试结果:")
    print(f"Critic损失值: {vf_loss.item():.4f}")
    print(f"截断触发比例: {clipfrac.item():.2%}")