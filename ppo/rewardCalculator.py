import torch

class RewardCalculator:
    def __init__(self, kl_coef=0.1):
        self.kl_ctl = type('', (), {'value': kl_coef})()  # 存储KL系数
        
    def _kl_penalty(self, p_logprob, q_logprob):
        """计算KL散度惩罚项"""
        return (p_logprob - q_logprob).exp() * (p_logprob - q_logprob - 1)
    
    def compute_rewards(self, scores, logprobs, ref_logprobs, masks):
        """
        计算完整奖励值
        参数:
            scores: torch.Tensor 形状 [batch_size]
            logprobs: torch.Tensor 形状 [batch_size, seq_len]
            ref_logprobs: torch.Tensor 形状 [batch_size, seq_len]
            masks: torch.Tensor 形状 [batch_size, seq_len]
        返回:
            rewards: 包含总奖励的列表
            non_score_rewards: 仅包含KL惩罚的奖励
            kls: KL散度值列表
        """
        rewards, non_score_rewards, kls = [], [], []
        
        # 逐样本处理
        for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
            # 计算每个token的KL散度
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(kl)
            
            # 计算基础奖励（负KL散度）
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            
            # 复制基础奖励作为总奖励的初始值
            reward = non_score_reward.clone()
            
            # 找到最后一个有效token的位置
            last_non_masked_index = torch.nonzero(mask, as_tuple=False)[-1]
            
            # 在最后一个有效token处加上分数
            reward[last_non_masked_index] += score
            
            rewards.append(reward)
            
        return rewards, non_score_rewards, kls
    
def masked_whiten(values, mask, shift_mean=True):
    """对 masked 数据进行标准化"""
    mean = (values * mask).sum() / mask.sum()
    var = ((values - mean)**2 * mask).sum() / mask.sum()
    whitened = (values - mean) / (var.sqrt() + 1e-8)
    return whitened if shift_mean else values

if __name__ == "__main__":
    # 测试用例
    calculator = RewardCalculator(kl_coef=0.1)
    
    # 构造测试数据
    scores = torch.tensor([0.8, 1.2])  # 两个样本的偏好得分
    logprobs = torch.tensor([  # 两个样本，每个序列长度3
        [-0.2, -0.5, -0.3],
        [-0.4, -0.6, -0.1]
    ])
    ref_logprobs = torch.tensor([
        [-0.1, -0.4, -0.4],
        [-0.3, -0.5, -0.2]
    ])
    masks = torch.tensor([  # 有效token标记
        [1, 1, 0],  # 第一个样本前两个token有效
        [1, 1, 1]   # 第二个样本全部有效
    ])
    
    # 执行计算
    rewards, non_score_rewards, kls = calculator.compute_rewards(
        scores, logprobs, ref_logprobs, masks
    )
    
    # 打印结果
    print("测试结果:")
    for i, (reward, kl) in enumerate(zip(rewards, kls)):
        print(f"\n样本 {i+1}:")
        print(f"KL散度: {kl.detach().numpy().round(4)}")
        print(f"总奖励: {reward.detach().numpy().round(4)}")
        print(f"掩码状态: {masks[i].numpy()}")