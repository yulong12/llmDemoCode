# 完整PPO训练代码（增强版）
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
from rewardCalculator import RewardCalculator
from advantageCalculator import AdvantageCalculator
from critic_loss import CriticLossCalculator
from ppo_actor_loss import PPOLossCalculator
# ========== 工具函数 ==========
def masked_mean(tensor, mask):
    """带掩码的均值计算
    Args:
        tensor (torch.Tensor): 输入张量
        mask (torch.Tensor): 掩码张量（0/1）
    Returns:
        torch.Tensor: 掩码均值
    """
    return (tensor * mask).sum() / mask.sum()

def clip_by_value(tensor, min_val, max_val):
    """数值截断函数
    Args:
        tensor (torch.Tensor): 输入张量
        min_val (float): 最小值
        max_val (float): 最大值
    Returns:
        torch.Tensor: 截断后的张量
    """
    return torch.clamp(tensor, min=min_val, max=max_val)

# ========== 经验回放机制 ==========
class ExperienceReplayBuffer:
    """经验回放缓冲区
    Attributes:
        buffer_size (int): 缓冲区最大容量
        device (torch.device): 存储设备（CPU/GPU）
        states (list): 状态列表
        actions (list): 动作列表
        rewards (list): 奖励列表
        next_states (list): 下一状态列表
        masks (list): 终止标记列表
    """
    def __init__(self, buffer_size=10000, device="cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.masks = []
    
    def add_experience(self, state, action, reward, next_state, mask):
        """添加单条经验
        Args:
            state (torch.Tensor): 当前状态
            action (torch.Tensor): 执行动作
            reward (float): 获得奖励
            next_state (torch.Tensor): 下一状态
            mask (int): 终止标记（0表示终止）
        """
        if len(self) >= self.buffer_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.masks.pop(0)
        self.states.append(state.to(self.device))
        self.actions.append(action.to(self.device))
        self.rewards.append(torch.tensor(reward).to(self.device))
        self.next_states.append(next_state.to(self.device))
        self.masks.append(torch.tensor(mask).to(self.device))
    
    def sample_batch(self, batch_size):
        """随机采样批次数据
        Args:
            batch_size (int): 采样数量
        Returns:
            tuple: 包含各元素的批次张量
        """
        indices = torch.randint(0, len(self), (batch_size,))
        return (
            torch.stack([self.states[i] for i in indices]),
            torch.stack([self.actions[i] for i in indices]),
            torch.stack([self.rewards[i] for i in indices]),
            torch.stack([self.next_states[i] for i in indices]),
            torch.stack([self.masks[i] for i in indices])
        )
    
    def __len__(self):
        return len(self.states)

# ========== 核心训练类 ==========
class PPOTrainer:
    def __init__(self, model, ref_model, tokenizer, config):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 模型配置
        self.model = model.to(self.device)
        self.ref_model = ref_model.to(self.device)
        self.tokenizer = tokenizer
        
        # 训练配置
        self.config = config
        self.buffer = ExperienceReplayBuffer(device=self.device)
        
        # 初始化组件
        self.reward_calculator = RewardCalculator(kl_coef=config.kl_coef)
        self.advantage_calculator = AdvantageCalculator(
            gamma=config.gamma, 
            lam=config.lam
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    def generate_experience(self, prompts):
        """生成响应并存入经验池
        Args:
            prompts (list): 输入提示列表
        """
        # 编码输入
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length
        ).to(self.device)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                do_sample=True,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应
        responses = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        # 获取参考模型输出（用于KL惩罚）
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
        
        # 存储经验数据
        for i in range(len(prompts)):
            self.buffer.add_experience(
                state=inputs["input_ids"][i],
                action=outputs[i],
                reward=self._compute_reward(responses[i]),
                next_state=None,  # 对话任务无明确状态转移
                mask=inputs["attention_mask"][i]
            )

    def _compute_reward(self, response):
        """计算综合奖励（示例使用情感分析）
        Args:
            response (str): 模型生成的响应
        Returns:
            float: 综合奖励值
        """
        # 实际应用应替换为专业奖励模型
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            device=0 if torch.cuda.is_available() else -1
        )
        return sentiment_pipe(response)[0]["score"]

    def train_step(self, batch_size):
        """执行单步训练
        Args:
            batch_size (int): 训练批次大小
        Returns:
            float: 当前步骤的损失值
        """
        # 从经验池采样
        states, actions, rewards, next_states, masks = self.buffer.sample_batch(batch_size)
        
        # 获取模型输出
        model_outputs = self.model(
            input_ids=states,
            attention_mask=masks,
            output_hidden_states=True
        )
        
        # 计算损失
        loss = self._compute_loss(
            states=states,
            actions=actions,
            rewards=rewards,
            masks=masks,
            logits=model_outputs.logits
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def _compute_loss(self, states, actions, rewards, masks, logits):
        """综合损失计算
        Args:
            states (torch.Tensor): 输入状态 [B, L]
            actions (torch.Tensor): 执行动作 [B, L]
            rewards (torch.Tensor): 奖励信号 [B]
            masks (torch.Tensor): 注意力掩码 [B, L]
            logits (torch.Tensor): 模型输出logits [B, L, V]
        """
        # ========== 准备输入数据 ==========
    # 获取参考模型输出（旧策略logprobs）
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=states, attention_mask=masks)
            old_logprobs = ref_outputs.logits.log_softmax(dim=-1).gather(
                2, actions.unsqueeze(-1)
            ).squeeze(-1) * masks
        
        # 当前模型输出（新策略logprobs）
        new_logprobs = logits.log_softmax(dim=-1).gather(
            2, actions.unsqueeze(-1)
        ).squeeze(-1) * masks

        # 价值预测（Critic输出）
        values = self.model.get_value(states, masks)  # 假设模型有get_value方法
        
        # ========== 调用损失计算模块 ==========
        # 初始化计算器
        critic_loss_calculator = CriticLossCalculator(
            cliprange_value=self.config.cliprange_value
        )
        ppo_loss_calculator = PPOLossCalculator(
            cliprange=self.config.cliprange,
            vf_coef=self.config.vf_coef
        )     

        # 计算价值损失
        vf_loss, _ = critic_loss_calculator.compute_vf_loss(
            values=values.detach(),     # 旧价值预测
            vpreds=values,              # 新价值预测
            returns=rewards,            # 实际收益
            mask=masks.float()          # 注意转换为浮点型
        )
        
        # 计算策略损失
        ratio = torch.exp(new_logprobs - old_logprobs)
        loss_dict = ppo_loss_calculator.compute_loss(
            old_logprobs=old_logprobs,
            values=values.detach(),
            logits=logits,
            logprobs=new_logprobs,
            mask=masks,
            advantages=self.advantage_calculator.compute_advantages(values, rewards, masks),
            returns=rewards
        )
        # ========== 综合损失 ==========
        total_loss = loss_dict['pg_loss'] + self.config.vf_coef * vf_loss
        
        # 记录监控指标
        self.metrics = {
            'pg_loss': loss_dict['pg_loss'].item(),
            'vf_loss': vf_loss.item(),
            'total_loss': total_loss.item(),
            'approx_kl': (old_logprobs - new_logprobs).mean().item()
        }
        return total_loss

    def save_checkpoint(self, epoch, path="checkpoints"):
        """保存模型检查点
        Args:
            epoch (int): 当前训练轮次
            path (str): 保存路径
        """
        os.makedirs(path, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }, f"{path}/checkpoint_epoch{epoch}.pt")
        print(f"Checkpoint saved at epoch {epoch}")

# ========== 配置与执行 ==========
class Config:
    # 设备配置
    batch_size = 4                  # 根据显存调整
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 训练参数
    kl_coef = 0.1
    gamma = 0.99
    lam = 0.95
    cliprange = 0.2
    cliprange_value = 0.2
    vf_coef = 0.5
    learning_rate = 1e-5
    weight_decay = 0.01
    
    # 生成参数
    max_input_length = 128
    max_length = 256
    top_p = 0.9
    
    # 训练控制
    epochs = 10
    save_interval = 2               # 每2个epoch保存一次

if __name__ == "__main__":
    # 初始化组件
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(Config.device)
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2").to(Config.device)
    
    # 训练数据
    class TrainingDataset(Dataset):
        def __init__(self):
            self.prompts = [
                "Explain machine learning in simple terms",
                "What are the benefits of renewable energy?",
                "How does photosynthesis work?",
                "Describe the process of making chocolate"
            ]
        
        def __len__(self):
            return len(self.prompts)
        
        def __getitem__(self, idx):
            return {"prompt": self.prompts[idx]}
    
    # 创建训练器
    trainer = PPOTrainer(model, ref_model, tokenizer, Config())
    dataloader = DataLoader(
        TrainingDataset(), 
        batch_size=Config.batch_size,
        shuffle=True
    )
    
    # 训练循环
    for epoch in range(Config.epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # 生成经验数据
            trainer.generate_experience(batch["prompt"])
            
            # 参数更新
            loss = trainer.train_step(Config.batch_size)
            epoch_loss += loss
        
        # 定期保存模型
        if (epoch + 1) % Config.save_interval == 0:
            trainer.save_checkpoint(epoch+1)
        
        print(f"Epoch {epoch+1} | Avg Loss: {epoch_loss/len(dataloader):.4f}")