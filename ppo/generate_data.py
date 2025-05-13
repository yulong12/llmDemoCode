import gym
import numpy as np
import torch
from torch.distributions import Categorical
from ppo_train import ActorCritic
# 1. 初始化环境和策略模型
env = gym.make('CartPole-v1')
model = ActorCritic(input_dim=4, action_dim=2)  # 使用前面定义的模型
model.eval()

# 2. 定义数据收集参数
num_episodes = 1000     # 收集1000条轨迹
max_steps = 200         # 每条轨迹最多200步
data_dict = {
    'states': [],
    'actions': [],
    'logprobs': [],
    'rewards': [],
    'masks': [],
    'next_states': []
}

# 3. 开始数据收集
for _ in range(num_episodes):
    state = env.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        # 转换state为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作概率和值
        with torch.no_grad():
            log_probs, _ = model(state_tensor)
        
        # 创建概率分布并采样动作
        dist = Categorical(logits=log_probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action.item())
        
        # 记录数据
        data_dict['states'].append(state)
        data_dict['actions'].append(action.item())
        data_dict['logprobs'].append(logprob.item())
        data_dict['rewards'].append(reward)
        data_dict['masks'].append(0 if done else 1)
        data_dict['next_states'].append(next_state)
        
        # 更新状态
        state = next_state
        step += 1

# 4. 转换为numpy数组并保存
np.savez('ppo_data.npz',
         states=np.array(data_dict['states'], dtype=np.float32),
         actions=np.array(data_dict['actions'], dtype=np.int64),
         logprobs=np.array(data_dict['logprobs'], dtype=np.float32),
         rewards=np.array(data_dict['rewards'], dtype=np.float32),
         masks=np.array(data_dict['masks'], dtype=np.float32),
         next_states=np.array(data_dict['next_states'], dtype=np.float32))

print(f"成功生成 {len(data_dict['states'])} 条样本数据")