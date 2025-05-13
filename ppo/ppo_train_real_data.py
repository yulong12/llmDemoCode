import gym
import numpy as np
import torch
from torch.distributions import Categorical

# 初始化环境和策略模型
env = gym.make('CartPole-v1')
model = ActorCritic(input_dim=4, action_dim=2)  # 确保已定义ActorCritic类
model.eval()

# 数据收集参数
num_episodes = 1000
max_steps = 200
data_dict = {
    'states': [],
    'actions': [],
    'logprobs': [],
    'rewards': [],
    'masks': [],
    'next_states': []
}

# 数据收集循环
for _ in range(num_episodes):
    # 关键修改：正确处理新版Gym返回的元组
    state, info = env.reset()  # 提取状态和info字典
    state = np.array(state, dtype=np.float32)  # 确保转换为numpy数组
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # 转换为张量的优化方式
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            log_probs, _ = model(state_tensor)
        
        dist = Categorical(logits=log_probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        # 关键修改：处理新版Gym的step返回
        next_state, reward, done, truncated, info = env.step(action.item())
        next_state = np.array(next_state, dtype=np.float32)  # 确保类型一致
        
        # 记录数据
        data_dict['states'].append(state)
        data_dict['actions'].append(action.item())
        data_dict['logprobs'].append(logprob.item())
        data_dict['rewards'].append(reward)
        data_dict['masks'].append(0 if (done or truncated) else 1)  # 处理终止条件
        data_dict['next_states'].append(next_state)
        
        state = next_state
        step += 1

# 保存数据
np.savez('ppo_data.npz',
         states=np.array(data_dict['states'], dtype=np.float32),
         actions=np.array(data_dict['actions'], dtype=np.int64),
         logprobs=np.array(data_dict['logprobs'], dtype=np.float32),
         rewards=np.array(data_dict['rewards'], dtype=np.float32),
         masks=np.array(data_dict['masks'], dtype=np.float32),
         next_states=np.array(data_dict['next_states'], dtype=np.float32))