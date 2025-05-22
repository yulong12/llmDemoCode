# 用到的库
- 1. trl 库
trl（Transformers Reinforcement Learning）是 Hugging Face 推出的一个用于大模型微调（如 SFT、RLHF 等）的库，专门支持大语言模型的有监督微调（Supervised Fine-Tuning, SFT）和强化学习微调（Reinforcement Learning from Human Feedback, RLHF）。
- 2. SFTConfig
SFTConfig 是一个配置类，用于定义有监督微调（SFT）训练的各种参数，比如 batch size、学习率、训练轮数、保存策略等。
你可以把它理解为训练的“参数说明书”。
- 3. SFTTrainer
SFTTrainer 是一个训练器类，封装了 Hugging Face Transformers 的训练流程，专门用于 SFT 任务。
它可以自动处理训练、评估、保存模型、日志记录等流程，让你更方便地对大模型进行有监督微调。

# 目录结构
- demo
    - data
    - demo
    - model
    - result
    - tmp

# 数据集
https://huggingface.co/datasets/HuggingFaceM4/ChartQA/tree/main/data

# base model
https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct

# 微调命令
```
nohup accelerate launch --config_file accelerate_config.yaml qwen2.5vl.py >logs/output_pt.log 2>&1 &
```
# wandb
需要获取wandb的key
https://wandb.ai/authorize
### 设置
- 可以通过环境变量指定`export WANDB_API_KEY=your_api_key_here
`
- 初始化时直接传入
`wandb.init(project="your-project-name", entity="your-username", api_key="your_api_key_here")`
