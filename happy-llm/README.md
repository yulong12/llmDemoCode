# LLM 训练指南
## 数据集

- 下载预训练数据集
```
modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir your_local_dir
```
- 解压预训练数据集
```
tar -xvf your_local_dir/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2
```

- 下载SFT数据集
```
huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_3.5M_CN --local-dir BelleGroup
```
## 处理数据集
- deal_dataset.py: 处理数据集的Python脚本
- deal_sftData.py: 处理sft数据集的Python脚本

## 训练tokenizer
- tokenizer.py: 训练tokenizer的Python脚本
## 支持的设备

本项目支持多种设备进行训练：
- CUDA GPU (推荐用于大规模训练)
- MPS (Apple Silicon GPU，适用于Mac用户)
- CPU (通用设备，适用于所有平台)

## 1. 原生训练
### 原生预训练
- pre_train.py:预训练Python代码
- pre_train.sh:预训练Shell脚本

### 原生sft
- sft_train.py: sft训练Python脚本
- sft_train_cpu.sh，sft_train_gpu.sh，sft_train_mps.sh: sft训练Shell脚本

## 2. 使用deepspeed进行训练
### 分布式预训练
- pretrain.py:分布式训练Python脚本
- pretrain.sh:分布式训练Shell脚本
- ds_config_zero2.json:DeepSpeed配置文件 

## 分布式sft微调
- finetune.py：SFT训练Python脚本
- finetune.sh：SFT训练启动脚本 

## 3. LoRA微调
- lora_finetune.py: LoRA微调训练Python脚本
- lora_finetune.sh: LoRA微调训练Shell脚本
- ds_config_zero3.json: DeepSpeed配置文件