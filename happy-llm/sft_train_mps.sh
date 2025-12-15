#!/bin/bash

# macOS MPS设备训练脚本
echo "Starting training on MPS device..."

# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 运行训练脚本，指定使用MPS设备
python sft_train.py \
    --device mps \
    --out_dir sft_model_215M_mps \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --dtype float32 \
    --num_workers 0 \
    --data_path ./BelleGroup_sft_tru.jsonl \
    --accumulation_steps 1 \
    --grad_clip 1.0 \
    --warmup_iters 0 \
    --log_interval 10 \
    --save_interval 100

echo "Training completed!"