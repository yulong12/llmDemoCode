#!/bin/bash

# CUDA GPU设备训练脚本
echo "Starting training on CUDA GPU..."

# 运行训练脚本，指定使用CUDA设备
python sft_train.py \
    --device cuda:0 \
    --out_dir sft_model_215M_gpu \
    --epochs 1 \
    --batch_size 64 \
    --learning_rate 2e-4 \
    --dtype bfloat16 \
    --num_workers 8 \
    --data_path ./BelleGroup_sft.jsonl \
    --accumulation_steps 8 \
    --grad_clip 1.0 \
    --warmup_iters 0 \
    --log_interval 100 \
    --save_interval 1000

echo "Training completed!"