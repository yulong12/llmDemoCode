#!/bin/bash

# CPU设备训练脚本
echo "Starting training on CPU..."

# 运行训练脚本，指定使用CPU设备
python sft_train.py \
    --device cpu \
    --out_dir sft_model_215M_cpu \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --dtype float32 \
    --num_workers 0 \
    --data_path ./BelleGroup_sft.jsonl \
    --accumulation_steps 1 \
    --grad_clip 1.0 \
    --warmup_iters 0 \
    --log_interval 10 \
    --save_interval 100

echo "Training completed!"