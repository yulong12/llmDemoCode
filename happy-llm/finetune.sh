#!/bin/bash
# 设置可见显卡
CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed finetune.py \
    --model_name_or_path autodl-tmp/qwen-1.5b \
    --train_files autodl-tmp/dataset/sft_data/BelleGroup.jsonl \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir autodl-tmp/output/sft \
    --evaluation_strategy no \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --logging_dir autodl-tmp/output/sft/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 8 \
    --save_total_limit 3 \
    --seed 42 \
    --max_seq_length 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --report_to swanlab
    # --resume_from_checkpoint autodl-tmp/output/sft/checkpoint-1000 \