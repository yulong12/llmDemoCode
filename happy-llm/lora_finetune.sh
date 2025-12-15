#!/bin/bash
# 设置环境变量，使用HF镜像加速下载
export HF_ENDPOINT=https://hf-mirror.com

# 设置可见显卡
CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed lora_finetune.py \
    --model_name_or_path autodl-tmp/qwen-1.5b \
    --train_files autodl-tmp/dataset/sft_data/BelleGroup.jsonl \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --output_dir autodl-tmp/output/lora_sft \
    --evaluation_strategy no \
    --learning_rate 3e-4 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --logging_dir autodl-tmp/output/lora_sft/logs \
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
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --deepspeed ./ds_config_zero3.json \
    --report_to swanlab
    # --resume_from_checkpoint autodl-tmp/output/lora_sft/checkpoint-1000 \