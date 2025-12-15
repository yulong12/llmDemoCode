# 自定义参数训练
python pre_train.py \
    --data_path ./data/seq_monkey_datawhale_truncated.jsonl \
    --device mps \
    --epochs 5 \
    --batch_size 8 \
    --save_interval 10 \
    --learning_rate 1e-4 \
    --accumulation_steps 16 \
    --out_dir ./my_model_output