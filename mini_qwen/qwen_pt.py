import os
import torch
from datasets import load_dataset,Dataset
import wandb
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AdamW,
)
from utils.utils import find_files,tokenize_dataset

# 训练相关参数
TRUNK_SIZE = 512  # 分块长度
TMP_PATH = "/archive/share/cql/aaa/tmp"  # 缓存目录
DATA_PATH = "data/pt"  # 原始数据目录
OUTPUT_PATH = "results/pt"  # 训练结果输出目录
CONFIG_PATH = "models/Qwen2.5-0.5B"  # 模型配置和权重目录
WANDB_LOG = True  # 是否启用wandb日志
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # CUDA内存分配优化

output_path = OUTPUT_PATH
model_path = CONFIG_PATH
# 加载模型配置
config = AutoConfig.from_pretrained(model_path)
# 初始化模型，使用bfloat16和flash_attention_2加速
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenized_datapath = os.path.join(DATA_PATH, "tokenized_dataset")

# 如果没有已分词数据集，则进行分词和分块处理
if not os.path.isdir(tokenized_datapath):
    directories = [
        "film_entertainment",
        "literature_emotion",
        "news_media",
    ]
    # 查找所有parquet文件并加载为Dataset
    data_files = find_files(directories)
    dataset = load_dataset("parquet", data_files=data_files, split="train", columns=["text"], cache_dir=TMP_PATH) 
    dataset = dataset.shuffle(seed=42)
    
    # 定义分词和分块的回调函数
    def map_callback(examples):
        result, _ = tokenize_dataset(examples, tokenizer, TRUNK_SIZE)
        return result
    # 批量分词和分块，提升处理效率
    train_dataset = dataset.map(
        map_callback,
        batched=True,
        batch_size=5000,
        remove_columns=dataset.column_names,
        num_proc=32,
    )
    # 保存分词后的数据集，避免重复处理
    train_dataset.save_to_disk(tokenized_datapath)
    
# 加载分词后的数据集
train_dataset = Dataset.load_from_disk(tokenized_datapath)

# 构建无MLM的语言建模数据整理器
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 设置训练参数
training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    per_device_train_batch_size=24,
    gradient_accumulation_steps=16,
    save_steps=10_000, 
    save_total_limit=3,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    report_to="wandb",
)

# 启动wandb日志
if WANDB_LOG:
    wandb.login()
    wandb.init(
        project="qwen-0.5B-pt",name="qwen-0.5B-pt"
    )

# 可选：自定义优化器（此处注释掉，Trainer会自动创建）
# optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# 初始化Trainer，负责训练流程
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    # optimizers=(optimizer, None)  # 可自定义优化器
)
torch.cuda.empty_cache()  # 训练前清理显存

# 开始训练
trainer.train()
# 保存最终模型和分词器
trainer.save_model()  
tokenizer.save_pretrained(output_path)