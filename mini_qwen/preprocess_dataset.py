import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from utils.utils import find_files, tokenize_dataset

# 配置参数
TRUNK_SIZE = 512  # 分块长度
TMP_PATH = "/archive/share/cql/aaa/tmp"  # 缓存目录
DATA_PATH = "data/pt"  # 原始数据目录
CONFIG_PATH = "models/Qwen2.5-0.5B"  # 模型配置和权重目录

# 输出分词数据集的路径
output_tokenized_path = os.path.join(DATA_PATH, "tokenized_dataset")

# 加载分词器
model_path = CONFIG_PATH
tokenizer = AutoTokenizer.from_pretrained(model_path)

if not os.path.isdir(output_tokenized_path):
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
    train_dataset.save_to_disk(output_tokenized_path)
    print(f"Tokenized dataset saved to {output_tokenized_path}")
else:
    print(f"Tokenized dataset already exists at {output_tokenized_path}")
