import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import torch
from torch.utils.data import Dataset
from torchdata.datapipes.iter import IterableWrapper
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    get_last_checkpoint
)
import swanlab

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "预训练模型参数地址"}
    )

@dataclass
class DataTrainingArguments:
    train_files: Optional[str] = field(default=None, metadata={"help": "训练数据路径"})
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "输入序列的最大长度"}
    )

def preprocess(sources, tokenizer, max_len):
    """
    将对话数据处理为模型可训练的格式
    只对assistant的回复计算loss，对system和human部分不计算loss
    """
    # 定义特殊token
    im_start = tokenizer("<|im_start|>").input_ids
    im_end = tokenizer("<|im_end|>").input_ids
    IGNORE_TOKEN_ID = -100  # 通常设置为-100
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('human').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    input_ids, targets = [], []
    
    # 处理每个样本
    for i in tqdm(range(len(sources)), desc="Processing conversations"):
        source = sources[i]
        # 确保以human开始
        if source[0]["from"] != "human":
            source = source[1:]
        
        input_id, target = [], []
        
        # 添加system message
        system_message = "You are a helpful assistant."
        system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
        input_id += system
        # system部分不计算loss
        target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
        
        assert len(input_id) == len(target)
        
        # 处理多轮对话
        for j, sentence in enumerate(source):
            role = sentence["from"]
            # 构建单轮对话
            role_tokens = _user if role == "human" else _assistant
            _input_id = im_start + role_tokens + tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            input_id += _input_id
            
            if role == "human":
                # human部分不计算loss
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
            else:
                # assistant部分需要计算loss
                _target = im_start + [IGNORE_TOKEN_ID] * len(role_tokens) + \
                    tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            
            target += _target
        
        assert len(input_id) == len(target)
        
        # Padding
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class SupervisedDataset(Dataset):
    """自定义监督微调数据集"""
    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 初始化SwanLab
    swanlab.init(project="sft", experiment_name="qwen-1.5b")
    
    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # 设置日志级别
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # 记录训练信息
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # 检查checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"输出路径 ({training_args.output_dir}) 非空 ")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"从 {last_checkpoint}恢复训练")
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 加载预训练模型
    logger.warning("加载预训练模型")
    logger.info(f"模型参数地址：{model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")
    
    # 初始化Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    # 确保有pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    logger.info("完成 tokenizer 加载")
    
    # 加载微调数据
    with open(data_args.train_files, 'r') as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{data_args.train_files}")
    logger.info(f'训练样本总数:{len(raw_data)}')
    
    # 创建训练数据集
    train_dataset = SupervisedDataset(
        raw_data=raw_data,
        tokenizer=tokenizer,
        max_len=data_args.max_seq_length
    )
    
    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=IterableWrapper(train_dataset),
        tokenizer=tokenizer
    )
    
    # 从 checkpoint 加载
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    logger.info("开始训练")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存模型
    logger.info("保存模型")
    trainer.save_model()
    trainer.save_state()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

if __name__ == "__main__":
    main()