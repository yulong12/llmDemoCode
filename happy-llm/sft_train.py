import os
import platform
import argparse
import time
import warnings
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer
from model import ModelConfig, FangQi
import json
import numpy as np
from torch.utils.data import Dataset
import swanlab

# 忽略警告
warnings.filterwarnings('ignore')

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def generate_loss_mask(self, input_ids):
        # 生成 loss mask, 0 表示不计算损失, 1 表示计算损失
        mask = [0] * len(input_ids)
        # 查找所有可能的assistant标记位置
        assistant_token_ids = [3, 1074, 537, 500, 203]  # "assistant\n"
        eos_token_id = 4  # EOS token
        
        # 简化方法：直接从最后一个assistant标记后开始计算损失
        last_assistant_pos = -1
        for i in range(len(input_ids) - len(assistant_token_ids)):
            match = True
            for j, token_id in enumerate(assistant_token_ids):
                if input_ids[i + j] != token_id:
                    match = False
                    break
            if match:
                last_assistant_pos = i + len(assistant_token_ids)
        
        # 如果找到了assistant标记，则从该位置到最后一个非填充token计算损失
        if last_assistant_pos != -1:
            # 找到最后一个非填充token的位置
            last_non_padding = len(input_ids) - 1
            for i in range(len(input_ids) - 1, -1, -1):
                if input_ids[i] != self.padding:
                    last_non_padding = i
                    break
            
            # 标记从assistant之后到末尾的区域计算损失
            for i in range(last_assistant_pos, min(last_non_padding + 1, len(mask))):
                mask[i] = 1
        else:
            # 如果没有找到assistant标记，回退到简单策略：计算最后20% tokens的损失
            start_pos = int(len(input_ids) * 0.8)
            for i in range(start_pos, len(mask)):
                mask[i] = 1
                
        return mask

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

def Logger(content):
    """日志记录器"""
    print(content)

def get_lr(it, all):
    """获取学习率"""
    # 1) linear warmup for warmup_iters steps
    # 1) 预热迭代的线性预热
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 2) if it > lr_decay_iters, return min learning rate
    # 2) 如果迭代次数超过学习率衰减迭代次数，则返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    # 3) 在两者之间，使用余弦衰减至最小学习率
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """训练一个epoch"""
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 获取学习率并更新优化器
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            # 添加检查确保不会除以零
            mask_sum = loss_mask.sum()
            if mask_sum > 0:
                loss = torch.sum(loss * loss_mask) / mask_sum
            else:
                # 如果mask_sum为0，使用默认损失计算
                loss = loss.mean()

        # 反向传播
        # 根据是否使用AMP（自动混合精度）来决定如何执行反向传播
        if scaler is not None:
            # 使用梯度缩放（仅适用于CUDA设备）
            scaler.scale(loss).backward()
        else:
            # 不使用梯度缩放（适用于MPS和CPU设备）
            loss.backward()

        # 更新权重
        if (step + 1) % args.accumulation_steps == 0:
            if scaler is not None:
                # 使用梯度缩放时的操作
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 不使用梯度缩放时的操作
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 确保损失值有效后再记录
            loss_val = loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else 0.0
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss_val * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss_val * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # 保存模型
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth'

            # 处理多卡保存
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
        
        # 定期保存模型
        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}_step{step+1}.pth'

            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """初始化模型"""
    def count_parameters(model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # 初始化模型
    model = FangQi(lm_config)

    # 加载预训练权重
    ckp = './my_model_output/pretrain_1024_18_6144.pth'
    # MPS设备需要特殊处理，因为某些权重可能不兼容
    if device_type == "mps":
        # MPS设备加载时使用cpu作为中间设备
        state_dict = torch.load(ckp, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(ckp, map_location=args.device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    
    # 多卡初始化
    if device_type == "cuda":
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            Logger(f"Using {num_gpus} GPUs with DataParallel!")
            model = torch.nn.DataParallel(model)
    elif device_type == "mps":
        # MPS设备不支持DataParallel，只使用单卡
        Logger("Using MPS device (single GPU only)")
    else:
        # CPU设备也不支持DataParallel
        Logger("Using CPU device")
    
    model = model.to(args.device)
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    parser.add_argument("--out_dir", type=str, default="sft_model_215M", help="输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="使用的设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, default="./BelleGroup_sft.jsonl", help="训练数据路径")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="预热迭代次数")
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    # 添加多卡参数
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="逗号分隔的GPU ID (例如 '0,1,2')")

    args = parser.parse_args()

    # 设置可见GPU
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # 自动设置主设备为第一个GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            args.device = "mps"
        else:
            args.device = "cpu"

    # 初始化swanlab
    if args.use_swanlab:
        run = swanlab.init(
            project="Happy-LLM",
            experiment_name="SFT-215M",
            config=args,
        )

    # 模型配置
    lm_config = ModelConfig(
        dim=1024,
        n_layers=18,
    )
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    
    # 确定设备类型（用于选择合适的上下文管理器）
    if args.device.startswith("cuda"):
        device_type = "cuda"
    elif args.device == "mps":
        device_type = "mps"
    else:
        device_type = "cpu"

    # 设置混合精度训练的上下文管理器
    # CPU和MPS训练时使用nullcontext，GPU训练时使用autocast
    if device_type == "cuda":
        ctx = torch.cuda.amp.autocast(dtype=getattr(torch, args.dtype))
    elif device_type == "mps":
        # MPS不支持autocast，使用float32
        ctx = nullcontext()
    else:  # CPU
        ctx = nullcontext() if args.dtype == "float32" else torch.cpu.amp.autocast(dtype=getattr(torch, args.dtype))

    # 初始化模型和分词器
    model, tokenizer = init_model()
    
    # 创建数据集和数据加载器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    
    # 根据设备类型调整数据加载器参数
    pin_memory = (device_type == "cuda")  # GPU训练时使用pin_memory
    if device_type == "mps":
        # MPS训练时不使用pin_memory，因为MPS有自己的内存管理
        pin_memory = False
        # MPS训练时将num_workers设置为0，避免多进程问题
        num_workers = 0
    else:
        num_workers = args.num_workers
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=True,
        num_workers=num_workers
    )

    # 缩放器和优化器
    # 根据设备类型初始化混合精度训练的梯度缩放器
    # MPS不支持GradScaler，只有CUDA和CPU支持
    if device_type == "cuda":
        # 只有在使用float16或bfloat16时才启用
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    elif device_type == "mps":
        # MPS不支持GradScaler，设置为None
        scaler = None
    else:  # CPU
        # CPU支持GradScaler，但通常不需要
        scaler = torch.cpu.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16'])) if hasattr(torch.cpu.amp, 'GradScaler') else None
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 开始训练
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)