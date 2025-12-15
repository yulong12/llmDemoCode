import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import argparse
import os
# 在导入其他库之前设置TOKENIZERS_PARALLELISM环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import math
from contextlib import nullcontext
import time
import swanlab
from model import FangQi, ModelConfig  # 假设模型定义在k_model.py中


def Logger(info):
    """日志打印函数"""
    print(info)


class PretrainDataset(Dataset):
    """预训练数据集类"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集
        
        Args:
            data_path (str): 训练数据文件路径
            tokenizer: 分词器对象
            max_length (int): 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 加载训练数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (input_ids, target_ids, loss_mask)
        """
        text = self.data[idx]['text'] if 'text' in self.data[idx] else str(self.data[idx])
        
        # 对文本进行编码
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        
        # 如果序列长度超过最大长度，则截断
        if len(encoded) > self.max_length - 1:
            encoded = encoded[:self.max_length - 1]
        
        # 添加EOS token
        input_ids = encoded + [self.tokenizer.eos_token_id]
        
        # 创建目标序列（输入序列向左移一位）
        target_ids = input_ids[1:] + [self.tokenizer.pad_token_id]
        
        # 创建损失掩码（padding位置为0，其他位置为1）
        loss_mask = [1] * len(input_ids)
        
        # 填充到最大长度
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.pad_token_id)
            target_ids.append(self.tokenizer.pad_token_id)
            loss_mask.append(0)
        
        # 转换为tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.float)
        
        return input_ids, target_ids, loss_mask


def get_lr(it, all):
    """
    计算当前迭代的学习率，使用余弦退火调度策略
    
    学习率调度策略：
    1. Warmup阶段：学习率从0线性增长到目标学习率
    2. 余弦退火阶段：学习率按余弦函数衰减到最小学习率
    3. 超出训练步数后：保持最小学习率
    
    Args:
        it (int): 当前迭代步数
        all (int): 总迭代步数
        
    Returns:
        float: 当前步数对应的学习率
    """
    warmup_iters = args.warmup_iters  # 预热迭代次数
    lr_decay_iters = all  # 学习率衰减的总迭代次数
    min_lr = args.learning_rate / 10  # 最小学习率，为初始学习率的1/10

    # Warmup阶段：线性增长
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 超出训练步数：保持最小学习率
    if it > lr_decay_iters:
        return min_lr
    
    # 余弦退火阶段
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 余弦系数
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch):
    """
    训练一个epoch的函数
    
    实现了完整的训练循环，包括：
    1. 数据加载和设备转移
    2. 动态学习率调整
    3. 前向传播和损失计算
    4. 梯度累积和反向传播
    5. 梯度裁剪和优化器更新
    6. 日志记录和模型保存
    
    Args:
        epoch (int): 当前epoch编号
    """
    start_time = time.time()  # 记录开始时间
    
    # 遍历数据加载器中的每个batch
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据转移到指定设备（GPU/CPU/MPS）
        X = X.to(args.device)  # 输入序列
        Y = Y.to(args.device)  # 目标序列
        loss_mask = loss_mask.to(args.device)  # 损失掩码，用于忽略padding token

        # 计算当前步骤的学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        # 更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度训练上下文
        with ctx:
            # 前向传播
            out = model(X, Y)
            # 计算损失并除以累积步数（用于梯度累积）
            loss = out.last_loss / args.accumulation_steps
            # 将loss_mask展平为一维
            loss_mask = loss_mask.view(-1)
            # 应用掩码计算有效损失（忽略padding位置）
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # 使用scaler进行混合精度的反向传播
        # 根据设备类型选择不同的缩放方式
        if device_type == "cuda":
            scaler.scale(loss).backward()
        elif device_type == "mps":
            # MPS不支持GradScaler，手动缩放
            loss.backward()
        else:  # CPU
            scaler.scale(loss).backward()

        # 每accumulation_steps步执行一次优化器更新
        if (step + 1) % args.accumulation_steps == 0:
            # 取消梯度缩放，准备梯度裁剪
            if device_type == "cuda":
                scaler.unscale_(optimizer)
            elif device_type == "mps":
                # MPS不支持unscale_，直接进行梯度裁剪
                pass
            else:  # CPU
                scaler.unscale_(optimizer)
                
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器步骤
            if device_type == "cuda":
                scaler.step(optimizer)
                # 更新scaler的缩放因子
                scaler.update()
            elif device_type == "mps":
                # MPS不支持GradScaler，直接更新优化器
                optimizer.step()
            else:  # CPU
                scaler.step(optimizer)
                # 更新scaler的缩放因子
                scaler.update()

            # 清零梯度，set_to_none=True可以节省内存
            optimizer.zero_grad(set_to_none=True)

        # 每log_interval步记录一次日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 打印训练进度信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复真实的loss值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            
            # 如果启用SwanLab，记录训练指标
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # 每save_interval步保存一次模型
        if (step + 1) % args.save_interval == 0:
            model.eval()  # 切换到评估模式
            # 构建检查点文件名
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'

            # 处理多卡保存：如果是DataParallel模型，需要访问.module属性
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()  # 切换回训练模式
        
        # 每20000步保存一个带步数标记的检查点
        if (step + 1) % 50 == 0:
            model.eval()
            # 构建带步数的检查点文件名
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step+1}.pth'

            # 保存模型状态字典
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """
    初始化模型和分词器
    
    功能包括：
    1. 加载预训练的分词器
    2. 创建Transformer模型
    3. 设置多GPU并行训练（如果可用）
    4. 将模型移动到指定设备
    5. 统计并打印模型参数量
    
    Returns:
        tuple: (model, tokenizer) 初始化后的模型和分词器
    """
    def count_parameters(model):
        """
        统计模型中可训练参数的数量
        
        Args:
            model: PyTorch模型
            
        Returns:
            int: 可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 从本地路径加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # 根据配置创建Transformer模型
    model = FangQi(lm_config)
    
    # 多卡初始化：检查可用GPU数量并设置DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and args.device.startswith("cuda"):
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        # 使用DataParallel包装模型以支持多GPU训练
        model = torch.nn.DataParallel(model)
    
    # 将模型移动到指定设备（GPU/CPU/MPS）
    model = model.to(args.device)
    
    # 计算并打印模型参数量（以百万为单位）
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    
    # 基础训练参数
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="auto", help="训练设备 (auto/cuda/mps/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    
    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, default="./seq_monkey_datawhale.jsonl", help="训练数据路径")
    
    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")
    
    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # 多GPU训练参数
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="使用的GPU ID，用逗号分隔 (例如: '0,1,2')")

    args = parser.parse_args()

    # ==================== 设备检测与设置 ====================
    # 自动检测最佳可用设备
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
            Logger("Using Apple Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            # 如果指定了GPU，使用第一个GPU
            if args.gpus is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
                args.device = "cuda:0"
            else:
                args.device = "cuda:0"
            Logger(f"Using CUDA device: {args.device}")
        else:
            args.device = "cpu"
            Logger("Using CPU")
    else:
        # 用户明确指定设备
        if args.device.startswith("cuda"):
            if args.gpus is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
                # 重新设置device为cuda:0（映射后的第一个GPU）
                args.device = "cuda:0"
        elif args.device == "mps" and not torch.backends.mps.is_available():
            Logger("MPS not available, falling back to CPU")
            args.device = "cpu"
        elif args.device == "mps":
            Logger("Using Apple Metal Performance Shaders (MPS)")

    # ==================== 实验跟踪初始化 ====================
    if args.use_swanlab:
        # 注意：使用前需要先登录 swanlab.login(api_key='your key')
        run = swanlab.init(
            project="Happy-LLM",  # 项目名称
            experiment_name=f"Pretrain-215M-{args.device}",  # 包含设备信息的实验名称
            config=args,  # 保存所有超参数
        )

    # ==================== 模型配置 ====================
    # 定义语言模型的配置参数
    lm_config = ModelConfig(
        dim=1024,      # 模型维度
        n_layers=18,   # Transformer层数
    )

    # ==================== 训练环境设置 ====================
    max_seq_len = lm_config.max_seq_len  # 最大序列长度
    args.save_dir = os.path.join(args.out_dir)  # 模型保存目录
    
    # 创建必要的目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 设置随机种子以确保结果可复现
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

    # ==================== 模型和数据初始化 ====================
    # 初始化模型和分词器
    model, tokenizer = init_model()
    
    # 创建训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    
    # 根据设备类型调整数据加载器参数
    pin_memory = (device_type == "cuda")  # GPU训练时使用pin_memory
    if device_type == "mps":
        # MPS训练时也设置pin_memory为False，因为MPS有自己的内存管理
        pin_memory = False
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  # 批次大小
        pin_memory=pin_memory,       # GPU训练时将数据加载到固定内存中，加速GPU传输
        drop_last=False,             # 不丢弃最后一个不完整的批次
        shuffle=True,                # 随机打乱数据
        num_workers=args.num_workers # 数据加载的并行工作进程数
    )

    # ==================== 优化器和训练组件初始化 ====================
    # 根据设备类型初始化混合精度训练的梯度缩放器
    # MPS不支持GradScaler，只有CUDA和CPU支持
    if device_type == "cuda":
        # 只有在使用float16或bfloat16时才启用
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    elif device_type == "mps":
        # MPS不支持GradScaler，设置为None，但为了代码兼容性，创建一个虚拟的
        class DummyScaler:
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                pass
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        scaler = DummyScaler()
    else:  # CPU
        # CPU训练时只在使用float16时启用
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16'])) if hasattr(torch.cuda, 'amp') else torch.cpu.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # 初始化Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==================== 开始训练 ====================
    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    
    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch)