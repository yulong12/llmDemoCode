import torch
import wandb
from trl import SFTConfig,SFTTrainer
from functools import partial
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from utils.utils import find_files,format_data_chartqa,collate_func,clear_memory
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

MODEL_APTH = "/mnt/nvme/test/model/qwen7vl"
DATA_PATH = "/mnt/nvme/test/data"
TMP_PATH = "/mnt/nvme/test/tmp"
OUTPUT_PATH = "/mnt/nvme/test/result"
SUBSET = -1

directories = ['ChatQA']
data_files = find_files(directories,DATA_PATH)
dataset = load_dataset("parquet", data_files=data_files, split='train', cache_dir=TMP_PATH) 
if SUBSET > 0:
    train_dataset = dataset.select(range(SUBSET))

train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()
train_dataset, eval_dataset = train_val_dataset.train_test_split(test_size=0.1, seed=42).values()

train_dataset = [format_data_chartqa(sample) for sample in train_dataset]
eval_dataset = [format_data_chartqa(sample) for sample in eval_dataset]
test_dataset = [format_data_chartqa(sample) for sample in test_dataset]

clear_memory()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_APTH,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(MODEL_APTH)
collate_fn = partial(collate_func, processor=processor)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=4,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


training_args = SFTConfig(
    # 训练过程中保存模型和检查点的目录。
    output_dir=OUTPUT_PATH,  
    # 训练数据将被完整遍历4次（4个epoch）。
    num_train_epochs=4,  
    # 每个设备（如每张显卡）上的训练 batch 大小为1。
    per_device_train_batch_size=1,  
    # 每个设备上的评估 batch 大小为1。
    per_device_eval_batch_size=1,  
    # 梯度累计16步再进行一次反向传播和参数更新，相当于实际 batch size = 1×16=16。
    gradient_accumulation_steps=16,
    # 启用梯度检查点，节省显存，适合大模型训练。
    gradient_checkpointing=True,  
    # 使用高效的 AdamW 优化器（fused 版本，速度快，显存占用低）。
    optim="adamw_torch_fused", 
    # 初始学习率为0.0002
    learning_rate=2e-4,  
    # 学习率调度策略为常数（训练过程中学习率不变）。
    lr_scheduler_type="constant",

    # Logging and evaluation
    # 每训练5步记录一次日志（如 loss）
    logging_steps=5, 
    # 每500步进行一次评估。
    eval_steps=500,  
    # 按步数（steps）进行评估
    eval_strategy="steps",  
    # 按步数（steps）保存模型。
    save_strategy="steps",  
    # 每50步保存一次模型检查点。
    save_steps=50,  
    # 以评估损失（eval_loss）作为选择最佳模型的指标。
    metric_for_best_model="eval_loss",  
    # Mixed precision and gradient settings
    # 启用 bfloat16 混合精度训练，节省显存、加速训练（需硬件支持）
    bf16=True,  
    # 梯度裁剪的最大范数，防止梯度爆炸。
    max_grad_norm=1.0,  
    # 学习率预热比例，训练初期逐步增大学习率，防止模型不稳定。
    warmup_ratio=0.03, 

    # 日志和指标同步到 Weights & Biases 平台，便于可视化和实验管理。
    report_to="wandb",  
    # Gradient checkpointing settings
    # 梯度检查点的额外参数，use_reentrant=False 可提升兼容性和稳定性
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Dataset configuration
    # 指定数据集中用于文本输入的字段名（为空时需自定义 collate_fn）。
    dataset_text_field="",  # Text field in dataset
    # 跳过数据集的预处理步骤，通常用于自定义数据处理流程。
    dataset_kwargs={"skip_prepare_dataset": True},
    # 输入序列的最大长度，超出部分会被截断。
    max_seq_length=800,  # Maximum sequence length for input
    # 不移除数据集中未用到的列，保留全部信息，便于自定义处理。
    remove_unused_columns = False
)

wandb.init(
    project="qwen25vl-sft-ChartQA-new",
    name="qwen25vl-sft-ChartQA-new",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

trainer.train()
trainer.save_model()