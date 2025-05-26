import os
from itertools import chain
import torch
import time
import gc


def format_to_r1(example):
    """
    功能: 将单条样本格式化为R1风格的prompt，包含系统提示和用户问题。
    输入: example(dict)，包含'problem'字段。
    输出: dict，prompt格式适用于对话式微调。
    """
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


def collator_ppo(data):
    """
    功能: PPO训练用的collate函数，将batch中的每个字段聚合成列表。
    输入: data(list of dict)，每个dict为一个样本。
    输出: dict，每个key对应一个batch的列表。
    """
    return {key: [d[key] for d in data] for key in data[0]}


def preprocess_ppo_dataset(examples, tokenizer):
    """
    功能: PPO训练用数据预处理，将每个问题格式化为prompt并分词。
    输入: examples(dict)，包含'question'字段；tokenizer(分词器)。
    输出: dict，包含'query'和'input_ids'列表。
    """
    new_examples = {
        "query": [],
        "input_ids": [],
    }
    for question in examples["question"]:
        query = "Question: " + question + "\n\nAnswer: "
        tokenized_question = tokenizer(query, truncation=True)
        new_examples["query"].append(query)
        new_examples["input_ids"].append(tokenized_question["input_ids"])
    return new_examples


def preprocess_rm_dataset(examples, tokenizer):
    """
    功能: RM（Reward Model）训练用数据预处理，将每个样本转为正/负对，分别分词。
    输入: examples(dict)，包含'question', 'response_j', 'response_k'字段；tokenizer(分词器)
    输出: dict，包含input_ids和attention_mask的正负对。
    """
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
        tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j, truncation=True)
        tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k, truncation=True)

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples


def format_to_chatml(data):
    """
    功能: 将数据格式化为ChatML风格，适用于对话模型。
    输入: data(list of dict)，每个dict包含'problem'和'generation'字段。
    输出: dict，包含格式化后的消息列表。
    """
    formatted_data = []
    for sample in data:
        problem = sample["problem"]
        generation = sample["generation"]

        formatted_data.append(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": generation},
            ]
        )
    return {"messages": formatted_data}


def formatting_prompts_func_distill(example):
    """
    功能: 用于知识蒸馏的数据格式化，将每组问答转为ChatML格式的字符串。
    输入: example(dict)，包含'problem'和'generation'列表。
    输出: list，格式化后的字符串列表。
    """
    output_texts = []
    for i in range(len(example["problem"])):
        human_text = example["problem"][i]
        gpt_text = example["generation"][i]
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts


def formatting_prompts_func(example):
    """
    功能: 用于SFT训练的数据格式化，将多轮对话转为ChatML格式的字符串。
    输入: example(dict)，包含'conversations'（多轮对话列表）。
    输出: list，格式化后的字符串列表。
    """
    output_texts = []
    for i in range(len(example["conversations"])):
        for item in example["conversations"][i]:
            if item["from"] == "human":
                human_text = item["value"]
            elif item["from"] == "gpt":
                gpt_text = item["value"]
            else:
                raise ValueError(f"Unknown sender: {item['from']}")
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts


def find_files(dirs, path="data/pt"):
    """
    功能: 遍历指定目录，查找所有以.parquet结尾的文件，返回文件路径列表。
    输入: dirs(list of str)，要查找的子目录名列表；path(str)，根目录。
    输出: files(list of str)，所有找到的parquet文件路径。
    """
    files = []
    for dir in dirs:
        base_path = os.path.join(path, dir)
        for dirpath, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(".parquet"):
                    full_path = os.path.join(dirpath, filename)
                    files.append(full_path)
    return files


def clear_memory():
    """
    功能: 清理全局变量、Python垃圾回收和CUDA显存，释放GPU资源。
    输入: 无
    输出: 无（打印显存信息）
    """
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def tokenize_dataset(examples, tokenizer, block_size=512):
    """
    功能: 预处理预训练数据集，将文本加eos后分词并拼接，再按block_size切分成块。   
    输入: examples(dict)，包含'text'字段；tokenizer(分词器)；block_size(int)，分块长度。   
    输出: result(dict)，分块后的token id；total_length(int)，总token数。   
    """
    eos_token = "<|im_end|>"
    text_examples = [text + eos_token for text in examples["text"]]  # 添加结束符
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)

    concatenated_examples = {
        k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()
    }
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    total_length = (total_length // block_size) * block_size  # 对齐块大小

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result, total_length


def print_trainable_parameters(model):
    """
    功能: 打印模型中可训练参数的数量、总参数量及可训练参数比例。
    输入: model(PyTorch模型)
    输出: 无（打印参数信息）
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

