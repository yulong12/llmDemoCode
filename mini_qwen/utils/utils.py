from __future__ import annotations
import os
from itertools import chain
import torch
import time
import gc
from langchain_core.vectorstores import VectorStore
from smolagents import Tool
from typing import List
from langchain_core.vectorstores import VectorStore
from smolagents import Tool
        
class SemanticRetriever(Tool):
    """
    A generic retrieval tool that fetches the *k* most semantically similar
    documents for a given query string.

    Parameters
    ----------
    vectordb : VectorStore
        An initialized vector database instance that implements the
        ``similarity_search`` API.
    top_k : int, optional
        Number of documents to return. Defaults to ``7``.

    Example
    -------
    >>> retriever = SemanticRetriever(vectordb)
    >>> results = retriever("large-language models for code generation")
    """

    name: str = "semantic_retriever"
    description: str = (
        "Return the top-k documents whose embeddings are most similar to the "
        "input query. The query should be phrased affirmatively, not as a question."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The search phrase, expressed in affirmative form and semantically "
                "aligned with the target documents."
            ),
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, *, top_k: int = 7, **kwargs) -> None:
        super().__init__(**kwargs)
        self._db: VectorStore = vectordb
        self._top_k: int = top_k

    def forward(self, query: str) -> str:  # noqa: D401
        """Retrieve and format the most similar documents for *query*."""
        if not isinstance(query, str):
            raise TypeError("`query` must be a string.")

        docs = self._similar_docs(query)

        formatted = [
            f"===== Document {idx} =====\n{doc.page_content}" for idx, doc in enumerate(docs)
        ]
        return "\nRetrieved documents:\n" + "\n".join(formatted)

    def _similar_docs(self, query: str) -> List:  # Return type intentionally generic
        """Return the raw documents from the underlying vector store."""
        return self._db.similarity_search(query, k=self._top_k)

def format_to_r1(example):
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
    # for PPO use
    return {key: [d[key] for d in data] for key in data[0]}

def preprocess_ppo_dataset(examples,tokenizer):
    # for PPO use
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

def preprocess_rm_dataset(examples,tokenizer):
    # for RM use
    # Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
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
    formatted_data = []
    for sample in data:
        problem = sample["problem"]
        generation = sample["generation"]
        
        formatted_data.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": generation}
            ]
        )
    return {"messages": formatted_data}

def formatting_prompts_func_distill(example):
    # for distill use
    output_texts = []
    for i in range(len(example["problem"])):
        human_text = example["problem"][i]
        gpt_text = example["generation"][i]
        text = f"<|im_start|>user\n{human_text}<|im_end|>\n<|im_start|>assistant\n{gpt_text}<|im_end|>"
        output_texts.append(text)
    return output_texts

def formatting_prompts_func(example):
    # for sft use
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

def find_files(dirs,path="data/pt"):
    """
    遍历目录，查找所有文件
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
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def tokenize_dataset(examples,tokenizer,block_size=512):
    """
    预处理预训练数据集，将文本分词并分块
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
    return result,total_length

def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量。
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
    
