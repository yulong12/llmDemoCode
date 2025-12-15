import json
from tqdm import tqdm
import os
def convert_message(data):
    """
    将原始数据转换为标准格式
    """
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

def safe_read_line(line_bytes):
    """尝试多种编码解码一行"""
    for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']:
        try:
            return line_bytes.decode(encoding)
        except (UnicodeDecodeError, AttributeError):
            continue
    return None

with open('BelleGroup_sft.jsonl', 'a', encoding='utf-8') as sft:
    with open('./data/BelleGroup/train_3.5M_CN.json', 'rb') as f:  # 以二进制模式打开
        for line_bytes in tqdm(f, desc="Processing", unit="lines"):
            line_str = safe_read_line(line_bytes)
            if line_str is None:
                print("警告：跳过无法解码的行")
                continue
            line_str = line_str.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                message = convert_message(item['conversations'])
                sft.write(json.dumps(message, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print("警告：跳过无效 JSON 行")
                continue