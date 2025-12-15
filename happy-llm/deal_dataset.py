import json
from tqdm import tqdm
import os

def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 配置路径
input_file = './data/mobvoi_seq_monkey_general_open_corpus.jsonl'
output_file = './data/seq_monkey_datawhale.jsonl'

# 确保输出目录存在
os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

# 获取文件总行数（用于进度条）
total_lines = 0
with open(input_file, 'rb') as f:  # 用二进制模式安全计数
    for _ in f:
        total_lines += 1

# 安全处理文件
with open(output_file, 'w', encoding='utf-8') as pretrain:  # 使用写模式而非追加
    with open(input_file, 'rb') as f:  # 二进制模式读取
        for line_bytes in tqdm(f, desc=f"Processing {os.path.basename(input_file)}", total=total_lines):
            try:
                # 尝试 UTF-8 解码，失败时使用 surrogateescape 保留原始字节
                line_str = line_bytes.decode('utf-8', errors='surrogateescape')
                # 清理可能存在的控制字符
                line_str = ''.join(c for c in line_str if c.isprintable() or c in ['\n', '\t', ' '])
                line = json.loads(line_str.strip())
                
                # 检查必需字段
                if 'text' not in line or not isinstance(line['text'], str):
                    continue
                    
                # 分块处理
                chunks = split_text(line['text'])
                for chunk in chunks:
                    if len(chunk.strip()) > 10:  # 过滤过短文本
                        pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
                        
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                # 尝试 GBK 解码（常见中文编码）
                try:
                    line_str = line_bytes.decode('gbk', errors='ignore')
                    line = json.loads(line_str.strip())
                    chunks = split_text(line['text'])
                    for chunk in chunks:
                        if len(chunk.strip()) > 10:
                            pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
                except Exception as inner_e:
                    # 记录错误但不中断处理
                    with open('decoding_errors.log', 'a', encoding='utf-8') as err_log:
                        err_log.write(f"Error processing line: {line_bytes}\n")
                        err_log.write(f"Main error: {str(e)}\n")
                        err_log.write(f"GBK fallback error: {str(inner_e)}\n\n")
                    continue

print(f"处理完成! 结果已保存至: {os.path.abspath(output_file)}")
print(f"遇到的解码错误已记录至: {os.path.abspath('decoding_errors.log')}")