# 这段代码的主要作用是批量处理多个 jsonl 格式的文本数据文件，
# 对每行文本进行分句和分词统计，并将结果按 token 数量分组后输出到新文件。
# 适用于大规模语料的预处理，常见于大模型训练数据准备场景。
import os
import json
import logging
from multiprocessing import Pool
from tqdm import tqdm
import util

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(process)d - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("pt_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MAX_TOKENS = 4050
TOKENIZER_PATH = "/mnt/nvme/models/Qwen2.5-72B"
PHYSICAL_CORES = 112  # 服务器物理核心数

def read_jsonl_lines(file_path: str) -> dict:
    """
    按行读取jsonl文件内容，将每行text断句后，返回{句子: token_num}的字典
    """
    result = {}
    tokenizer = util.get_tokenizer(TOKENIZER_PATH)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get('text', '')
                split_text = util.split_sentences(text)
                for sent in split_text:
                    sent = sent.strip()
                    if len(sent) > 0:
                        token_num = len(tokenizer.encode(sent, add_special_tokens=False))
                        result[sent] = token_num
            except Exception as e:
                logger.warning(f"解析失败: {file_path} 行内容: {line[:50]}... 错误: {e}")
    return result

def process_file(args):
    """
    并发处理单个文件，返回目录名、文件路径、内容字典
    """
    file_path, output_root = args
    filename=util.extract_filename(file_path)

    logger.info(f"开始处理文件: {file_path}, 大小: {os.path.getsize(file_path) / 1024**3:.2f}GB")
    filecontent = read_jsonl_lines(file_path)
    return filename, file_path, filecontent

def ensure_dir_exists(path: str):
    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    SOURCE_DIRS = [
        "/mnt/nvme/data/process/industry_corpus_json"
        # "/mnt/nvme/data/process/testdata/aerospace",
        # "/mnt/nvme/data/process/testdata/literature_emotion"
        # "/mnt/nvme/data/process/industry_corpus_json/computer_communication",
        # "/mnt/nvme/data/process/industry_corpus_json/fire_safety_food_safety"
    ]
    OUTPUT_ROOT = "/mnt/nvme/data/process/pt_train"
    ensure_dir_exists(OUTPUT_ROOT)

    # 汇总所有文件
    all_files = []
    for src_dir in SOURCE_DIRS:
        files = util.get_all_jsonl_files(src_dir)
        all_files.extend([(file, OUTPUT_ROOT) for file in files])
    logger.info(f"总共需要处理 {len(all_files)} 个文件")

    # 并发处理所有文件
    results_by_dir = {}
    with Pool(processes=PHYSICAL_CORES) as pool:
        for filename, file_path, filecontent in tqdm(pool.imap_unordered(process_file, all_files), total=len(all_files)):
            if filename not in results_by_dir:
                results_by_dir[filename] = {"all_contents": {}, "total_token_count": 0, "total_input_size": 0}
            file_size = os.path.getsize(file_path) / 1024**3  # GB
            file_token_count = sum(filecontent.values())
            results_by_dir[filename]["all_contents"].update(filecontent)
            results_by_dir[filename]["total_token_count"] += file_token_count
            results_by_dir[filename]["total_input_size"] += file_size
            logger.info(f"分词: {file_path}, 大小: {file_size:.4f}GB, token数: {file_token_count/10000:.2f}万")

    # 每个目录单独输出
    for filename, stats in results_by_dir.items():
        logger.info(f"目录 {filename} ls: {stats['total_input_size']:.4f}GB, 总token数: {stats['total_token_count']/10000:.2f}万")
        res = util.split_contents_by_token_limit(stats["all_contents"], max_token=MAX_TOKENS)
        logger.info(f"目录 {filename} 拼接后共{len(res)}条")
        output_file = os.path.join(OUTPUT_ROOT, f"{filename}.json")
        util.save_dict_to_file(res, output_file)
        output_size_gb = os.path.getsize(output_file) / 1024**3
        output_token_count = sum(item.get("token", 0) for item in res)
        logger.info(f"已保存到 {output_file}")
        logger.info(f"输出文件{filename}.json 大小: {output_size_gb:.4f}GB, token总数: {output_token_count/10000:.2f}万")