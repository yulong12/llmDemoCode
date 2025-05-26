# 脚本实现了高效并发地将大批量 Parquet 文件转换为 JSONL 文件的功能，并且自动按目录结构输出，适合在多核服务器上批量数据转换。
# 该脚本使用了多进程和多线程的组合方式来提高处理速度，并且在处理过程中记录日志以便于后续的调试和监控。
# 该脚本的主要功能包括：
# 1. 递归查找指定目录下的所有 Parquet 文件。
# 2. 将每个 Parquet 文件转换为 JSONL 格式，并按行业、语言和等级信息输出到相应的目录。
# 3. 使用多进程和多线程的组合方式来提高处理速度。
# 4. 处理过程中记录日志，包括开始处理、成功转换和失败的文件信息。
# 5. 支持按需调整进程和线程数，以适应不同的服务器配置。
# 6. 处理过程中捕获异常并记录错误信息，确保脚本的健壮性。
# 7. 支持在处理过程中输出当前的 CPU 插槽数，以便于监控和调试。
# 8. 支持在处理过程中输出当前的物理核心数，以便于监控和调试。
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import pyarrow.parquet as pq
import psutil
# 配置日志系统，日志同时输出到终端和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("dataset_conversion1.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_parquet(input_path, output_base):
    """
    处理单个parquet文件，将其内容转换为jsonl格式并写入输出目录。
    """
    try:
        # 解析目录结构，获取行业、语言、等级信息
        path_parts = input_path.split(os.sep)
        industry = path_parts[-4]
        lang = path_parts[-3]
        level = path_parts[-2]
        # 构建输出目录
        output_dir = os.path.join(output_base, industry, lang, level)
        os.makedirs(output_dir, exist_ok=True)
        # 构建输出文件路径
        output_path = os.path.join(
            output_dir,
            os.path.basename(input_path).replace(".parquet", ".jsonl")
        )
        logger.info(f"Start processing: {input_path}")
        # 读取parquet文件并分批写入jsonl
        parquet_file = pq.ParquetFile(input_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for batch_idx, record_batch in enumerate(parquet_file.iter_batches(batch_size=10000)):
                df = record_batch.to_pandas()
                for _, row in df.iterrows():
                    json_line = json.dumps(row.to_dict(), ensure_ascii=False)
                    f.write(json_line + '\n')
                logger.debug(f"{input_path} - Batch {batch_idx+1} processed ({len(df)} records)")
        logger.info(f"Successfully converted: {input_path} -> {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {str(e)}", exc_info=True)
        return False

def worker(file_list, output_base, thread_num=4):
    """
    每个进程的工作函数，内部用线程池并发处理分配到的parquet文件列表。
    """
    success_count = 0
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(process_parquet, f, output_base) for f in file_list]
        for future in as_completed(futures):
            if future.result():
                success_count += 1
    return success_count

def find_parquet_files(root_dir):
    """
    递归查找指定目录下所有parquet文件，返回文件路径列表。
    """
    parquet_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".parquet"):
                full_path = os.path.join(dirpath, fname)
                parquet_files.append(full_path)
    return parquet_files
def get_cpu_sockets():
    sockets = set()
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("physical id"):
                sockets.add(line.strip().split(":")[1])
    return len(sockets)
print("CPU sockets:", get_cpu_sockets())
def main():
    """
    主函数：查找所有parquet文件，分配到多个进程，每个进程内部再用线程池并发处理。
    """
    input_root = "/mnt/nvme/data/pt/IndustryCorpus2"
    output_root = "./industry_corpus_json1"
    logger.info("Scanning for parquet files...")
    parquet_files = find_parquet_files(input_root)
    logger.info(f"Found {len(parquet_files)} parquet files to process")

    # 进程和线程数可根据机器配置调整
    # process_num = min(cpu_count(), 16)  # 最多16进程
    physical_cores = psutil.cpu_count(logical=False)  # 物理核心数
    cpu_socket=get_cpu_sockets()
    thread_num = 3  # 每进程4线程
    process_num = min(cpu_count(), cpu_socket * physical_cores)  
    logger.info(f"physical_cores {physical_cores} , cpu_socket {cpu_socket},  thread_num {thread_num}, process_num {process_num}")

    chunk_size = (len(parquet_files) + process_num - 1) // process_num
    # 将文件列表均分给每个进程
    tasks = [parquet_files[i*chunk_size:(i+1)*chunk_size] for i in range(process_num)]

    logger.info(f"Using {process_num} processes, each with {thread_num} threads")

    # 多进程并发，每个进程内部再用线程池
    with Pool(processes=process_num) as pool:
        results = [pool.apply_async(worker, (task, output_root, thread_num)) for task in tasks]
        total_success = sum(r.get() for r in results)

    logger.info(f"Processing completed. Success: {total_success}/{len(parquet_files)}")

if __name__ == "__main__":
    main()