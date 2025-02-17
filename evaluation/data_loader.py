import bz2
import json
from loguru import logger
from collections import defaultdict
from bs4 import BeautifulSoup


def load_data_in_batches(dataset_path, batch_size, 
                         domain_counts={"sports": 10, "movie": 10, "finance": 10, "open": 10, "music": 10}):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.
    Additionally, it allows limiting the number of items per domain.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.
    domain_counts (dict): A dictionary specifying the maximum number of items per domain.
                         Example: {"sports": 100, "movie": 50, "finance": 200, "open": 300, "music": 150}

    Yields:
    dict: A batch of data.
    """
    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    # 初始化 domain 計數器
    if domain_counts is None:
        domain_counts = {}
    domain_counter = defaultdict(int)

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    domain = item.get("domain")  # 假設每筆資料有一個 "domain" 欄位

                    # 檢查 domain 是否在 domain_counts 中，並且是否已達到最大數量
                    if domain in domain_counts and domain_counter[domain] >= domain_counts[domain]:
                        continue  # 跳過該筆資料

                    # 將資料加入 batch
                    for key in batch:
                        batch[key].append(item[key])

                    # 更新 domain 計數器
                    if domain:
                        domain_counter[domain] += 1

                    # 如果 batch 達到指定大小，回傳 batch
                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()

                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")

            # 回傳剩餘的資料作為最後一個 batch
            if batch["query"]:
                yield batch

    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


def read_html(batch_search_results):
    text = ""
    for idx, search_results in enumerate(batch_search_results):     
        for html_text in search_results:
            soup = BeautifulSoup(html_text["page_result"], "lxml")
            text += soup.get_text(" ", strip=True)
            if not text:
                text=""
    
    return text


def process_bz2_file(file_path):
    """逐行處理 .bz2 檔案"""
    domain_counter = defaultdict(int)
    first_record_keys = None
    num_records = 0

    with bz2.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            num_records += 1

            # 記錄第一筆資料的 key
            if first_record_keys is None:
                first_record_keys = list(record.keys())

            # 統計 domain 欄位
            if "domain" in record:
                domain = record["domain"]
                domain_counter[domain] += 1

    return first_record_keys, num_records, domain_counter

def main():
    # 檔案路徑
    file_path = "data/crag_task_1_dev_v4_release.jsonl.bz2"  # 替換為你的 .bz2 檔案路徑

    # 處理檔案
    first_record_keys, num_records, domain_counter = process_bz2_file(file_path)

    # 顯示第一筆資料的 key
    if first_record_keys:
        print("第一筆資料的 key:", first_record_keys)

    # 統計資料筆數
    print(f"資料總筆數: {num_records}")

    # 輸出 domain 統計結果
    print("\ndomain 欄位的值出現次數:")
    for domain, count in domain_counter.items():
        print(f"{domain}: {count} 次")

if __name__ == "__main__":
    main()