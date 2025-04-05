import bz2
import json
from loguru import logger
from collections import defaultdict
from bs4 import BeautifulSoup


def load_data_in_batches(dataset_path, batch_size, 
                         domain_counts={"sports": 10, "movie": 10, "finance": 10, "open": 10, "music": 10}):

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


def read_html(search_results):
    text = ""
    if isinstance(search_results, str):
        soup = BeautifulSoup(search_results, "lxml")
        text += soup.get_text(" ", strip=True)
        return text
    
    for html_text in search_results:
        soup = BeautifulSoup(html_text["page_result"], "lxml")
        text += soup.get_text(" ", strip=True)
        if not text:
            text=""

    return text
