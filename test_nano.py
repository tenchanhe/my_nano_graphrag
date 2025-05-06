from evaluation.data_loader import load_data_in_batches, read_html
from evaluation.evaluation import evaluate_predictions
from evaluation.kg_utils import kg_insert, kg_query
from evaluation.query_list import query_dict
from custom_codes.config_setting import query_param

# MODEL = "llama3.3"
# # WORKING_DIR = "./nano_llama3.3_cache_ollama"
# WORKING_DIR = "./nano2_llama3.3_cache_ollama"

# MODEL = "llama3.2"
MODEL = "phi4"
# WORKING_DIR = "./nano_nccu_cache"
WORKING_DIR = "./nano_xiaoming_cache"

# MODEL = "qwen2.5:32b"
# WORKING_DIR = "./nano_xiyouji_cache_ollama"

if __name__ == "__main__":
    from time import time

    # with open("./kg_input/mock_data.txt", encoding="utf-8-sig") as f:
    # with open("./test.html", encoding="utf-8-sig") as f:
    with open("./kg_input/xiaoming.txt", encoding="utf-8-sig") as f:
        TEXT = f.read()
    # print(FAKE_TEXT)
    # TEXT = read_html(TEXT)

    start = time()
    kg_insert(MODEL, TEXT, WORKING_DIR)
    print("indexing time:", time() - start)

    # kg_query(MODEL, "", WORKING_DIR, query_param)
