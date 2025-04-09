from evaluation.evaluation import evaluate_predictions
from evaluation.kg_utils import kg_insert, kg_query
from custom_codes.config_setting import query_param


if __name__ == "__main__":

    MODEL = "llama3.2"
    # MODEL = "qwen2.5:32b"
    WORKING_PATH = "kg_cache_esg"
    
    with open("kg_input/Set-of-GRI-Stnds-2021.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()
    # print(markdown_content)
                
    kg_insert(MODEL, markdown_content, WORKING_PATH)
