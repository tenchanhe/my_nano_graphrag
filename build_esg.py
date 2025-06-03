from evaluation.kg_utils import kg_insert
from custom_codes.config_setting import query_param


if __name__ == "__main__":

    # MODEL = "llama3.2"
    # WORKING_PATH = "kg_cache_esg_llama"
    MODEL = "phi4"
    WORKING_PATH = "kg_cache_esg_phi"
    
    with open("kg_input/Set-of-GRI-Stnds-2021.md", "r", encoding="utf-8") as f:
        markdown_content = f.read()
    # print(markdown_content)
                
    kg_insert(MODEL, markdown_content, WORKING_PATH)
