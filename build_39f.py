import os
from evaluation.kg_utils import kg_insert
from custom_codes.config_setting import query_param


if __name__ == "__main__":

    # MODEL = "llama3.2"
    # WORKING_PATH = "kg_cache_esg_llama"
    MODEL = "qwen2.5:7b"
    WORKING_PATH = "kg_delta_qwen"

    input_dir = "/tmp2/chten/delta/magic_doc/39_files_md/"

    # files = []
    # for filename in os.listdir(input_dir):
    #     with open(input_dir+filename, 'r', encoding="utf-8") as f:
    #         pdf_file = f.read()
    #         files.append(pdf_file)

    with open("../../delta/magic_doc/39_files_md/2023台北植物園_植光計畫.md", 'r', encoding="utf-8") as f:
        pdf_file = f.read()
                
    kg_insert(MODEL, pdf_file, WORKING_PATH)
