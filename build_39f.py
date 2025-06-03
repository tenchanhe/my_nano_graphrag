import os
from evaluation.kg_utils import kg_insert, kg_query
from custom_codes.config_setting import query_param


if __name__ == "__main__":

    # MODEL = "llama3.2"
    # WORKING_PATH = "kg_cache_esg_llama"
    MODEL = "qwen2.5:7b"
    # MODEL = "phi4"
    # WORKING_PATH = "kg_delta_qwen"
    WORKING_PATH = "kg_delta_qwen2"

    input_dir = "/tmp2/chten/delta/my_magic_doc/39_files_md/"

    # # files = []
    # for filename in os.listdir(input_dir):
    #     print(filename)
    #     with open(input_dir+filename, 'r', encoding="utf-8") as f:
    #         pdf_file = f.read()
    #         # files.append(pdf_file)
                   
    #     kg_insert(MODEL, pdf_file, WORKING_PATH)
    
    # with open("../../delta/my_magic_doc/39_files_md/2023台北植物園_植光計畫.md", 'r', encoding="utf-8") as f:
        #     pdf_file = f.read()
    
    result = kg_query(MODEL, "請問台達照明解決方案，如何符合LEED認證對室內照明(INTERIOR LIGHTING)要求", WORKING_PATH, query_param)
    print(result)
    result = kg_query(MODEL, "請問台達照明解決方案，符合LEED認證對室內照明(INTERIOR LIGHTING)要求的成功案例為何?", WORKING_PATH, query_param)
    # result = kg_query(MODEL, "LEED 綠建築 「既有建築 解決方案", WORKING_PATH, query_param)

    print(result)
