import os
import json
from evaluation.kg_utils import kg_insert, kg_query
from evaluation.evaluation import evaluate_predictions
from custom_codes.config_setting import query_param


MODEL = "phi4"
# EVAL_MODEL = "phi4"
EVAL_MODEL = "qwen2.5:32b"


def process_jsonl_files(folder_path):
    # 獲取文件夾中所有.jsonl文件
    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"在 {folder_path} 中沒有找到任何.jsonl文件")
        return None
    
    # print(f"在 {folder_path} 中找到 {len(jsonl_files)} 個.jsonl文件:")
    # print("=" * 50)
    
    # 初始化結果字典
    files_top10_data = {}
    
    # 處理每個jsonl文件
    for filename in jsonl_files:
        file_path = os.path.join(folder_path, filename)
        record_count = 0
        top10_records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # 嘗試解析每一行JSON
                    try:
                        record = json.loads(line.strip())
                        record_count += 1
                        
                        # 只收集前10筆資料
                        if record_count <= 10:
                            top10_records.append(record)
                            
                    except json.JSONDecodeError:
                        print(f"警告: {filename} 中發現無效的JSON行 (行號: {record_count + 1})")
                        continue
            
            files_top10_data[filename] = top10_records
            
        except Exception as e:
            print(f"處理文件 {filename} 時出錯: {str(e)}")
            files_top10_data[filename] = []  # 即使出錯也添加空列表
    
    return files_top10_data

if __name__ == "__main__":
    result = process_jsonl_files("./esg_data/UltraDomain/")
    # breakpoint()
    
    queries, ground_truths, predictions = [], [], []

    for filename, records in result.items():
        # print(f"\n文件: {filename}")
        
        "input', 'answers', 'context'"
        for i in range(len(records)):
            working_dir = "kg_cache_ultra/" + filename + "/" + str(i)

            # kg_insert(MODEL, records[i]['context'], working_dir)

            if os.path.exists(working_dir):
                response = kg_query(MODEL, records[i]['input'], working_dir, query_param)

                queries.append(records[i]['input'])
                ground_truths.append(records[i]['answers'][0])
                predictions.append(response)

        # break
        

    evaluation_results, record_list = evaluate_predictions(
        queries, ground_truths, predictions, EVAL_MODEL
    )

    for i in range(len(queries)):
        print(queries[i])
    print()
    for i in range(len(queries)):
        print(ground_truths[i])
    print()
    for i in range(len(queries)):
        print(predictions[i].replace('\n', ''))
    print()
    for i in range(len(queries)):
        print(record_list[i])

    print(evaluation_results)