import csv
from custom_codes.config_setting import query_param
from evaluation.kg_utils import kg_query
from evaluation.evaluation import evaluate_predictions


# MODEL = "llama3.2"
# WORKING_PATH = "kg_cache_esg_llama"
MODEL = "phi4"
WORKING_PATH = "kg_cache_esg_phi4"

EVAL_MODEL = "qwen2.5:32b"

if __name__ == "__main__":
    queries, ground_truths, predictions = [], [], []
    with open('esg_data/gri_questions.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        
        # 逐行列印
        next(csv_reader)
        for row in csv_reader:
            # print(row)  # 每行是一個 list
            # breakpoint()
            row[0]
            result = kg_query(MODEL, row[0], WORKING_PATH, query_param)

            # evaluation
            queries.append(row[0])
            ground_truths.append(row[1])
            predictions.append(result)

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