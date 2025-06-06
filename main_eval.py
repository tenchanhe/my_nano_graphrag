from my_nano_graphrag.custom_codes.data_loader import load_data_in_batches, read_html
from my_nano_graphrag.custom_codes.evaluation import evaluate_predictions
from my_nano_graphrag.custom_codes.kg_utils import kg_insert, kg_query
from my_nano_graphrag.custom_codes.query_list import query_dict
from my_nano_graphrag.custom_codes.config_setting import query_param


if __name__ == "__main__":

    # MODEL = "llama3.2"
    MODEL = "phi4"
    EVAL_MODEL = "qwen2.5:32b"
    # EVAL_MODEL = "phi4:latest"
    BATCH_SIZE = 50
    # QUERY_MODE = 'local'
    
    DATASET_PATH = "evaluation/data/crag_task_1_dev_v4_release.jsonl.bz2"    
    # DATASET_PATH = "evaluation/data/dev_data.jsonl.bz2"
    
    # dataset_setting={"sports": 10, "movie": 10, "finance": 10, "open": 10, "music": 10}
    dataset_setting=None

    
    # Generate predictions
    queries, ground_truths, predictions = [], [], []
    urls, types, ids = [], [], []
    id = -1

    for batch in load_data_in_batches(DATASET_PATH, BATCH_SIZE, dataset_setting):
        for i in range(len(batch['query'])):
            id += 1
            if batch['query'][i] in query_dict.keys():
                working_path = query_dict[batch['query'][i]]
                
                # # KG index(build KG)
                # page_list = [
                #     read_html(page['page_result']) if page['page_result'] != "" else read_html(page['page_snippet'])
                #     for page in batch['search_results'][i]
                # ]
                # kg_insert(MODEL, page_list, working_path)
            
                # KG query
                result = kg_query(MODEL, batch['query'][i], working_path, query_param)

                # evaluation
                queries.append(batch["query"][i])
                ground_truths.append(batch["answer"][i])
                predictions.append(result)
                urls.append([pages['page_url'] for pages in batch['search_results'][i]])
                ids.append(id)
                # print([pages['page_name'] for pages in batch['search_results'][i]])
                # print(batch['search_results'][i][2]['page_result'])
                # types.append(batch['question_type'][i])

                del query_dict[batch['query'][i]]
        
        if len(query_dict) == 0:
            break
    
    evaluation_results, record_list = evaluate_predictions(
        queries, ground_truths, predictions, EVAL_MODEL
    )

    # for i in range(len(queries)):
    #     print("query: ", queries[i])
    #     print("answer: ", ground_truths[i])
    #     print("predict: ", predictions[i])
    #     print("yes_or_not: ", record_list[i])
    #     print(urls[i])
    #     print()

    for i in range(len(queries)):
        print(ids[i])
    print()
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
    print()
    for i in range(len(queries)):
        print(urls[i])

    print(evaluation_results)
