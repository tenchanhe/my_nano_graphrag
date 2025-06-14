from my_nano_graphrag.evaluation.data_loader import load_data_in_batches, read_html
from my_nano_graphrag.evaluation.evaluation import evaluate_predictions
from my_nano_graphrag.evaluation.kg_utils import kg_insert, kg_query
from my_nano_graphrag.custom_codes.config_setting import query_param


if __name__ == "__main__":

    MODEL = "llama3.2"
    # MODEL = "qwen2.5:32b"
    EVAL_MODEL = "qwen2.5:32b"
    BATCH_SIZE = 10
    # QUERY_MODE = 'naive'
    
    DATASET_PATH = "evaluation/data/crag_task_1_dev_v4_release.jsonl.bz2"    
    # DATASET_PATH = "evaluation/data/dev_data.jsonl.bz2"
    
    # dataset_setting={"sports": 10, "movie": 10, "finance": 10, "open": 10, "music": 10}
    dataset_setting=None

    # set_query="where did the ceo of salesforce previously work?"
    # WORKING_DIR = "./nano_salesforce_cache_ollama"
    # set_query="what was mike epps's age at the time of next friday's release?"
    # WORKING_DIR = "./nano_mike_qwen_cache_ollama"
    set_query = "what age did ferdinand magelan discovered the philippines"
    WORKING_DIR = "./nano_magelan_qwen_cache"
    # set_query = "what was the 76ers' record the year allen iverson won mvp?"
    # WORKING_DIR = "./nano_iverson_qwen_cache"
    # set_query="what was taylor swifts age when she released her debut album?"
    # WORKING_DIR = "./nano_taylor_qwen_cache"

    
    stop = False
    # Generate predictions
    queries, ground_truths, predictions = [], [], []
    urls, types = [], []

    for batch in load_data_in_batches(DATASET_PATH, BATCH_SIZE, dataset_setting):
        for i in range(len(batch['query'])):
            if batch['query'][i] == set_query:
                # KG index(build KG)
                page_list = [
                    read_html(page['page_result']) if page['page_result'] != "" else read_html(page['page_snippet'])
                    for page in batch['search_results'][i]
                ]
                kg_insert(MODEL, page_list, WORKING_DIR)
            
                # KG query
                result = kg_query(MODEL, batch['query'][i], WORKING_DIR, query_param)

                # evaluation
                queries.append(batch["query"][i])
                ground_truths.append(batch["answer"][i])
                predictions.append(result)
                # breakpoint()
                urls.append([pages['page_url'] for pages in batch['search_results'][i]])
                # breakpoint()
                # types.append(batch['question_type'][i])

                stop = True
                break
        if stop:
            break
    
    for i in range(len(queries)):
        print("query: ", queries[i])
        print("answer: ", ground_truths[i])
        print("predict: ", predictions[i])
        print(urls[i])
        print()

    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, EVAL_MODEL
    )

    # import ipdb;ipdb.set_trace()
    print(evaluation_results)
