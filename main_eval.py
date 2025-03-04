from tqdm.auto import tqdm
from evaluation.data_loader import load_data_in_batches, read_html
from evaluation.evaluation import evaluate_predictions
from evaluation.kg_utils import kg_insert, kg_query


if __name__ == "__main__":

    MODEL = "llama3.2"
    EVAL_MODEL = "qwen2.5:32b"
    BATCH_SIZE = 10
    QUERY_MODE = 'local'
    
    DATASET_PATH = "evaluation/data/crag_task_1_dev_v4_release.jsonl.bz2"
    WORKING_DIR = "./nano_salesforce_cache_ollama"
    
    # DATASET_PATH = "evaluation/data/dev_data.jsonl.bz2"
    # WORKING_DIR = "./nano_test_cache_ollama"
    
    dataset_setting={"sports": 10, "movie": 10, "finance": 10, "open": 10, "music": 10}
    set_query = 'where did the ceo of salesforce previously work?'
    stop = False

    # Generate predictions
    queries, ground_truths, predictions = [], [], []
    urls, types = [], []

    for batch in tqdm(load_data_in_batches(DATASET_PATH, BATCH_SIZE, dataset_setting), desc="Generating predictions"):
        for i in range(len(batch['query'])):
            if batch['query'][i] == set_query:
                # # KG index(build KG)
                # insert_text = read_html(batch['search_results'][i])
                # kg_insert(MODEL, insert_text, WORKING_DIR)
            
                # KG query
                result = kg_query(MODEL, batch['query'][i], QUERY_MODE, WORKING_DIR)

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
        print(urls[i])
        print("predict: ", predictions[i])
        print()

    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, EVAL_MODEL
    )

    # import ipdb;ipdb.set_trace()
    print(evaluation_results)
