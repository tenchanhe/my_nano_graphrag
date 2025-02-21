from tqdm.auto import tqdm
from evaluation.data_loader import load_data_in_batches, read_html
from evaluation.evaluation import evaluate_predictions
from evaluation.kg_utils import kg_insert, kg_query



if __name__ == "__main__":
    # from models.user_config import UserModel

    MODEL = "llama3.2"
    EVAL_MODEL = "llama3.3"
    DATASET_PATH = "evaluation/data/dev_data.jsonl.bz2"
    # DATASET_PATH = "evaluation/data/dev_data.jsonl.bz2"
    WORKING_DIR = "./nano_test_cache_ollama"
    BATCH_SIZE = 2
    dataset_setting={"sports": 10, "movie": 10, "finance": 10, "open": 10, "music": 10}


    # Generate predictions
    queries, ground_truths, predictions = [], [], []

    for batch in tqdm(load_data_in_batches(DATASET_PATH, BATCH_SIZE, dataset_setting), desc="Generating predictions"):
        # index KG
        # breakpoint()
        for i in range(len(batch['query'])):
            # insert_text = read_html(batch['search_results'][i])
            # kg_insert(MODEL, insert_text, WORKING_DIR)
            
            # KG query
            result = kg_query(MODEL, batch['query'][i], 'local', WORKING_DIR)

            # evaluation
            # batch_ground_truths = batch.pop("answer")
            
            queries.append(batch["query"][i])
            ground_truths.append(batch["answer"][i])
            predictions.append(result)
            break
        break
    
    for i in range(len(queries)):
        print("query: ", queries[i])
        print("answer: ", ground_truths[i])
        print("predict: ", predictions[i])
        print()

    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, EVAL_MODEL
    )

    # import ipdb;ipdb.set_trace()
    print(evaluation_results)
