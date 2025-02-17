from tqdm.auto import tqdm
from evaluation.data_loader import load_data_in_batches, read_html
from evaluation.evaluation import evaluate_predictions
from evaluation.kg_utils import kg_insert, kg_query


if __name__ == "__main__":
    # from models.user_config import UserModel

    DATASET_PATH = "evaluation/data/crag_task_1_dev_v4_release.jsonl.bz2"
    WORKING_DIR = "./nano_test_cache_ollama"
    dataset_setting={"sports": 10, "movie": 10, "finance": 10, "open": 10, "music": 10}


    # Generate predictions
    queries, ground_truths, predictions = [], [], []

    for batch in tqdm(load_data_in_batches(DATASET_PATH, 1, dataset_setting), desc="Generating predictions"):
        # index KG
        # breakpoint()
        insert_text = read_html(batch['search_results'])
        kg_insert(insert_text, WORKING_DIR)
        
        # KG query
        result = kg_query(batch['query'], 'local', WORKING_DIR)

        # evaluation
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(result)
        break
    
    for i in range(len(queries)):
        print("query: ", queries[i])
        print("predict: ", predictions[i])
        print()

    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, "llama3.2"
    )

    # import ipdb;ipdb.set_trace()
    print(evaluation_results)
