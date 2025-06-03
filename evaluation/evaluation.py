import json
import ollama
from loguru import logger
from evaluation.prompt_list import PROMPTS
from tqdm.auto import tqdm
from custom_codes.config_setting import HOST
# from ollama import chat
# from transformers import LlamaTokenizerFast

# tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return PROMPTS['INSTRUCTIONS'] + PROMPTS['IN_CONTEXT_EXAMPLES']


# def attempt_api_call(client, model_name, messages, max_retries=10):
#     """Attempt an API call with retries upon encountering specific errors."""
#     # todo: add default response when all efforts fail
#     for attempt in range(max_retries):
#         try:
#             response = client.chat.completions.create(
#                 model=model_name,
#                 messages=messages,
#                 response_format={"type": "json_object"},
#             )
#             return response.choices[0].message.content
#         # except (APIConnectionError, RateLimitError):
#             logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
#         except Exception as e:
#             logger.error(f"Unexpected error: {e}")
#             break
#     return None


# def log_response(messages, response, output_directory="api_responses"):
#     """Save the response from the API to a file."""
#     os.makedirs(output_directory, exist_ok=True)
#     file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
#     file_path = os.path.join(output_directory, file_name)
#     with open(file_path, "w") as f:
#         json.dump({"messages": messages, "response": response}, f)


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        # breakpoint()
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        # print("asnwer=", model_resp, answer)
        return answer
    except:
        return -1


# def trim_predictions_to_max_token_length(prediction):
#     """Trims prediction output to 75 tokens using Llama2 tokenizer"""
#     max_token_length = 75
#     tokenized_prediction = tokenizer.encode(prediction)
#     trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
#     trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
#     return trimmed_prediction

def attempt_ollama_call(model_name, messages):
    # ollama_client = ollama.Client(host='http://140.119.164.70:11435')
    ollama_client = ollama.Client(
        host=HOST,
        headers={'authorization': 'Bearer chten:u1rRsAhk1hNY6gHMqr4t4F2Dm5QOeKzy'}
    )
    # breakpoint()

    try:        
        response = ollama_client.chat(model=model_name, messages=messages)
        return response["message"]["content"]
    
    except Exception as e:
        logger.error(f"Error calling Ollama: {e}")
        return None


def evaluate_predictions(queries, ground_truths, predictions, evaluation_model_name):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = get_system_message()
    record_list = [False for _ in range(len(queries))]

    # breakpoint()
    for _idx, prediction in enumerate(tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions"
    )):
        query = queries[_idx]
        ground_truth = str(ground_truths[_idx]).strip()

        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]

        if "i don't know" in prediction_lowercase:
            n_miss += 1
            record_list[_idx] = False
            continue
        elif prediction_lowercase == ground_truth_lowercase:
            n_correct_exact += 1
            n_correct += 1
            record_list[_idx] = True
            continue

        # 使用 Ollama 進行推理
        response = attempt_ollama_call(evaluation_model_name, messages)
        # breakpoint( )
        if response:
            # log_response(messages, response)
            eval_res = parse_response(response)
            # print("eval= ", eval_res)
            if eval_res == 1:
                n_correct += 1
                record_list[_idx] = True
            else:
                record_list[_idx] = False

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
    }
    logger.info(results)
    return results, record_list

