import os
import logging
import ollama
import numpy as np
from custom_codes.graphrag_custom import MyGraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# !!! qwen2-7B maybe produce unparsable results and cause the extraction of graph to fail.

# MODEL = "llama3.3"
# # WORKING_DIR = "./nano_llama3.3_cache_ollama"
# WORKING_DIR = "./nano2_llama3.3_cache_ollama"

# MODEL = "llama3.2"
# WORKING_DIR = "./nano_llama3.2_cache_ollama"

MODEL = "qwen2.5:32b"
WORKING_DIR = "./nano_xiaoming_cache_ollama"

# MODEL = "qwen2.5:32b"

EMBEDDING_MODEL = "mxbai-embed-large:latest"
EMBEDDING_MODEL_DIM = 1024
EMBEDDING_MODEL_MAX_TOKENS = 8192


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    ollama_client = ollama.Client(host='http://140.119.164.70:11435')
    embed_text = []
    for text in texts:
        data = ollama_client.embed(model=EMBEDDING_MODEL, input=text)
        # breakpoint()
        embed_text.append(data.embeddings[0])

    return embed_text


async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # print("kwars: ", kwargs)
    # breakpoint()
    # remove kwargs that are not supported by ollama
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)

    ollama_client = ollama.AsyncClient(host='http://140.119.164.70:11435',
)
    messages = []
    if system_prompt: # query
        messages.append({"role": "system", "content": system_prompt})
    else: # index
        messages.append({"role": "system", "content": "You are an intelligent assistant and will follow the instructions given to you to fulfill the goal. The answer should be in the format as in the given example."})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    
    kwargs = {"options": {"num_ctx": 32000}}
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)

    result = response["message"]["content"]
    # breakpoint()
    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    # -----------------------------------------------------
    return result


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def query():
    rag = MyGraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=ollama_embedding,
    )
    print(
        # rag.query("這個故事的主題是什麼？", param=QueryParam(mode="global"))
        # rag.query("誰是李小明？", param=QueryParam(mode="local"))
        rag.query("智慧未來科技是誰創辦的？", param=QueryParam(mode="local"))
    )


def insert():
    from time import time

    # with open("./kg_input/mock_data.txt", encoding="utf-8-sig") as f:
    with open("./kg_input/xiaoming.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()
    # print(FAKE_TEXT)

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = MyGraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=ollama_embedding,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])


if __name__ == "__main__":
    # insert()
    query()
