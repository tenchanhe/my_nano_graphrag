import os
import logging
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from ollama import chat

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

MODEL = "qwen2.5:32b"
WORKING_DIR = "./nano_test_cache_ollama"


EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device=None
)


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


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
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({"role": "system","content": "You are an intelligent assistant and will follow the instructions given to you to fulfill the goal. The answer should be in the format as in the given example."})

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
    
    # option = {"options": {"num_ctx": 32000}}
    # response = await ollama_client.chat(model=MODEL, messages=messages, options=option)

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


def kg_query(query, query_mode, work_dir):
    rag = GraphRAG(
        working_dir=work_dir,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=local_embedding,
    )
    return rag.query(query, param=QueryParam(mode=query_mode))


def kg_insert(text, work_dir):
    from time import time

    remove_if_exist(f"{work_dir}/vdb_entities.json")
    remove_if_exist(f"{work_dir}/kv_store_full_docs.json")
    remove_if_exist(f"{work_dir}/kv_store_text_chunks.json")
    remove_if_exist(f"{work_dir}/kv_store_community_reports.json")
    remove_if_exist(f"{work_dir}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=work_dir,
        enable_llm_cache=True,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=local_embedding,
    )
    start = time()
    rag.insert(text)
    print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])
