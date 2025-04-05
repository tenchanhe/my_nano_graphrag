from nano_graphrag.base import QueryParam

HOST = "http://140.119.164.70:11435"
# HOST = "http://140.119.164.60:8080"

query_param = QueryParam(
    # mode='local',
    # mode='global',
    mode = "naive",
    only_need_context = False,
    response_type = "Multiple Paragraphs",
    # response_type = "Just Simple Answer",
    level = 2,
    top_k = 20,
    # naive search
    # naive_max_token_for_text_unit = 12000,
    # local search
    local_max_token_for_text_unit  = 4000,  # 12000 * 0.33
    local_max_token_for_local_context  = 4800,  # 12000 * 0.4
    local_max_token_for_community_report  = 3200,  # 12000 * 0.27
    local_community_single_one = False,
    # global search
    global_min_community_rating = 0,
    global_max_consider_community = 512,
    global_max_token_for_community_report = 16384,
    global_special_community_map_llm_kwargs = {"response_format": {"type": "json_object"}}
)