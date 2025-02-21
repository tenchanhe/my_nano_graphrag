from dataclasses import asdict
from nano_graphrag.graphrag import GraphRAG
from nano_graphrag.base import QueryParam
from custom_codes.op_custom import local_query, global_query

class MyGraphRAG(GraphRAG):
    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "local":
            param.response_type = "Just Simple Answer"
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        # elif param.mode == "naive":
        #     response = await naive_query(
        #         query,
        #         self.chunks_vdb,
        #         self.text_chunks,
        #         param,
        #         asdict(self),
        #     )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response