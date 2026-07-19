from typing import Dict, List, Optional
from loguru import logger

from .routing_strategies.base_router import BaseRouter
from .routing_strategies.llm_router import LLMRouter
from .routing_strategies.keyword_router import KeywordRouter
from .routing_strategies.embedding_router import EmbeddingRouter


class SchemaRouter:

    def __init__(self, router_type: str = "llm", llm=None, embed_model=None):
        self.router = self._create_router(router_type, llm, embed_model)

    def _create_router(self, router_type: str, llm: Optional[object] = None, embed_model: Optional[object] = None) -> BaseRouter:
        if router_type == "llm":
            if llm is None:
                raise ValueError("LLM instance required for LLM routing")
            return LLMRouter(llm)
        elif router_type == "keyword":
            return KeywordRouter()
        elif router_type == "embedding":
            return EmbeddingRouter(embed_model)
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

    def route(
        self,
        question: str,
        graph,
        top_k_databases: int = 3,
        top_k_tables: int = 5,
        multi_db_schema: Optional[Dict] = None,
    ) -> Dict[str, List[str]]:
        try:
            return self.router.route(question, graph, top_k_databases, top_k_tables, multi_db_schema)
        except Exception as e:
            logger.warning(f"Router failed: {e}, falling back to keyword routing")
            return KeywordRouter().route(question, graph, top_k_databases, top_k_tables)