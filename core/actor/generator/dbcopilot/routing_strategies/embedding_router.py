from typing import Dict, List, Optional
from operator import attrgetter

import numpy as np
from loguru import logger

from .base_router import BaseRouter
from ..schema_graph import snode


class EmbeddingRouter(BaseRouter):

    DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self, embed_model=None):
        self._embed_model = embed_model

    def _load_model(self):
        if self._embed_model is not None:
            return self._embed_model
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.DEFAULT_MODEL_NAME)
            return self._embed_model
        except ImportError:
            return None

    def route(
        self,
        question: str,
        graph,
        top_k_databases: int = 3,
        top_k_tables: int = 5,
        multi_db_schema: Optional[Dict] = None,
    ) -> Dict[str, List[str]]:
        embed_model = self._load_model()
        if embed_model is None:
            logger.warning(
                "sentence_transformers not available, "
                "falling back to keyword routing"
            )
            from .keyword_router import KeywordRouter
            return KeywordRouter().route(
                question, graph, top_k_databases, top_k_tables
            )

        d_nodes = sorted(
            [n for n in graph[snode]], key=attrgetter("name")
        )

        db_tables: Dict[str, List[str]] = {}
        all_tables: List[str] = []

        for d_node in d_nodes:
            db_id = d_node.name
            t_nodes = sorted(
                [n for n in graph[d_node]], key=attrgetter("name")
            )
            table_names = [t.name for t in t_nodes]
            db_tables[db_id] = table_names
            all_tables.extend(table_names)

        if not all_tables:
            return {}

        question_emb = np.asarray(
            embed_model.encode(question), dtype=np.float32
        )
        question_norm = np.linalg.norm(question_emb)

        table_texts = [t.replace("_", " ") for t in all_tables]
        table_embs = np.asarray(
            embed_model.encode(table_texts), dtype=np.float32
        )
        table_norms = np.linalg.norm(table_embs, axis=1)

        similarities = np.dot(table_embs, question_emb) / (
            table_norms * question_norm + 1e-8
        )

        db_scores: Dict[str, float] = {}
        for db_id, tables in db_tables.items():
            indices = [all_tables.index(t) for t in tables]
            db_scores[db_id] = float(np.max(similarities[indices]))

        sorted_dbs = sorted(
            db_scores, key=db_scores.get, reverse=True
        )[:top_k_databases]

        routing_result = {}
        for db_id in sorted_dbs:
            tables = db_tables[db_id]
            sims = [
                (t, float(similarities[all_tables.index(t)]))
                for t in tables
            ]
            sims.sort(key=lambda x: x[1], reverse=True)
            routing_result[db_id] = [t for t, _ in sims[:top_k_tables]]

        if not routing_result and d_nodes:
            first_db = d_nodes[0].name
            t_nodes = sorted(
                [n for n in graph[d_nodes[0]]], key=attrgetter("name")
            )
            fallback_tables = [t.name for t in t_nodes]
            routing_result[first_db] = fallback_tables[:top_k_tables]

        return routing_result
