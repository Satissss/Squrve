
from typing import Dict, List, Optional
import networkx as nx
from operator import attrgetter

from .base_router import BaseRouter
from ..schema_graph import snode, Node


class KeywordRouter(BaseRouter):

    def route(
        self,
        question: str,
        graph: nx.DiGraph,
        top_k_databases: int = 3,
        top_k_tables: int = 5,
        multi_db_schema: Optional[Dict] = None,
    ) -> Dict[str, List[str]]:
        question_lower = question.lower()
        question_tokens = set(question_lower.split())
        routing_result = {}

        # 遍历所有数据库（snode 的邻居）
        d_nodes = [n for n in graph[snode]]
        for d_node in sorted(d_nodes, key=attrgetter("name")):
            db_id = d_node.name
            scores = {}
            # 获取该数据库下的所有表
            t_nodes = [n for n in graph[d_node]]
            for t_node in sorted(t_nodes, key=attrgetter("name")):
                table_name = t_node.name
                score = self._compute_relevance(
                    question_tokens, table_name
                )
                scores[table_name] = score

            top_tables = sorted(scores, key=scores.get, reverse=True)[:top_k_tables]
            if any(scores[t] > 0 for t in top_tables):
                routing_result[db_id] = top_tables

        if not routing_result and d_nodes:
            first_db = d_nodes[0].name
            t_nodes = [n for n in graph[d_nodes[0]]]
            all_tables = [t.name for t in t_nodes]
            routing_result[first_db] = all_tables[:top_k_tables]

        return routing_result

    def _compute_relevance(
        self,
        question_tokens: set,
        table_name: str
    ) -> int:
        score = 0
        table_tokens = set(table_name.lower().split("_"))

        common = question_tokens & table_tokens
        score += len(common) * 3

        return score

