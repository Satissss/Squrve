from typing import Dict, List, Optional
import json

from loguru import logger

from .base_router import BaseRouter
from ..schema_graph import to_summary_text


class LLMRouter(BaseRouter):

    def __init__(self, llm):
        self.llm = llm

    def route(
        self,
        question: str,
        graph,
        top_k_databases: int = 3,
        top_k_tables: int = 5,
        multi_db_schema: Optional[Dict] = None,
    ) -> Dict[str, List[str]]:
        summary = to_summary_text(graph, multi_db_schema)
        prompt = self._build_prompt(question, summary, top_k_databases, top_k_tables)

        try:
            response = self.llm.complete(prompt).text
            return self._parse_response(response, top_k_tables)
        except Exception as e:
            logger.warning(f"LLMRouter failed: {e}")
            raise

    def _build_prompt(self, question: str, summary: str, top_k_db: int, top_k_table: int) -> str:
        return f"""You are a Schema Routing expert. Given a natural language question and a list of databases with their tables and foreign key relations, identify which databases and tables are relevant.

Available Databases:
{summary}

Question: "{question}"

Task:
1. Select the most relevant database(s) (up to {top_k_db})
2. For each selected database, select the most relevant table(s) (up to {top_k_table})
3. Use foreign key relationships to include related tables when relevant

Respond in JSON format:
{{
    "selected_databases": [
        {{
            "db_id": "database_name",
            "relevant_tables": ["table1", "table2"],
            "reasoning": "brief explanation"
        }}
    ]
}}

JSON Response:"""

    def _parse_response(self, response: str, top_k_table: int) -> Dict[str, List[str]]:
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]

        result = json.loads(json_str.strip())
        routing_result = {}
        for db_info in result.get("selected_databases", []):
            db_id = db_info.get("db_id")
            tables = db_info.get("relevant_tables", [])
            if db_id and tables:
                routing_result[db_id] = tables[:top_k_table]

        return routing_result