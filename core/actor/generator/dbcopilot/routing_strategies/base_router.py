from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseRouter(ABC):

    @abstractmethod
    def route(
        self,
        question: str,
        graph,
        top_k_databases: int = 3,
        top_k_tables: int = 5,
        multi_db_schema: Optional[Dict] = None,
    ) -> Dict[str, List[str]]:
        pass