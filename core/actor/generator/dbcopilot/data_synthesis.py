from typing import List, Optional
from loguru import logger


class DataSynthesizer:

    def __init__(self, llm=None):
        self.llm = llm

    def synthesize_questions(
        self,
        db_id: str,
        table_names: List[str],
        graph,
        num_questions: int = 10
    ) -> List[str]:
        logger.warning("DataSynthesizer not fully implemented (reserved for future extension)")
        return []