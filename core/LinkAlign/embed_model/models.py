from typing import Any, List
import torch

from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer
from core.utils import initialize_model_safely


class LocalHuggingFaceModel(BaseEmbedding):
    def __init__(
            self,
            instructor_model_name: str = "BAAI/bge-large-en-v1.5",
            instruction: str = "Represent the Computer Science documentation or question:",
            device: str = None,
            **kwargs: Any,
    ) -> None:
        # Use safe model initialization to avoid meta tensor issues
        self._model = SentenceTransformer(model_name_or_path=instructor_model_name)
        self._instruction = instruction
        super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode(query).tolist()  # [[self._instruction, query]]
        return embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode(text).tolist()
        return embeddings

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode([text for text in texts]).tolist()
        return embeddings

    async def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str):
        pass


