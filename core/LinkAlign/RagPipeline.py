# -*- coding: utf-8 -*-
import warnings

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    SummaryIndex,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from typing import Union, List
from pathlib import Path
import os


class RagPipeLines:
    EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
    INDEX_METHODS = ["SummaryIndex", "VectorStoreIndex"]
    MAX_CHUNK_SIZE = 15000
    COUNT = 1

    @classmethod
    def build_index_from_source(
            cls,
            data_source: Union[str, List[str]],
            persist_dir: str = None,
            is_vector_store_exist: bool = False,
            index_method: str = None,
            embed_model_name=None,
            parser=None,
    ):
        # Set default embedding model if not provided
        embed_model_name = embed_model_name or cls.EMBED_MODEL_NAME
        Settings.embed_model = HuggingFaceEmbedding(embed_model_name, trust_remote_code=True)

        # Validate index method
        index_method = index_method if index_method in cls.INDEX_METHODS else None

        # Use default parser if not provided
        parser = parser or SentenceSplitter(chunk_size=cls.MAX_CHUNK_SIZE, chunk_overlap=0)

        if not persist_dir:
            raise ValueError("persist_dir cannot be None or empty.")

        # Load from existing vector store if requested
        if is_vector_store_exist:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            return load_index_from_storage(storage_context)

        os.makedirs(persist_dir, exist_ok=True)

        # Determine if input is directory-based
        if isinstance(data_source, list):
            is_dir = False
        elif isinstance(data_source, str):
            is_dir = Path(data_source).is_dir()
        else:
            raise TypeError("data_source must be a string path or a list of file paths.")

        # Determine whether to use VectorStoreIndex
        use_vector_store = index_method == "VectorStoreIndex" if index_method else is_dir

        # Load documents based on source type
        if use_vector_store:
            documents = SimpleDirectoryReader(data_source).load_data()
            index = VectorStoreIndex.from_documents(
                documents, transformations=[parser], show_progress=True
            )
        else:
            if isinstance(data_source, list):
                reader = SimpleDirectoryReader(input_files=data_source)
            else:
                reader = SimpleDirectoryReader(input_files=[data_source]) if not is_dir else SimpleDirectoryReader(
                    data_source)
            documents = reader.load_data()
            index = SummaryIndex.from_documents(
                documents, transformations=[parser], show_progress=True
            )

        # Persist and return the index
        index.storage_context.persist(persist_dir=persist_dir)
        return index

    @classmethod
    def get_query_engine(
            cls,
            index: Union[SummaryIndex, VectorStoreIndex] = None,
            query_template: str = None,
            similarity_top_k: int = 5,
            node_ids: List[str] = None,
            **kwargs
    ):
        if index is None:
            raise ValueError("The 'index' argument cannot be None.")

        prompt_template = PromptTemplate(query_template) if query_template else None

        if isinstance(index, SummaryIndex):
            return index.as_query_engine(
                text_qa_template=prompt_template,
                similarity_top_k=similarity_top_k,
                **kwargs
            )

        if isinstance(index, VectorStoreIndex):
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=similarity_top_k,
                node_ids=node_ids,
                **kwargs
            )
            return RetrieverQueryEngine.from_args(
                retriever=retriever,
                text_qa_template=prompt_template
            )

        raise TypeError("Unsupported index type. Expected SummaryIndex or VectorStoreIndex.")

    @classmethod
    def get_retriever(
            cls,
            index: VectorStoreIndex = None,
            similarity_top_k: int = 5,
            node_ids: List[str] = None,
            **kwargs
    ):
        if index is None:
            warnings.warn(
                "Failed to create vector retriever: 'index' must not be None.",
                category=UserWarning
            )
            return None

        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            node_ids=node_ids,
            **kwargs
        )


if __name__ == "__main__":
    vector_dir = r"..."
    vector_index = RagPipeLines.build_index_from_source(
        data_source=vector_dir,
        persist_dir=vector_dir + r"\vector_store",
        is_vector_store_exist=True,
        index_method="VectorStoreIndex"
    )
