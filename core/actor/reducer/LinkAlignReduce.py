from os import PathLike
from pathlib import Path

from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.llms.llm import LLM
from typing import Union, List, Dict
import pandas as pd
import random

from core.data_manage import Dataset, transform_name, single_central_process
from core.actor.reducer.BaseReduce import BaseReducer
from core.LinkAlign.SchemaLinkingTool import SchemaLinkingTool
from core.LinkAlign.RagPipeline import RagPipeLines
from core.utils import (
    parse_schemas_from_nodes,
    parse_schema_from_df,
    save_dataset,
    load_dataset,
    parse_schema_link_from_str
)

@BaseReducer.register_actor
class LinkAlignReducer(BaseReducer):
    """ Use LinkAlign’s Schema Reduce to get a sample’s related schema subset """

    NAME = "LinkAlignReducer"

    def __init__(
            self,
            # Basic
            dataset: Dataset = None,
            llm: LLM = None,
            output_format: str = "dataframe",  # output in `dataframe` or `json`
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/instance_schemas",  # Relative to the path of data_manage.py
            automatic: bool = True,  # Automatic parameter configuration
            reserve_size: int = 80,
            use_external: bool = True,  # Use external knowledge to explain the question
            # Retrieval
            skip_retrieval: bool = False,
            retrieval_mode: str = "agent",  # Pipeline or Agent
            top_k: int = 50,  # The most similar Top K schema is selected
            retrieval_turn_n: int = 2,
            min_retrival_size: int = 300,
            # Filter
            skip_filter: bool = False,
            filter_mode: str = "agent",
            filter_num: int = 2,
            filter_turn_n: int = 2,
            min_fiter_size: int = 80,  # only the schema size fewer than this num, then open `Response Filtering`
            filter_chunk_size: int = 300,
            init_retain_rate: float = 0.6,
            decay_rate: float = 0.55,  # Retention rate decays exponentially with retrieval epochs
            # Post Retrieval
            skip_post_retrieval: bool = False,  # only effective when skip_filter is True
            post_top_k: int = 10,
            post_turn_n: int = 1,
            **kwargs
    ):

        # Basic
        self.dataset: Dataset = dataset
        self.llm: LLM = llm
        self.output_format: str = output_format
        self.is_save: bool = is_save
        self.save_dir: Union[str, PathLike] = save_dir
        self.automatic: bool = automatic
        self.use_external: bool = use_external
        self.reserve_size: int = reserve_size

        # Retrieval
        self.skip_retrieval: bool = skip_retrieval
        self.retrieval_mode: str = retrieval_mode
        self.top_k: int = top_k
        self.retrieval_turn_n: int = retrieval_turn_n
        self.min_retrival_size: int = min_retrival_size

        # Response Filtering
        self.skip_filter: bool = skip_filter
        self.filter_mode: str = filter_mode
        self.filter_num: int = filter_num
        self.filter_turn_n: int = filter_turn_n
        self.min_fiter_size: int = min_fiter_size
        self.filter_chunk_size: int = filter_chunk_size
        self.init_retain_rate: float = init_retain_rate
        self.decay_rate: float = decay_rate

        # Post Retrieval
        self.skip_post_retrieval: bool = skip_post_retrieval
        self.post_top_k: int = post_top_k
        self.post_turn_n: int = post_turn_n

    @classmethod
    def load_retrieval_top_k(cls, db_size: int):
        if db_size <= 200:
            return 40
        elif db_size <= 400:
            return 50
        elif db_size <= 1000:
            return 60
        elif db_size <= 2500:
            return 70
        else:
            return 80

    @classmethod
    def load_retrieval_turn_n(cls, db_size: int):
        if db_size <= 200:
            return 2
        elif db_size <= 350:
            return 3
        elif db_size <= 1000:
            return 6
        elif db_size <= 2500:
            return 8
        else:
            return 10

    @classmethod
    def load_post_retrival_param(cls, db_size):
        if db_size <= 200:
            return 5, 1
        elif db_size <= 500:
            return 10, 1
        elif db_size <= 1000:
            return 15, 1
        elif db_size <= 2000:
            return 15, 2
        else:
            return 20, 1

    @classmethod
    def set_retriever(
            cls,
            retriever: VectorIndexRetriever,
            data: pd.DataFrame,
    ):
        table_lis, col_lis = list(data["table_name"]), list(data["column_name"])
        file_name_lis = []
        for table, col in zip(table_lis, col_lis):
            file_name_lis.append(transform_name(table, col))

        index = retriever.index
        sub_ids = []
        doc_info_dict = index.ref_doc_info
        for key, ref_doc_info in doc_info_dict.items():
            if ref_doc_info.metadata["file_name"] not in file_name_lis:
                sub_ids.extend(ref_doc_info.node_ids)
        retriever.change_node_ids(sub_ids)

    def get_retain_schema(self, schema: Union[List[Dict], pd.DataFrame]) -> pd.DataFrame:
        schema_df = pd.DataFrame(schema) if isinstance(schema, list) else schema
        if "turn_n" not in schema_df.columns:
            schema_df["turn_n"] = 0
        turn_n_lis = schema_df["turn_n"].unique().tolist()
        df_lis = []
        for n in turn_n_lis:
            temp_df = schema_df[schema_df["turn_n"] == n]
            df_reserver_rate = self.init_retain_rate * pow(self.decay_rate, n)
            if df_reserver_rate <= 0.1:
                continue
            temp_df = temp_df.sample(int(len(temp_df) * df_reserver_rate), random_state=42)
            df_lis.append(temp_df)

        schema_df = pd.concat(df_lis, axis=0, ignore_index=True)

        return schema_df

    def response_filtering(
            self,
            data: pd.DataFrame,
            question: str,
            chunk_size: int = None,
            turn_n: int = None,
            reserve_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        chunk_size = self.filter_chunk_size if not chunk_size else chunk_size
        turn_n = self.filter_turn_n if not turn_n else turn_n
        is_single_mode = not self.dataset.is_multi_database
        # 切分数据为多个块
        chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        retained_chunks = [reserve_df] if reserve_df is not None else []
        voted_objective_database = {}

        for chunk_df in chunks:
            schema_context = parse_schema_from_df(chunk_df)
            filter_args = {
                "mode": self.filter_mode,
                "llm": self.llm,
                "query": question,
                "context_str": schema_context,
                "turn_n": turn_n,
                "is_single_mode": is_single_mode
            }
            response = SchemaLinkingTool.locate_selector(**filter_args)
            if is_single_mode:
                schema_links = parse_schema_link_from_str(response)  # [a.a1, b.b1]
                table_field_pairs = [link.split(".")[:2] for link in schema_links]
                for table, field in table_field_pairs:
                    chunk_df = chunk_df.query("not (`table_name` == @table and `column_name` == @field)")
            else:
                if response in voted_objective_database.keys():
                    voted_objective_database[response] += len(chunk_df)
                else:
                    voted_objective_database[response] = len(chunk_df)

            retained_chunks.append(chunk_df)

        final_df = pd.concat(retained_chunks, ignore_index=True).drop_duplicates(
            subset=['table_name', 'column_name'],
            ignore_index=True
        )

        if not is_single_mode:
            objective_db = [k for k, v in voted_objective_database.items() if
                            v == max(voted_objective_database.values())]
            objective_db = objective_db[0] if len(objective_db) == 1 else random.choice(objective_db)
            final_df = final_df[final_df["db_id"] == objective_db].reset_index(drop=True)

        return final_df

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def act(self, item, schema: Union[Dict, List] = None, data_logger=None, **kwargs):
        # schema 参数尽量为空即可，否则可能与嵌入 schema 存在不一致
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        row = self.dataset[item]
        source = self.dataset.schema_source
        db_size = row.get("db_size", 0)
        question = row["question"]
        if data_logger:
            data_logger.info(f"{self.NAME}.act context | db_size={db_size} | question_preview={question}")
        if self.use_external:
            external = self.load_external_knowledge(row.get("external", None))
            question = question if external is None else question + "\n" + external

        save_path = None
        if self.is_save:
            instance_id = row.get("instance_id")
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            if self.output_format == "dataframe":
                save_path = save_path / f"{self.name}_{instance_id}.csv"
            else:
                save_path = save_path / f"{self.name}_{instance_id}.json"

        if db_size <= self.reserve_size:
            if data_logger:
                data_logger.info(
                    f"[LinkAlignReducer.act] Item {item}: db_size ({db_size}) <= reserve_size ({self.reserve_size}), using original schema")
            sub_schema = self.dataset.get_db_schema(item) if not schema else schema
            if self.output_format == "dataframe":
                if isinstance(sub_schema, dict):
                    sub_schema = single_central_process(sub_schema)
                sub_schema = pd.DataFrame(sub_schema)
            if save_path:
                save_dataset(sub_schema, new_data_source=save_path)
                self.dataset.setitem(item, "instance_schemas", str(save_path))
            return sub_schema

        vector_index, retriever = None, None
        if not self.skip_retrieval or not self.skip_post_retrieval:
            vector_index = self.dataset.get_vector_index(item)
            retriever = RagPipeLines.get_retriever(index=vector_index)

        if self.skip_retrieval:
            if data_logger:
                data_logger.info(f"[LinkAlignReducer.act] Item {item}: skip_retrieval=True, using original schema")
            sub_schema = self.dataset.get_db_schema(item) if not schema else schema
        else:
            if db_size <= self.min_retrival_size:
                if data_logger:
                    data_logger.info(
                        f"[LinkAlignReducer.act] Item {item}: db_size ({db_size}) <= min_retrival_size ({self.min_retrival_size}), using original schema")
                sub_schema = self.dataset.get_db_schema(item) if not schema else schema
            else:
                # 必须保证完成 schema init，并且索引已经建立
                top_k, turn_n = self.top_k, self.retrieval_turn_n
                if self.automatic:
                    top_k = self.load_retrieval_top_k(db_size)
                    turn_n = self.load_retrieval_turn_n(db_size)

                retriever.similarity_top_k = top_k
                retrieval_args = {
                    "mode": self.retrieval_mode,
                    "llm": self.llm,
                    "question": question,
                    "retriever_lis": [retriever],
                    "open_locate": False,
                    "output_format": "node",
                    "retrieve_turn_n": turn_n,
                    "schema_source": source,
                    "db_id": row["db_id"]
                }
                nodes = SchemaLinkingTool.retrieve_complete_selector(**retrieval_args)
                sub_schema = parse_schemas_from_nodes(nodes=nodes, output_format="list", schema_source=source,
                                                      db_id=row["db_id"])

        if isinstance(sub_schema, dict):
            sub_schema = single_central_process(sub_schema)

        df = pd.DataFrame(sub_schema)
        if data_logger:
            data_logger.info(f"{self.NAME}.to_dataframe output | item={item} | shape={df.shape}")
        if not self.skip_filter:
            if data_logger:
                data_logger.info(f"[LinkAlignReducer.act] Item {item}: Starting post-retrieval, skip_post_retrieval=False")
            retain_schema = self.get_retain_schema(sub_schema)
            if data_logger:
                data_logger.info(
                    f"{self.NAME}.retain_schema output | item={item} | shape={retain_schema.shape if hasattr(retain_schema, 'shape') else 'N/A'}")
            for filter_iter in range(self.filter_num):
                if len(df) > self.min_fiter_size:
                    df = self.response_filtering(data=df, question=question, reserve_df=retain_schema)
                    if data_logger:
                        data_logger.info(
                            f"{self.NAME}.response_filtering iteration | item={item} | iteration={filter_iter + 1} | df_size={len(df)}")
            if not self.skip_post_retrieval:
                post_top_k, post_turn_n = self.post_top_k, self.post_turn_n
                if self.automatic:
                    post_top_k, post_turn_n = self.load_post_retrival_param(db_size)
                self.set_retriever(retriever, df)  # 对 retriever 重新进行设置，检索剩余的模式
                retriever.similarity_top_k = post_top_k
                post_retrieval_args = {
                    "mode": self.retrieval_mode,
                    "llm": self.llm,
                    "question": question,
                    "retriever_lis": [retriever],
                    "open_locate": False,
                    "output_format": "node",
                    "retrieve_turn_n": post_turn_n,
                    "schema_source": source,
                    "db_id": row["db_id"]
                }
                nodes = SchemaLinkingTool.retrieve_complete_by_multi_agent_debate(**post_retrieval_args)
                sub_df = parse_schemas_from_nodes(nodes=nodes, schema_source=source, db_id=row["db_id"])
                df = pd.concat([df, sub_df], axis=0)
                if data_logger:
                    data_logger.info(
                        f"{self.NAME}.post_retrieval output | item={item} | nodes={len(nodes) if nodes else 0} | combined_shape={df.shape}")

        if self.output_format == "dataframe":
            if save_path:
                save_dataset(df, new_data_source=save_path)
                self.dataset.setitem(item, "instance_schemas", str(save_path))
            return df

        assert isinstance(df, pd.DataFrame)
        sub_schema = df.to_dict(orient='records')
        if save_path:
            save_dataset(sub_schema, new_data_source=save_path)
            self.dataset.setitem(item, "instance_schemas", str(save_path))

        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return sub_schema
