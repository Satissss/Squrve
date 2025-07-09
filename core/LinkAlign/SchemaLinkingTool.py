# -*- coding: utf-8 -*-
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core import (
    SummaryIndex,
    VectorStoreIndex,
    Settings,
    QueryBundle,

)
from llama_index.core.llms.llm import LLM
from llama_index.core.indices.utils import default_format_node_batch_fn
from llama_index.core.schema import MetadataMode
from llama_index.core.base.base_retriever import BaseRetriever

from core.LinkAlign.prompts.PipelinePromptStore import *
from core.LinkAlign.RagPipeline import RagPipeLines
from core.LinkAlign.prompts.AgentPromptStore import *
from core.utils import *

from typing import Union

import asyncio


class SchemaLinkingTool:
    @classmethod
    def link_schema_by_rag(
            cls,
            llm: LLM = None,
            index: Union[SummaryIndex, VectorStoreIndex] = None,
            is_add_example: bool = True,
            question: str = None,
            similarity_top_k: int = 5,
            **kwargs
    ) -> str:
        if not index:
            raise Exception("The index cannot be empty!")

        if not question:
            raise Exception("The question cannot be empty!")

        if not llm:
            raise Exception("The llm cannot be empty!")

        Settings.llm = llm

        few_examples = SCHEMA_LINKING_FEW_EXAMPLES if is_add_example else ""

        query_template = SCHEMA_LINKING_TEMPLATE.format(few_examples=few_examples, question=question)

        engine_args = {
            "index": index,
            "query_template": query_template,
            "similarity_top_k": similarity_top_k,
            **kwargs
        }

        engine = RagPipeLines.get_query_engine(**engine_args)

        response = engine.query(question).response

        return response

    @classmethod
    def retrieve(
            cls,
            retriever_lis: List[BaseRetriever],
            query_lis: List[Union[str, QueryBundle]]
    ) -> List[NodeWithScore]:
        """ 串行化检索 """
        nodes_lis = []

        for retriever in retriever_lis:
            for query in query_lis:
                nodes = retriever.retrieve(query)
                nodes_lis.extend(nodes)

        nodes_lis.sort(key=lambda x: x.score, reverse=True)

        return nodes_lis

    @classmethod
    def parallel_retrieve(
            cls,
            retriever_list: List[BaseRetriever],  # Different source documents per retriever
            query_list: List[Union[str, QueryBundle]]
    ) -> List[NodeWithScore]:

        async def retrieve_all() -> List[List[NodeWithScore]]:
            async def retrieve(retriever: BaseRetriever, query: Union[str, QueryBundle]):
                return await retriever.aretrieve(query)

            tasks = [
                asyncio.create_task(retrieve(retriever, query))
                for retriever in retriever_list
                for query in query_list
            ]
            return await asyncio.gather(*tasks)

        if not retriever_list:
            raise ValueError("The 'retriever_list' must not be empty.")
        if not query_list:
            raise ValueError("The 'query_list' must not be empty.")

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(retrieve_all())
        finally:
            loop.close()

        # Flatten and sort results
        all_nodes = [node for result in results for node in result]
        all_nodes.sort(key=lambda x: x.score, reverse=True)

        return all_nodes

    @classmethod
    def query_rewriting(
            cls,
            llm=None,
            query: str = None,
            context: str = None
    ) -> str:
        """Use the LLM to rewrite or enhance the input query based on provided context."""

        if not query:
            raise ValueError("The input query must not be empty.")

        if llm is None:
            raise ValueError("The 'llm' parameter must not be None.")

        prompt = QUERY_REWRITING_TEMPLATE.format(question=query, context=context or "")

        rewritten_query = llm.complete(prompt=prompt).text

        return rewritten_query

    @classmethod
    def retrieve_complete(
            cls,
            question: str = None,
            retriever_lis: List[VectorIndexRetriever] = None,
            llm=None,
            open_reason_enhance: bool = True,
            open_locate: bool = False,  # Typically disabled during testing
            open_agent_debate: bool = False,  # Effective only if `open_locate` is True
            turn_n: int = 2,
            output_format: str = "database",  # "database" for database names, "schema" for schema details
            remove_duplicate: bool = True,
            is_all: bool = True,
            enhanced_question: str = None,
            **kwargs
    ):
        """
        Step One: Retrieve potential database schemas.
        Mode: Pipeline.
        """

        if not question:
            raise ValueError("The input 'question' must not be empty.")

        if not retriever_lis:
            raise ValueError("The input 'retriever_lis' (retriever list) must not be empty.")

        if llm is None:
            raise ValueError("The 'llm' parameter must not be None.")

        # Step 1: Retrieve with original question
        nodes = cls.parallel_retrieve(retriever_lis, [question])
        nodes = [set_node_turn_n(node, 0) for node in nodes]

        # Step 2: Reasoning-based enhancement (if enabled)
        if open_reason_enhance:
            context = parse_schema_from_df(parse_schemas_from_nodes(nodes))

            if not remove_duplicate:
                # Retrieve with both original and enhanced question
                analysis = cls.query_rewriting(llm=llm, query=question, context=context)
                enhanced_question = question + analysis
                nodes += cls.parallel_retrieve(retriever_lis, [enhanced_question])
            else:
                index_list = [ret.index for ret in retriever_lis]
                sub_ids = get_sub_ids(nodes, index_list, is_all=is_all)

                for ret in retriever_lis:
                    ret.change_node_ids(sub_ids)

                if enhanced_question is None:
                    analysis = cls.query_rewriting(llm=llm, query=question, context=context)
                    enhanced_question = question + analysis

                enhanced_nodes = cls.parallel_retrieve(retriever_lis, [enhanced_question])
                enhanced_nodes = [set_node_turn_n(node, 1) for node in enhanced_nodes]
                nodes += enhanced_nodes

                for ret in retriever_lis:
                    ret.back_to_original_ids()

        # Step 3: Sort all nodes by score
        nodes.sort(key=lambda node: node.score, reverse=True)

        # Step 4: Locate database (optional)
        if open_locate:
            if open_agent_debate:
                predicted_database = cls.locate_with_multi_agent(
                    llm=llm, query=question, nodes=nodes, turn_n=turn_n
                )
            else:
                schema_context = get_all_schemas_from_schema_text(nodes, output_format="schema")
                predicted_database = cls.locate(
                    llm=llm, query=question, context=schema_context
                )
            return predicted_database

        # Step 5: Output schema or database name
        return get_all_schemas_from_schema_text(
            nodes=nodes,
            output_format=output_format,
            schemas_format="str",
            is_all=is_all
        )

    @classmethod
    def retrieve_complete_by_multi_agent_debate(
            cls,
            question: str = None,
            retrieve_turn_n: int = 2,
            locate_turn_n: int = 2,
            retriever_lis: List[VectorIndexRetriever] = None,
            llm=None,
            open_locate: bool = False,  # Enable in formal experiments only
            open_agent_debate: bool = False,
            output_format: str = "database",  # 'database' or 'schema'
            remove_duplicate: bool = True,
            is_all: bool = True,
            **kwargs
    ):
        """
        Step One: Retrieve potential database schemas.
        Mode: Agent.
        """

        if not question:
            raise ValueError("The input question cannot be empty!")
        if not retriever_lis:
            raise ValueError("The retriever list cannot be empty!")
        if not llm:
            raise ValueError("The LLM cannot be empty!")

        enhanced_question = question
        nodes = cls.parallel_retrieve(retriever_lis, [question])
        nodes = [set_node_turn_n(node, 0) for node in nodes]

        index_lis = [ret.index for ret in retriever_lis]
        sub_ids = get_ids_from_source(nodes)

        for turn in range(retrieve_turn_n):
            if remove_duplicate:
                for ret in retriever_lis:
                    ret.change_node_ids(sub_ids)

            if turn > 0 or not remove_duplicate:
                retrieved_nodes = cls.parallel_retrieve(retriever_lis, [enhanced_question])
                retrieved_nodes = [set_node_turn_n(node, turn) for node in retrieved_nodes]
                nodes += retrieved_nodes

            if remove_duplicate:
                sub_ids = get_sub_ids(nodes, index_lis, is_all=is_all)
                for ret in retriever_lis:
                    ret.back_to_original_ids()

            # Generate context and enhance question using LLM with multi-agent debate
            schemas_context = parse_schema_from_df(parse_schemas_from_nodes(nodes, **kwargs))
            analysis = llm.complete(JUDGE_TEMPLATE.format(question=question, context=schemas_context)).text
            annotation = llm.complete(ANNOTATOR_TEMPLATE.format(question=question, analysis=analysis)).text
            enhanced_question = question + annotation

        if remove_duplicate:
            for ret in retriever_lis:
                ret.change_node_ids(sub_ids)

        final_nodes = cls.parallel_retrieve(retriever_lis, [enhanced_question])
        final_nodes = [set_node_turn_n(node, retrieve_turn_n) for node in final_nodes]
        nodes += final_nodes

        if remove_duplicate:
            for ret in retriever_lis:
                ret.back_to_original_ids()

        # Sort by turn number and score
        nodes.sort(key=lambda x: (x.metadata["turn_n"], x.score))

        if open_locate:
            if open_agent_debate:
                return cls.locate_with_multi_agent(llm=llm, query=question, nodes=nodes, turn_n=locate_turn_n)
            else:
                schemas = get_all_schemas_from_schema_text(nodes=nodes, output_format='schema', is_all=is_all)
                return cls.locate(llm=llm, query=question, context=schemas)
        else:
            return get_all_schemas_from_schema_text(nodes=nodes, output_format=output_format, is_all=is_all)

    @classmethod
    def load_rf_template(
            cls,
            mode: str = "agent",  # agent or pipeline
            is_single_mode: bool = True  # Single-DB / Multi-DB
    ):
        mode = mode if mode in ["agent", "pipeline"] else "agent"
        if mode == "agent":
            if is_single_mode:
                return {
                    "SOURCE_TEXT_TEMPLATE": SOURCE_TEXT_TEMPLATE,
                    "FAIR_EVAL_DEBATE_TEMPLATE": FAIR_EVAL_DEBATE_TEMPLATE,
                    "DATA_ANALYST_ROLE_DESCRIPTION": DATA_ANALYST_ROLE_DESCRIPTION,
                    "DATABASE_SCIENTIST_ROLE_DESCRIPTION": DATABASE_SCIENTIST_ROLE_DESCRIPTION,
                    "SUMMARY_TEMPLATE": SUMMARY_TEMPLATE
                }
            else:
                return {
                    "SOURCE_TEXT_TEMPLATE": MULTI_SOURCE_TEXT_TEMPLATE,
                    "FAIR_EVAL_DEBATE_TEMPLATE": MULTI_FAIR_EVAL_DEBATE_TEMPLATE,
                    "DATA_ANALYST_ROLE_DESCRIPTION": MULTI_DATA_ANALYST_ROLE_DESCRIPTION,
                    "DATABASE_SCIENTIST_ROLE_DESCRIPTION": MULTI_DATABASE_SCIENTIST_ROLE_DESCRIPTION,
                    "SUMMARY_TEMPLATE": MULTI_SUMMARY_TEMPLATE
                }
        else:
            if is_single_mode:
                return {
                    "LOCATE_TEMPLATE": LOCATE_TEMPLATE
                }
            else:
                return {
                    "LOCATE_TEMPLATE": MULTI_LOCATE_TEMPLATE
                }

    @classmethod
    def locate(
            cls,
            llm=None,
            query: str = None,
            context_str: str = None,  # 检索的所有数据库schema
            is_single_mode: bool = True,
            **kwargs
    ) -> str:
        """
            Step two: isolate irrelevant schema information.
            Mode: Pipeline
        """
        if not query:
            raise Exception("输入的查询不能为空！")

        if not llm:
            raise Exception("The llm cannot be empty!")

        prompt_loader = cls.load_rf_template(mode='pipeline', is_single_mode=is_single_mode)
        prompt = prompt_loader['LOCATE_TEMPLATE'].format(question=query, context=context_str)

        # print(prompt)
        database = llm.complete(prompt=prompt).text  # 增强后的问题查询
        #
        return database

    @classmethod
    def locate_with_multi_agent(
            cls,
            llm=None,
            turn_n: int = 2,
            query: str = None,
            nodes: List[NodeWithScore] = None,
            context_lis: List[str] = None,
            context_str: str = None,
            is_single_mode: bool = True,
            **kwargs
    ) -> str:
        """
            Step two: isolate irrelevant schema information.
            Mode: Agent
        """
        if not query:
            raise Exception("The query cannot be empty!")

        if not llm:
            raise Exception("The llm cannot be empty!")

        prompt_loader = cls.load_rf_template(mode='agent', is_single_mode=is_single_mode)

        if context_str or context_lis:
            pass
        elif nodes:
            context_lis = get_all_schemas_from_schema_text(nodes, output_format="schema", schemas_format="list")
        else:
            raise Exception("输入参数中没有包含 database schemas")

        if not context_str:
            context_str = ""
            for ind, context in enumerate(context_lis):
                context_str += f"""[The Start of Candidate Database"{ind + 1}"'s Schema]
{context}
[The End of Candidate Database"{ind + 1}"'s Schema]
                    """
        source_text = prompt_loader['SOURCE_TEXT_TEMPLATE'].format(query=query, context_str=context_str)

        chat_history = []

        # one-by-one
        for i in range(turn_n):
            data_analyst_prompt = prompt_loader['FAIR_EVAL_DEBATE_TEMPLATE'].format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=prompt_loader['DATA_ANALYST_ROLE_DESCRIPTION'],
                agent_name="data analyst"
            )
            data_analyst_debate = llm.complete(data_analyst_prompt).text
            chat_history.append(
                f'[Debate Turn: {i + 1}, Agent Name:"data analyst", Debate Content:{data_analyst_debate}]')

            data_scientist_prompt = prompt_loader['FAIR_EVAL_DEBATE_TEMPLATE'].format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=prompt_loader['DATABASE_SCIENTIST_ROLE_DESCRIPTION'],
                agent_name="database scientist"
            )
            data_scientist_debate = llm.complete(data_scientist_prompt).text
            chat_history.append(
                f'[Debate Turn: {i + 1}, Agent Name:"database scientist", Debate Content:{data_scientist_debate}]')

        # print(chat_history)
        summary_prompt = prompt_loader['FAIR_EVAL_DEBATE_TEMPLATE'].format(
            source_text=source_text,
            chat_history="\n".join(chat_history),
            role_description=prompt_loader['SUMMARY_TEMPLATE'],
            agent_name="debate terminator"
        )

        database = llm.complete(summary_prompt).text

        return database

    @classmethod
    def generate_schema(
            cls,
            llm=None,
            query: str = None,
            context: str = None,
            **kwargs
    ):
        """
            Step there: extract schemas for SQL generation.
            Mode: Pipeline.
        """
        if not llm:
            raise Exception("The llm cannot be empty!")

        if not context:
            raise Exception("The context cannot be empty!")

        context_str = f"[The Start of Database Schemas]\n{context}\n[The End of Database Schemas]"
        query = SCHEMA_LINKING_MANUAL_TEMPLATE.format(few_examples=SCHEMA_LINKING_FEW_EXAMPLES,
                                                      context_str=context_str,
                                                      question=query)
        predict_schema = llm.complete(query).text

        return predict_schema

    @classmethod
    def generate_by_multi_agent(
            cls,
            llm=None,
            query: str = None,
            context: str = None,
            turn_n: int = 2,
            linker_num: int = 1,  # schema linker 角色的数量
            **kwargs
    ):
        """
            Step there: extract schemas for SQL generation.
            Mode: Agent
        """
        if not llm:
            raise Exception("The llm cannot be empty!")

        if not context:
            raise Exception("The context cannot be empty!")

        context_str = f"[The Start of Database Schemas]\n{context}\n[The End of Database Schemas]"
        source_text = GENERATE_SOURCE_TEXT_TEMPLATE.format(query=query, context_str=context_str)

        chat_history = []

        # one-by-one
        for i in range(turn_n):
            data_analyst_prompt = GENERATE_FAIR_EVAL_DEBATE_TEMPLATE.format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=GENERATE_DATA_ANALYST_ROLE_DESCRIPTION,
                agent_name="data analyst"
            )
            for j in range(linker_num):
                data_analyst_debate = llm.complete(data_analyst_prompt).text
                chat_history.append(
                    f"""[Debate Turn: {i + 1}, Agent Name:"data analyst {j}", Debate Content:{data_analyst_debate}]""")
            data_scientist_prompt = GENERATE_FAIR_EVAL_DEBATE_TEMPLATE.format(
                source_text=source_text,
                chat_history="\n".join(chat_history),
                role_description=GENERATE_DATABASE_SCIENTIST_ROLE_DESCRIPTION,
                agent_name="data scientist"
            )
            data_scientist_debate = llm.complete(data_scientist_prompt).text
            chat_history.append(
                f"""[Debate Turn: {i + 1}, Agent Name:"data scientist", Debate Content:{data_scientist_debate}]""")

        summary_prompt = GENERATE_FAIR_EVAL_DEBATE_TEMPLATE.format(
            source_text=source_text,
            chat_history="\n".join(chat_history),
            role_description=GENERATE_SUMMARY_TEMPLATE,
            agent_name="debate terminator"
        )
        schema = llm.complete(summary_prompt).text

        return schema

    @classmethod
    def retrieve_complete_selector(cls, mode: str, **kwargs):
        mode = mode if mode in ["agent", "pipeline"] else "pipeline"
        if mode == "pipeline":
            res = cls.retrieve_complete(**kwargs)
        else:
            res = cls.retrieve_complete_by_multi_agent_debate(**kwargs)
        return res

    @classmethod
    def locate_selector(cls, mode: str, **kwargs):
        mode = mode if mode in ["agent", "pipeline"] else "pipeline"
        if mode == "pipeline":
            res = cls.locate(**kwargs)
        else:
            res = cls.locate_with_multi_agent(**kwargs)
        return res

    @classmethod
    def generate_selector(cls, mode: str, **kwargs):
        mode = mode if mode in ["agent", "pipeline"] else "pipeline"
        if mode == "pipeline":
            res = cls.generate_schema(**kwargs)
        else:
            res = cls.generate_by_multi_agent(**kwargs)
        return res


def filter_nodes_by_database(
        nodes: List[NodeWithScore],
        database: Union[str, List],
        output_format: str = "str"
):
    schema_lis = []
    for node in nodes:
        file_path = node.node.metadata["file_path"]
        db = file_path.split("\\")[-1].split(".")[0].strip()
        if type(database) == str:
            if db == database:
                schema_lis.append(default_format_node_batch_fn([node.node]))
        elif type(database) == List:
            if db in database:
                schema_lis.append(default_format_node_batch_fn([node.node]))
    if output_format == "str":
        return "\n".join(schema_lis)

    return schema_lis


def get_all_schemas_from_schema_text(
        nodes: List[NodeWithScore],
        output_format: str = "database",  # database or schema or node
        schemas_format: str = "str",  # 当输出格式为 node 时无效
        is_all: bool = True
):
    if output_format == "node":
        return nodes

    databases = []

    for node in nodes:
        file_path = node.node.metadata["file_path"]
        db = file_path.split("\\")[-1].split(".")[0].strip()
        databases.append(db)

    databases = list(set(databases))

    if output_format == "database":
        return databases

    if is_all:
        schemas = []
        for path in [node.node.metadata["file_path"] for node in nodes]:
            with open(path, "r", encoding="utf-8") as file:
                schema = file.read().strip()
                schemas.append(schema)

        if schemas_format == "str":
            schemas = "\n".join(schemas)
    else:
        summary_nodes = nodes
        fmt_node_txts = []
        for idx in range(len(summary_nodes)):
            file_path = summary_nodes[idx].node.metadata["file_path"]
            db = file_path.split("\\")[-1].split(".")[0].strip()
            fmt_node_txts.append(
                f"### Database Name: {db}\n#Following is the table creation statement for the database {db}\n"
                f"{summary_nodes[idx].get_content(metadata_mode=MetadataMode.LLM)}"
            )
        schemas = "\n\n".join(fmt_node_txts)

    if output_format == "all":
        return databases, schemas, nodes
    else:
        return schemas


def get_sub_ids(
        nodes: List[NodeWithScore],
        index_lis: List[VectorStoreIndex],
        is_all: bool = True
):
    if is_all:
        file_name_lis = []
        for node in nodes:
            file_name = node.node.metadata["file_name"]
            file_name_lis.append(file_name)

        sub_ids = []
        duplicate_ids = []
        for index in index_lis:
            doc_info_dict = index.ref_doc_info
            for key, ref_doc_info in doc_info_dict.items():
                if ref_doc_info.metadata["file_name"] not in file_name_lis:
                    sub_ids.extend(ref_doc_info.node_ids)
                else:
                    duplicate_ids.extend(ref_doc_info.node_ids)

        return sub_ids
    else:
        exist_node_ids = [node.node.id_ for node in nodes]
        all_ids = []
        for index in index_lis:
            doc_info_dict = index.ref_doc_info
            for key, ref_doc_info in doc_info_dict.items():
                all_ids.extend(ref_doc_info.node_ids)
        sub_ids = [id_ for id_ in all_ids if id_ not in exist_node_ids]

        return sub_ids


def get_ids_from_source(
        source: Union[List[VectorStoreIndex], List[NodeWithScore]]
):
    node_ids = []
    for data in source:
        if isinstance(data, VectorStoreIndex):
            doc_info_dict = data.ref_doc_info
            for key, ref_doc_info in doc_info_dict.items():
                node_ids.extend(ref_doc_info.node_ids)

        elif isinstance(data, NodeWithScore):

            node_ids.append(data.node.node_id)

    # 去重
    node_ids = list(set(node_ids))

    return node_ids
