from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from os import PathLike
from typing import Union, List, Dict, Optional
import pandas as pd
from pathlib import Path
from loguru import logger

from core.actor.parser.BaseParse import BaseParser, parallel_slice_parse
from core.data_manage import Dataset, single_central_process
from core.utils import load_dataset, save_dataset
from core.db_connect import execute_sql
from llama_index.core.llms.llm import LLM

@BaseParser.register_actor
class CHESSSelectorParser(BaseParser):
    """
    Parser that replicates the schema selection from CHESS-SQL: filters columns, selects tables, and selects columns.
    """
    NAME = "CHESSSelectorParser"

    SKILL = """# CHESSSelectorParser

CHESS-SQL schema selection: three-stage coarse-to-fine pipeline—filter columns (per-column relevance via LLM + evidence), select tables, select columns—each stage uses question and evidence as hint. Optional parallel for column filtering. Advantage: granular pruning for large schemas; drawback: many LLM calls (one per column in step 1).

## Inputs
- `schema`: DB schema. If absent, loaded from dataset.

## Output
`schema_links`

## Steps
1. Filter columns: per-column LLM judges relevant/irrelevant (question + evidence); keep relevant only.
2. Select tables: from filtered schema, LLM picks needed tables.
3. Select columns: from selected tables, LLM picks needed columns.
4. Return `schema_links` (table.column list).
"""

    FILTER_COLUMN_TEMPLATE = """You are a detail-oriented data scientist tasked with evaluating the relevance of database column information for answering specific SQL query question based on provided hint.

Your goal is to assess whether the given column details are pertinent to constructing an SQL query to address the question informed by the hint. Label the column information as "relevant" if it aids in query formulation, or "irrelevant" if it does not.

Procedure:
1. Carefully examine the provided column details.
2. Understand the question about the database and its associated hint.
3. Decide if the column details are necessary for the SQL query based on your analysis.

Here are some examples of how to determine if the column information is relevant or irrelevant to the question and the hint:

# (Omitted examples for brevity, copy from the provided template_filter_column.txt)

# The guidelines and the rest of the prompt...
Column information:
{COLUMN_PROFILE}

Question:
{QUESTION}

HINT:
{HINT}

Take a deep breath and provide your answer in the following json format:

```json
{{
  "chain_of_thought_reasoning": "One line explanation of why or why not the column information is relevant to the question and the hint.",
  "is_column_information_relevant": "Yes" or "No"
}}
```

Only output a json as your response."""

    SELECT_TABLES_TEMPLATE = """You are an expert and very smart data analyst. 
Your task is to analyze the provided database schema, comprehend the posed question, and leverage the hint to identify which tables are needed to generate a SQL query for answering the question.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema provides a detailed definition of the database's structure, including tables, their columns, primary keys, foreign keys, and any relevant details about relationships or constraints.
For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical hint to identify the tables that will be used in the SQL query.

Question:
{QUESTION}

Hint:
{HINT}

The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Task:
Based on the database schema, question, and hint provided, your task is to determine the tables that should be used in the SQL query formulation. 
For each of the selected tables, explain why exactly it is necessary for answering the question. Your explanation should be logical and concise, demonstrating a clear understanding of the database schema, the question, and the hint.

Please respond with a JSON object structured as follows:

```json
{{
  "chain_of_thought_reasoning": "Explanation of the logical analysis that led to the selection of the tables.",
  "table_names": ["Table1", "Table2", "Table3", ...]
}}
```

Note that you should choose all and only the tables that are necessary to write a SQL query that answers the question effectively.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars. 

Only output a json as your response."""

    SELECT_COLUMNS_TEMPLATE = """You are an expert and very smart data analyst.
Your task is to examine the provided database schema, understand the posed question, and use the hint to pinpoint the specific columns within tables that are essential for crafting a SQL query to answer the question.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. Special attention should be given to the examples listed beside each column, as they directly hint at which columns are relevant to our query.

For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical hint to identify the columns that will be used in the SQL query.

Question:
{QUESTION}

Hint:
{HINT}

The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Task:
Based on the database schema, question, and hint provided, your task is to determine the columns that should be used in the SQL query formulation. 
For each of the selected columns, explain why exactly it is necessary for answering the question. Your explanation should be logical and concise, demonstrating a clear understanding of the database schema, the question, and the hint.

Please respond with a JSON object structured as follows:

```json
{{
  "chain_of_thought_reasoning": "Explanation of the logical analysis that led to the selection of the columns.",
  "table1": ["column1", "column2", "column3", ...],
  "table2": ["column1", "column2", "column3", ...],
  ...
}}
```

Note that you should choose all and only the columns that are necessary to write a SQL query that answers the question effectively.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars. 

Only output a json as your response."""

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Union[LLM, List[LLM]] = None,
            output_format: str = "list",  # output in `list` or `str`
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/schema_links",
            use_external: bool = False,
            open_parallel: bool = False,
            max_workers: int = None,
            **kwargs
    ):
        super().__init__(dataset, llm, output_format, is_save, save_dir, use_external, **kwargs)
        self.open_parallel = open_parallel
        self.max_workers = max_workers

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        try:
            external = load_dataset(external)
        except FileNotFoundError:
            logger.debug("External file not found, treat it as content.")
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def _format_column_profile(self, schema_row: pd.Series, db_type: str) -> str:
        """Format a single column profile for LLM evaluation."""
        table = schema_row['table_name']
        column = schema_row['column_name']
        description = schema_row.get('column_descriptions', column)
        data_type = schema_row.get('data_type', 'unknown')
        is_nullable = schema_row.get('is_nullable', 'unknown')
        is_primary_key = schema_row.get('is_primary_key', False)
        is_foreign_key = schema_row.get('is_foreign_key', False)
        example = schema_row.get('sample_rows', None)

        profile = f"Table: {table}\nColumn: {column}\nDescription: {description}\nData Type: {data_type}\nNullable: {is_nullable}"

        if is_primary_key:
            profile += "\nPrimary Key: Yes"
        if is_foreign_key:
            profile += "\nForeign Key: Yes"
        if example:
            profile += f"\nExample of values in the column: `{example}`"
        return profile

    def _llm_call(self, prompt: str) -> str:
        return self.llm.complete(prompt).text

    def _parse_json(self, response: str) -> Dict:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON")
        return {}

    @parallel_slice_parse
    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, data_logger=None, update_dataset=True,
            **kwargs):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        data_row = self.dataset[item]
        question = data_row["question"]
        evidence = data_row.get("evidence", "")
        db_type = data_row.get("db_type", "sqlite")

        # evidence 与 external 实为同一类先验知识，提示词使用 evidence (HINT)，故将 external 赋给 evidence
        if self.use_external:
            external_knowledge = self.load_external_knowledge(data_row.get("external", None))
            if external_knowledge:
                evidence = evidence + "\n" + external_knowledge if evidence else external_knowledge
                logger.debug("已加载外部知识")

        # Use base class method to process schema
        schema_df = self.process_schema(item, schema)

        # Step 1: Filter columns
        column_profiles = []
        for _, schema_row in schema_df.iterrows():
            profile = self._format_column_profile(schema_row, db_type)
            column_profiles.append((schema_row['table_name'], schema_row['column_name'], profile))

        def filter_single(profile_kwargs):
            table, column, profile = profile_kwargs
            prompt = self.FILTER_COLUMN_TEMPLATE.format(COLUMN_PROFILE=profile, QUESTION=question, HINT=evidence)
            response = self._llm_call(prompt)
            result = self._parse_json(response)
            is_relevant = result.get("is_column_information_relevant", "No").lower() == "yes"
            return table, column, is_relevant

        relevant_columns = {}
        if self.open_parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(filter_single, (t, c, p)) for t, c, p in column_profiles]
                for future in as_completed(futures):
                    t, c, relevant = future.result()
                    if relevant:
                        if t not in relevant_columns:
                            relevant_columns[t] = []
                        relevant_columns[t].append(c)
        else:
            for t, c, p in column_profiles:
                t, c, relevant = filter_single((t, c, p))
                if relevant:
                    if t not in relevant_columns:
                        relevant_columns[t] = []
                    relevant_columns[t].append(c)
        filtered_links = [f"{t}.{c}" for t, cols in relevant_columns.items() for c in cols]
        self.log_schema_links(data_logger, filtered_links, stage="Filter Columns")
        tentative_schema = relevant_columns

        # Step 2: Select tables
        schema_str = "\n".join([f"{table} ({', '.join(cols)})" for table, cols in tentative_schema.items()])
        prompt = self.SELECT_TABLES_TEMPLATE.format(DATABASE_SCHEMA=schema_str, QUESTION=question, HINT=evidence)
        response = self._llm_call(prompt)
        result = self._parse_json(response)
        selected_tables = result.get("table_names", [])
        if not isinstance(selected_tables, list):
            selected_tables = []
        # Only keep tables that exist in tentative_schema
        selected_tables = [t for t in selected_tables if t in tentative_schema]
        tentative_schema = {t: tentative_schema.get(t, []) for t in selected_tables}
        table_links = [f"{t}.{c}" for t, cols in tentative_schema.items() for c in cols]
        self.log_schema_links(data_logger, table_links, stage="Select tables")

        # Step 3: Select columns
        schema_str = "\n".join([f"{table} ({', '.join(cols)})" for table, cols in tentative_schema.items()])
        prompt = self.SELECT_COLUMNS_TEMPLATE.format(DATABASE_SCHEMA=schema_str, QUESTION=question, HINT=evidence)
        response = self._llm_call(prompt)
        result = self._parse_json(response)
        selected_schema = {k: v for k, v in result.items() if k != "chain_of_thought_reasoning"}

        # Convert to schema_links list
        schema_links = [f"{table}.{col}" for table, cols in selected_schema.items() for col in cols]

        # Dedup
        schema_links = list(set(schema_links))
        self.log_schema_links(data_logger, schema_links, stage="final")
        output = self.format_output(schema_links)

        # Use base class method to save output
        if update_dataset:
            self.save_output(output, item)

        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return output

    def merge_results(self, results: List):
        if not results:
            logger.info("Input results empty!")
            return []

        merge_result = []
        for row in results:
            if not isinstance(row, List):
                raise TypeError(f"Each row must be a list, but got {type(row)}: {row}")

            merge_result.extend(row)

        return merge_result
