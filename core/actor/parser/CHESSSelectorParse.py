from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from os import PathLike
from typing import Union, List, Dict, Optional
import pandas as pd
from pathlib import Path
from loguru import logger

from core.actor.parser.BaseParse import BaseParser
from core.data_manage import Dataset, single_central_process
from core.utils import load_dataset, save_dataset
from core.db_connect import execute_sql
from llama_index.core.llms.llm import LLM


class CHESSSelectorParser(BaseParser):
    """
    Parser that replicates the schema selection from CHESS-SQL: filters columns, selects tables, and selects columns.
    """
    NAME = "CHESSSelectorParser"

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
Based on the database schema, question, and hint provided, your task is to identify all and only the columns that are essential for crafting a SQL query to answer the question.
For each of the selected columns, explain why exactly it is necessary for answering the question. Your reasoning should be concise and clear, demonstrating a logical connection between the columns and the question asked.

Tip: If you are choosing a column for filtering a value within that column, make sure that column has the value as an example.


Please respond with a JSON object structured as follows:

```json
{{
  "chain_of_thought_reasoning": "Your reasoning for selecting the columns, be concise and clear.",
  "table_name1": ["column1", "column2", ...],
  "table_name2": ["column1", "column2", ...],
  ...
}}
```

Make sure your response includes the table names as keys, each associated with a list of column names that are necessary for writing a SQL query to answer the question.
For each aspect of the question, provide a clear and concise explanation of your reasoning behind selecting the columns.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars.

Only output a json as your response."""

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Union[LLM, List[LLM]] = None,
            output_format: str = "list",  # 'list' or 'str'
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/schema_links",
            generate_num: int = 1,
            open_parallel: bool = False,
            max_workers: int = None,
            db_path: Optional[Union[str, PathLike]] = None,
            credential: Optional[Dict] = None,
            **kwargs
    ):
        self.dataset = dataset
        self.llm = llm if isinstance(llm, LLM) else llm[0] if isinstance(llm, list) else None
        self.output_format = output_format
        self.is_save = is_save
        self.save_dir = save_dir
        self.generate_num = generate_num
        self.open_parallel = open_parallel
        self.max_workers = max_workers
        self.db_path = db_path or (dataset.db_path if dataset else None)
        self.credential = credential or (dataset.credential if dataset else None)

    def _format_column_profile(self, row: pd.Series, db_type: str = "sqlite") -> str:
        profile = f"Table name: `{row['table_name']}`\nOriginal column name: `{row['column_name']}`\nData type: {row.get('data_type', 'UNKNOWN')}"
        if 'column_description' in row:
            profile += f"\nDescription: {row['column_description']}"
        if 'value_description' in row:
            profile += f"\nValue description: {row['value_description']}"
        example = ''
        if self.db_path:
            try:
                query = f"SELECT {row['column_name']} FROM {row['table_name']} LIMIT 1"
                result = execute_sql(db_type, self.db_path, query, self.credential)
                if result and not isinstance(result, str):  # Check if result is not an error message
                    example = str(result)
            except Exception as e:
                logger.warning(f"Failed to fetch example for {row['table_name']}.{row['column_name']}: {e}")
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

    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, **kwargs):
        data_row = self.dataset[item]
        question = data_row["question"]
        evidence = data_row.get("evidence", "")
        db_size = data_row.get("db_size", 0)
        db_type = data_row.get("db_type", "sqlite")

        if isinstance(schema, (str, PathLike)):
            schema = load_dataset(schema)
        if schema is None:
            schema = self.dataset.get_db_schema(item)
        if schema is None:
            raise Exception("Failed to load schema")
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)
        if not isinstance(schema, pd.DataFrame):
            raise Exception("Schema must be a DataFrame")

        # Step 1: Filter columns
        column_profiles = []
        for _, schema_row in schema.iterrows():
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

        tentative_schema = relevant_columns

        # Step 2: Select tables
        schema_str = "\n".join([f"{table} ({', '.join(cols)})" for table, cols in tentative_schema.items()])
        prompt = self.SELECT_TABLES_TEMPLATE.format(DATABASE_SCHEMA=schema_str, QUESTION=question, HINT=evidence)
        response = self._llm_call(prompt)
        result = self._parse_json(response)
        selected_tables = result.get("table_names", [])
        tentative_schema = {t: tentative_schema.get(t, []) for t in selected_tables}

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

        if self.is_save:
            instance_id = data_row['instance_id']
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            save_path = save_path / f"{self.NAME}_{instance_id}.json"
            save_dataset(schema_links, new_data_source=save_path)
            self.dataset.setitem(item, "schema_links", str(save_path))

        return schema_links if self.output_format == "list" else str(schema_links)
