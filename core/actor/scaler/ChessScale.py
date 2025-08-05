import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Optional, Dict
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.scaler.BaseScale import BaseScaler
from core.data_manage import Dataset
from core.utils import parse_schema_from_df, load_dataset, save_dataset
from llama_index.core.llms.llm import LLM

class ChessScaler(BaseScaler):
    """Scaler implementation based on CHESS-SQL's candidate generation for producing multiple SQL candidates."""

    NAME = "ChessScaler"

    CANDIDATE_TEMPLATE = '''You are an experienced database expert.
Now you need to generate a SQL query given the database information, a question and some additional information.

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "recursive divide-and-conquer approach to SQL query generation from natural language".

Database admin instructions:
1. **SELECT Clause:** Only select columns mentioned in the user's question.
2. **Aggregation (MAX/MIN):** Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:** Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC`.
4. **Handling NULLs:** If a column may contain NULL values, use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:** Only include tables essential to answer the question.
6. **Strictly Follow Hints:** Adhere to all provided hints.
7. **Thorough Question Analysis:** Address all conditions mentioned in the question.
8. **DISTINCT Keyword:** Use `SELECT DISTINCT` when the question requires unique values.
9. **Column Selection:** Carefully analyze column descriptions and hints to choose the correct column.
10. **JOIN Preference:** Prioritize `INNER JOIN` over nested `SELECT` statements.
11. **SQLite Functions Only:** Use only functions available in SQLite.

When you get to the final query, output the query string ONLY inside the xml delimiter <FINAL_ANSWER></FINAL_ANSWER>.

【Database Info】
{DATABASE_SCHEMA}

【Question】
Question: {QUESTION}

Evidence: {HINT}

【Answer】
'''

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Union[LLM, List[LLM]] = None,
            generate_num: int = 10,
            temperature: float = 0.5,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            open_parallel: bool = True,
            max_workers: int = None,
            **kwargs
    ):
        self.dataset = dataset
        self.llm = llm if isinstance(llm, list) else [llm]
        self.generate_num = generate_num
        self.temperature = temperature
        self.is_save = is_save
        self.save_dir = save_dir
        self.open_parallel = open_parallel
        self.max_workers = max_workers

    def _extract_keywords(self, question: str) -> List[str]:
        """Extract keywords from the question using LLM"""
        prompt = f"""Extract key entities and concepts from the following question that are relevant for SQL query generation. 
        Focus on table names, column names, conditions, and operations.

        Question: {question}

        Return only a Python list of keywords, for example: ['customer', 'name', 'age', '>', '30']"""
        
        try:
            response = self.llm[0].complete(prompt, temperature=0.2).text
            match = re.search(r'\[.*\]', response)
            if match:
                keywords_str = match.group(0)
                keywords = eval(keywords_str)
                return keywords if isinstance(keywords, list) else []
            return []
        except Exception as e:
            logger.warning(f"Failed to extract keywords: {e}")
            return []

    def _retrieve_context(self, question: str, schema: str, keywords: List[str]) -> str:
        """Retrieve relevant context from schema based on keywords"""
        if not keywords:
            return schema
            
        relevant_tables = []
        schema_lines = schema.split('\n')
        
        for line in schema_lines:
            line_lower = line.lower()
            if any(keyword.lower() in line_lower for keyword in keywords):
                relevant_tables.append(line)
        
        if relevant_tables:
            return '\n'.join(relevant_tables)
        return schema

    def _generate_single_candidate(self, llm_: LLM, question: str, schema: str, evidence: str) -> Optional[str]:
        """Generate a single SQL candidate"""
        try:
            prompt = self.CANDIDATE_TEMPLATE.format(
                DATABASE_SCHEMA=schema,
                QUESTION=question,
                HINT=evidence
            )
            
            response = llm_.complete(prompt, temperature=self.temperature).text
            
            sql_match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', response, re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip()
            else:
                lines = response.split('\n')
                for line in lines:
                    if line.strip().upper().startswith('SELECT'):
                        return line.strip()
            return None
        except Exception as e:
            logger.warning(f"Failed to generate candidate: {e}")
            return None

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            sub_questions: Union[str, List[str]] = None,
            **kwargs
    ) -> List[str]:
        row = self.dataset[item]
        question = row['question']
        evidence = row.get('evidence', '') or kwargs.get('evidence', '')

        # Load and process schema
        if isinstance(schema, (str, Path)):
            schema = load_dataset(schema)

        if schema is None:
            instance_schema_path = row.get("instance_schemas")
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
            if schema is None:
                schema = self.dataset.get_db_schema(item)
            if schema is None:
                raise Exception("Failed to load a valid database schema for the sample!")

        if isinstance(schema, dict):
            schema = pd.DataFrame(schema)  # Assuming single_central_process equivalent
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            schema = parse_schema_from_df(schema)
        else:
            raise Exception("Failed to load a valid database schema for the sample!")

        # Information Retrieval
        keywords = self._extract_keywords(question)
        context = self._retrieve_context(question, schema, keywords)

        # Generate candidates
        def process_serial():
            pred_sqls = []
            for i in range(self.generate_num):
                llm_ = self.llm[i % len(self.llm)]
                sql = self._generate_single_candidate(llm_, question, context, evidence)
                if sql:
                    pred_sqls.append(sql)
            return pred_sqls

        def process_parallel():
            pred_sqls = []
            with ThreadPoolExecutor(max_workers=self.max_workers or self.generate_num) as executor:
                futures = []
                for i in range(self.generate_num):
                    llm_ = self.llm[i % len(self.llm)]
                    futures.append(executor.submit(self._generate_single_candidate, llm_, question, context, evidence))
                for future in as_completed(futures):
                    try:
                        sql = future.result()
                        if sql:
                            pred_sqls.append(sql)
                    except Exception as e:
                        logger.warning(f"Error in parallel generation: {e}")
            return pred_sqls

        pred_sqls = process_parallel() if self.open_parallel else process_serial()

        # Deduplicate
        pred_sqls = list(dict.fromkeys(pred_sqls))

        if self.is_save:
            instance_id = row.get("instance_id")
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            
            # Save each SQL candidate in separate files
            sql_paths = []
            for i, sql in enumerate(pred_sqls):
                sql_save_path = save_path / f"{self.NAME}_{instance_id}_{i}.sql"
                save_dataset(sql, new_data_source=sql_save_path)
                sql_paths.append(str(sql_save_path))
            
            # Set dataset field - single path if one SQL, list of paths if multiple
            if len(sql_paths) == 1:
                self.dataset.setitem(item, "pred_sql", sql_paths[0])
            else:
                self.dataset.setitem(item, "pred_sql", sql_paths)

        return pred_sqls 