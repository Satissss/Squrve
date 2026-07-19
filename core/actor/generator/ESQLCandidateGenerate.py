import json
import os
import re
import logging
from os import PathLike
from pathlib import Path
from typing import Union, List, Optional
from func_timeout import func_timeout, FunctionTimedOut

from core.actor.generator.BaseGenerate import BaseGenerator
from core.actor.prompt.esql_prompts import (
    CANDIDATE_SQL_GENERATION_TEMPLATE,
    fill_candidate_sql_prompt_template,
    question_relevant_descriptions_prep,
    db_column_meaning_prep,
    sql_generation_and_refinement_few_shot_prep,
)
from core.esql_utils.esql_db_utils import (
    execute_sql,
    get_schema,
    get_schema_tables_and_columns_dict,
)
from core.esql_utils.esql_retrieval_utils import (
    get_relevant_db_descriptions,
    get_db_column_meanings,
)
from core.data_manage import Dataset
from loguru import logger


@BaseGenerator.register_actor
class ESQLCandidateGenerator(BaseGenerator):
    NAME = "ESQLCandidateGenerator"

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        llm=None,
        is_save: bool = True,
        save_dir: Union[str, PathLike] = "../files/pred_sql",
        few_shot_data_path: str = "../few-shot-data/question_enrichment_few_shot_examples.json",
        generation_level_shot_number: int = 3,
        generation_few_shot_schema_existance: bool = False,
        db_sample_limit: int = 10,
        relevant_description_number: int = 20,
        seed: int = 42,
        dataset_path: str = None,
        mode: str = "dev",
        **kwargs
    ):
        super().__init__()
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir
        self.few_shot_data_path = few_shot_data_path
        self.generation_level_shot_number = generation_level_shot_number
        self.generation_few_shot_schema_existance = generation_few_shot_schema_existance
        self.db_sample_limit = db_sample_limit
        self.relevant_description_number = relevant_description_number
        self.seed = seed
        self.dataset_path = dataset_path
        self.mode = mode

        if self.dataset_path is None:
            self.dataset_path = self.dataset.db_path if self.dataset else None

    CSG_SYSTEM_PROMPT = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, samples and evidence. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."

    def _get_llm_response(self, prompt: str, temperature: float = 0.0) -> str:
        try:
            return self.llm.complete(
                prompt,
                temperature=temperature,
                system_prompt=self.CSG_SYSTEM_PROMPT,
                response_format={"type": "json_object"},
            ).text
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return ""

    def _safe_get_db_descriptions(self, db_id: str, question: str) -> str:
        candidate_paths = [
            str(Path(self.dataset_path) / db_id / "database_description"),
            str(Path(self.dataset_path).parent / self.mode / f"{self.mode}_databases" / db_id / "database_description"),
            str(Path(self.dataset_path).parent / self.mode / "database" / db_id / "database_description"),
        ]
        for ddir in candidate_paths:
            if os.path.exists(ddir):
                try:
                    return question_relevant_descriptions_prep(
                        database_description_path=ddir,
                        question=question,
                        relevant_description_number=self.relevant_description_number
                    )
                except Exception:
                    pass
        return ""

    def _safe_get_column_meanings(self, db_id: str) -> str:
        candidate_paths = [
            str(Path(self.dataset_path).parent / self.mode / "column_meaning.json"),
            str(Path(self.dataset_path).parent / "column_meaning.json"),
        ]
        for cpath in candidate_paths:
            if os.path.exists(cpath):
                try:
                    return db_column_meaning_prep(cpath, db_id)
                except Exception:
                    pass
        return ""

    def _get_db_samples(self, db_path: str) -> str:
        try:
            from core.esql_utils.esql_db_utils import get_db_tables, get_db_columns_of_table
            tables = get_db_tables(db_path)
            samples = []
            for table in tables:
                try:
                    rows = execute_sql(db_path, f"SELECT * FROM `{table}` LIMIT {self.db_sample_limit}", fetch="all")
                    if rows:
                        columns = get_db_columns_of_table(db_path, table)
                        samples.append(f"## Table {table}:")
                        header = ", ".join(columns)
                        samples.append(f"Columns: {header}")
                        for row in rows:
                            samples.append(str(row))
                except Exception:
                    pass
            return "\n".join(samples)
        except Exception as e:
            logger.warning(f"Error getting DB samples: {e}")
            return ""

    @staticmethod
    def _clean_sql(sql: str) -> str:
        sql = re.sub(r'^[\"\\\s]+|[\"\\\}\s]+$', '', sql).strip()
        sql = re.sub(r'^```(?:sql)?\s*\n?', '', sql)
        sql = re.sub(r'\n?\s*```\s*$', '', sql)
        sql = re.sub(r'\s*\}\s*$', '', sql)
        return sql.strip()

    def act(self, item, schema=None, schema_links=None, data_logger=None, **kwargs):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        logger.info(f"ESQLCandidateGenerator starting | item={item}")

        row = self.dataset[item]
        question = row['question']
        db_id = row['db_id']
        evidence = row.get('evidence', 'None')
        if not evidence or evidence == '':
            evidence = 'None'
        db_type = row.get('db_type', 'sqlite')

        if db_type == 'sqlite':
            db_path = str(Path(self.dataset_path) / (db_id + ".sqlite")) if self.dataset_path else None
        else:
            db_path = self.dataset_path

        if not db_path:
            raise ValueError(f"Database path is required for {db_type} database")

        # Step 1: Get database schema
        original_schema_dict = get_schema_tables_and_columns_dict(db_path=db_path)
        db_schema = get_schema(db_path)

        # Step 2: Get database descriptions (column descriptions via BM25)
        db_descriptions = self._safe_get_db_descriptions(db_id, question)

        # Step 3: Get column meanings
        db_column_meanings = self._safe_get_column_meanings(db_id)
        db_descriptions = db_descriptions + "\n\n" + db_column_meanings

        # Step 4: Get database samples
        db_samples = self._get_db_samples(db_path)

        # Step 5: Prepare few-shot examples
        import random
        random.seed(self.seed)
        few_shot_examples = sql_generation_and_refinement_few_shot_prep(
            few_shot_data_path=self.few_shot_data_path,
            q_db_id=db_id,
            level_shot_number=self.generation_level_shot_number,
            schema_existance=self.generation_few_shot_schema_existance,
            mode=self.mode
        )

        # Step 6: Build and send CSG prompt
        prompt = fill_candidate_sql_prompt_template(
            template=CANDIDATE_SQL_GENERATION_TEMPLATE,
            schema=db_schema,
            db_samples=db_samples,
            question=question,
            few_shot_examples=few_shot_examples,
            evidence=evidence,
            db_descriptions=db_descriptions,
        )

        logger.info(f"CSG prompt prepared for question: {question[:80]}...")
        if data_logger:
            data_logger.info(f"{self.NAME}.csg_prompt_length | len={len(prompt)}")

        response = self._get_llm_response(prompt, temperature=0.0)

        # Step 7: Parse response
        possible_sql = ""
        exec_err = ""
        try:
            response_json = json.loads(response)
            reasoning = response_json.get("chain_of_thought_reasoning", "")
            possible_sql = response_json.get("SQL", "")
            possible_sql = self._clean_sql(possible_sql)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse CSG response as JSON, trying raw extraction: {response[:200]}")
            if "#SQL:" in response:
                possible_sql = response.split("#SQL:")[-1].strip()
                possible_sql = self._clean_sql(possible_sql)
            elif "SQL" in response:
                possible_sql = response.split("SQL")[-1].replace(":", "").strip()
                possible_sql = self._clean_sql(possible_sql)
            else:
                possible_sql = "SELECT 1"
                possible_sql = self._clean_sql(possible_sql)

        # Step 8: Execute candidate SQL to check for errors
        if possible_sql and possible_sql != "SELECT 1":
            try:
                func_timeout(30, execute_sql, args=(db_path, possible_sql))
            except FunctionTimedOut:
                exec_err = "timeout"
            except Exception as e:
                exec_err = str(e)

        # Save output
        possible_sql = self.save_output(possible_sql, item, row.get("instance_id"))

        logger.info(f"ESQLCandidateGenerator completed | SQL: {possible_sql[:100]}...")
        if data_logger:
            data_logger.info(f"{self.NAME}.possible_sql | sql={possible_sql}")
            data_logger.info(f"{self.NAME}.exec_err | err={exec_err}")
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return possible_sql