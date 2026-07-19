import json
import os
import re
import sqlite3
from os import PathLike
from pathlib import Path
from typing import Union, List, Optional
from func_timeout import func_timeout, FunctionTimedOut

from core.actor.optimizer.BaseOptimize import BaseOptimizer
from core.actor.prompt.esql_prompts import (
    SQL_REFINEMENT_TEMPLATE,
    fill_sql_refinement_prompt_template,
    sql_possible_conditions_prep,
    question_relevant_descriptions_prep,
    db_column_meaning_prep,
)
from core.esql_utils.esql_db_utils import (
    execute_sql,
    get_schema,
    get_schema_tables_and_columns_dict,
    collect_possible_conditions,
    clean_sql,
)
from core.esql_utils.esql_retrieval_utils import (
    get_relevant_db_descriptions,
)
from core.data_manage import Dataset, load_dataset
from core.utils import sql_clean
from loguru import logger


@BaseOptimizer.register_actor
class ESQLRefineOptimizer(BaseOptimizer):
    NAME = "ESQLRefineOptimizer"

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        llm=None,
        is_save: bool = True,
        save_dir: Union[str, PathLike] = "../files/optimized_sql",
        open_parallel: bool = True,
        max_workers: Optional[int] = None,
        db_sample_limit: int = 10,
        relevant_description_number: int = 20,
        seed: int = 42,
        dataset_path: str = None,
        mode: str = "dev",
        **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.db_sample_limit = db_sample_limit
        self.relevant_description_number = relevant_description_number
        self.seed = seed
        self.dataset_path = dataset_path
        self.mode = mode

        if self.dataset_path is None:
            self.dataset_path = self.dataset.db_path if self.dataset else None

    SR_SYSTEM_PROMPT = "You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, evidence, possible SQL and possible conditions. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question."

    def _get_llm_response(self, prompt: str, temperature: float = 0.0) -> str:
        try:
            result = self.llm.complete(
                prompt,
                temperature=temperature,
                system_prompt=self.SR_SYSTEM_PROMPT,
                response_format={"type": "json_object"},
            )
            if result is None or not result.text:
                logger.warning(f"SR LLM returned empty response. prompt_len={len(prompt)}")
                return ""
            logger.info(f"SR LLM response length: {len(result.text)}")
            return result.text
        except Exception as e:
            logger.error(f"Error getting SR LLM response: {e}")
            import traceback
            traceback.print_exc()
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

    def optimize_single_sql(
        self,
        sql: str,
        question: str,
        schema: str,
        db_type: str,
        schema_links: Union[str, List] = "None",
        db_id: Optional[str] = None,
        db_path: Optional[Union[str, Path]] = None,
        credential: Optional[dict] = None,
        **kwargs
    ) -> str:
        if db_type != "sqlite":
            logger.warning("ESQLRefineOptimizer currently optimized for SQLite only.")
            return sql

        db_path = str(db_path) if db_path else None
        sql = sql_clean(sql) if sql else sql

        row = kwargs.get("row", {})
        evidence = row.get('evidence', 'None') if row else kwargs.get("evidence", "None")
        if not evidence or evidence == '':
            evidence = 'None'

        # Get database schema
        db_schema = get_schema(db_path)

        # Get database descriptions
        db_descriptions = self._safe_get_db_descriptions(db_id, question)

        # Get column meanings and concatenate with descriptions (matching original paper)
        db_column_meanings = self._safe_get_column_meanings(db_id)
        db_descriptions = db_descriptions + "\n\n" + db_column_meanings

        # Collect possible conditions
        possible_conditions_dict_list = collect_possible_conditions(db_path=db_path, sql=sql) if sql else []
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)

        # Check execution error
        exec_err = ""
        if sql:
            try:
                func_timeout(30, execute_sql, args=(db_path, sql))
            except FunctionTimedOut:
                exec_err = "timeout"
            except Exception as e:
                exec_err = str(e)

        # Build SR prompt
        prompt = fill_sql_refinement_prompt_template(
            template=SQL_REFINEMENT_TEMPLATE,
            schema=db_schema,
            question=question,
            evidence=evidence,
            possible_sql=sql,
            possible_conditions=possible_conditions,
            exec_err=exec_err,
            db_descriptions=db_descriptions,
        )

        logger.info(f"SR prompt prepared for question: {question[:80]}...")

        response = self._get_llm_response(prompt, temperature=0.0)
        if not response:
            logger.warning(f"SR LLM empty response for question: {question[:80]}")
            return sql

        # Strip markdown code block markers before JSON parsing
        response = re.sub(r'^```(?:json)?\s*\n?', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n?\s*```\s*$', '', response, flags=re.MULTILINE)
        response = response.strip()

        # Parse response
        refined_sql = sql
        try:
            response_json = json.loads(response)
            refined_sql = response_json.get("SQL", sql)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse SR response as JSON: {response[:200]}")
            if "#SQL:" in response:
                refined_sql = response.split("#SQL:")[-1].strip()
            elif "SQL" in response:
                refined_sql = response.split("SQL")[-1].replace(":", "").strip()

        refined_sql = re.sub(r'^[\"\\\s]+|[\"\\\}\s]+$', '', refined_sql).strip()
        refined_sql = sql_clean(refined_sql) if refined_sql else sql
        return refined_sql

    def act(
        self,
        item,
        schema=None,
        schema_links=None,
        pred_sql=None,
        data_logger=None,
        **kwargs
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        logger.info(f"ESQLRefineOptimizer processing item {item}")

        if self.dataset is None:
            raise ValueError("Dataset is required for ESQLRefineOptimizer")

        row = self.dataset[item]
        question = row.get('enriched_question') or row['question']
        db_type = row.get('db_type', 'sqlite')
        db_id = row.get("db_id")
        db_path = str(Path(self.dataset_path) / (db_id + ".sqlite")) if self.dataset_path and db_type == "sqlite" else None

        try:
            schema = self.process_schema(schema, item)
        except Exception as e:
            logger.error(f"SR process_schema failed: {e}")
            import traceback
            traceback.print_exc()
            schema = ""

        if schema_links is None:
            schema_links = row.get("schema_links", "None")

        try:
            sql_list, is_single = self.load_pred_sql(pred_sql, item)
        except Exception as e:
            logger.error(f"SR load_pred_sql failed: {e}")
            import traceback
            traceback.print_exc()
            sql_list, is_single = [""], True
        if data_logger:
            data_logger.info(f"{self.NAME}.input_sql_count | count={len(sql_list)}")

        refined_sqls = []
        for sql in sql_list:
            refined = self.optimize_single_sql(
                sql=sql,
                question=question,
                schema=schema,
                db_type=db_type,
                schema_links=schema_links,
                db_id=db_id,
                db_path=db_path,
                row=row,
                **kwargs
            )
            refined_sqls.append(refined)

        output = self.save_output(refined_sqls, item, row.get("instance_id"))

        logger.info(f"ESQLRefineOptimizer completed processing item {item}")
        if data_logger:
            data_logger.info(f"{self.NAME}.refined_sql | output={refined_sqls}")
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return output