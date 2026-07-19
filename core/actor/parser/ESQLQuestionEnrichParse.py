import json
import os
import random
import re
from os import PathLike
from pathlib import Path
from typing import Union, List, Optional

from core.actor.parser.BaseParse import BaseParser
from core.actor.prompt.esql_prompts import (
    QUESTION_ENRICHMENT_TEMPLATE,
    fill_question_enrichment_prompt_template,
    sql_possible_conditions_prep,
    question_enrichment_few_shot_prep,
    question_relevant_descriptions_prep,
    db_column_meaning_prep,
)
from core.esql_utils.esql_db_utils import (
    execute_sql,
    get_schema,
    get_schema_tables_and_columns_dict,
    collect_possible_conditions,
)
from core.esql_utils.esql_retrieval_utils import (
    get_relevant_db_descriptions,
    get_db_column_meanings,
)
from core.data_manage import Dataset
from loguru import logger
from func_timeout import func_timeout, FunctionTimedOut


@BaseParser.register_actor
class ESQLQuestionEnrichParser(BaseParser):
    NAME = "ESQLQuestionEnrichParser"

    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        llm=None,
        output_format: str = "list",
        is_save: bool = True,
        save_dir: Union[str, PathLike] = "../files/schema_links",
        few_shot_data_path: str = "../few-shot-data/question_enrichment_few_shot_examples.json",
        enrichment_level: str = "complex",
        enrichment_level_shot_number: int = 3,
        enrichment_few_shot_schema_existance: bool = False,
        db_sample_limit: int = 10,
        relevant_description_number: int = 20,
        seed: int = 42,
        dataset_path: str = None,
        mode: str = "dev",
        **kwargs
    ):
        super().__init__(dataset, llm, output_format, is_save, save_dir, **kwargs)
        self.few_shot_data_path = few_shot_data_path
        self.enrichment_level = enrichment_level
        self.enrichment_level_shot_number = enrichment_level_shot_number
        self.enrichment_few_shot_schema_existance = enrichment_few_shot_schema_existance
        self.db_sample_limit = db_sample_limit
        self.relevant_description_number = relevant_description_number
        self.seed = seed
        self.dataset_path = dataset_path
        self.mode = mode

        if self.dataset_path is None:
            self.dataset_path = self.dataset.db_path if self.dataset else None

    QE_SYSTEM_PROMPT = "You are excellent data scientist and can link the information between a question and corresponding database perfectly. Your objective is to analyze the given question, corresponding database schema, database column descriptions and the evidence to create a clear link between the given question and database items which includes tables, columns and values. With the help of link, rewrite new versions of the original question to be more related with database items, understandable, clear, absent of irrelevant information and easier to translate into SQL queries. This question enrichment is essential for comprehending the question's intent and identifying the related database items. The process involves pinpointing the relevant database components and expanding the question to incorporate these items."

    def _get_llm_response(self, prompt: str, temperature: float = 0.0) -> str:
        try:
            if isinstance(self.llm, list) and self.llm:
                llm = self.llm[0]
            else:
                llm = self.llm
            return llm.complete(
                prompt,
                temperature=temperature,
                system_prompt=self.QE_SYSTEM_PROMPT,
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

    def act(self, item, schema=None, data_logger=None, update_dataset=True, **kwargs):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        logger.info(f"ESQLQuestionEnrichParser starting | item={item}")

        row = self.dataset[item]
        question = row['question']
        db_id = row['db_id']
        q_id = row.get('question_id', item)
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

        # Step 1: Get possible_sql from previous stage (CSG)
        possible_sql = row.get('pred_sql', None)
        if not possible_sql:
            logger.warning("No candidate SQL found, using empty string")
            possible_sql = ""

        # Step 2: Get database schema
        original_schema_dict = get_schema_tables_and_columns_dict(db_path=db_path)
        db_schema = get_schema(db_path)

        # Step 3: Get database descriptions
        db_descriptions = self._safe_get_db_descriptions(db_id, question)

        # Step 3b: Get column meanings and concatenate with descriptions (matching original paper)
        db_column_meanings = self._safe_get_column_meanings(db_id)
        db_descriptions = db_descriptions + "\n\n" + db_column_meanings

        # Step 4: Get database samples
        db_samples = self._get_db_samples(db_path)

        # Step 5: Collect possible conditions from candidate SQL
        possible_conditions_dict_list = []
        if possible_sql:
            try:
                cleaned_sql = re.sub(r'^[\"\\\s]+|[\"\\\}\s]+$', '', possible_sql).strip()
                possible_conditions_dict_list = collect_possible_conditions(db_path=db_path, sql=cleaned_sql)
            except Exception:
                possible_conditions_dict_list = []
        possible_conditions = sql_possible_conditions_prep(possible_conditions_dict_list=possible_conditions_dict_list)

        # Step 6: Prepare few-shot examples
        random.seed(self.seed)
        few_shot_examples = question_enrichment_few_shot_prep(
            few_shot_data_path=self.few_shot_data_path,
            q_id=q_id,
            q_db_id=db_id,
            level_shot_number=self.enrichment_level_shot_number,
            schema_existance=self.enrichment_few_shot_schema_existance,
            enrichment_level=self.enrichment_level,
            mode=self.mode
        )

        # Step 7: Build and send QE prompt
        prompt = fill_question_enrichment_prompt_template(
            template=QUESTION_ENRICHMENT_TEMPLATE,
            schema=db_schema,
            db_samples=db_samples,
            question=question,
            possible_conditions=possible_conditions,
            few_shot_examples=few_shot_examples,
            evidence=evidence,
            db_descriptions=db_descriptions,
        )

        logger.info(f"QE prompt prepared for question: {question[:80]}...")
        if data_logger:
            data_logger.info(f"{self.NAME}.qe_prompt_length | len={len(prompt)}")

        response = self._get_llm_response(prompt, temperature=0.0)

        # Step 8: Parse response
        enriched_question = question
        enrichment_reasoning = ""
        try:
            response_json = json.loads(response)
            enrichment_reasoning = response_json.get("chain_of_thought_reasoning", "")
            enriched_question_raw = response_json.get("enriched_question", "")
            if enriched_question_raw:
                enriched_question = question + " " + enrichment_reasoning + " " + enriched_question_raw
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse QE response as JSON: {response[:200]}")
            if "enriched_question" in response:
                enriched_question = response.split("enriched_question")[-1].replace(":", "").strip().strip('"').strip("'")
            enriched_question = question + " " + enriched_question

        # Step 9: Collect schema_links from enrichment (extracted columns and values)
        schema_links = []
        try:
            from core.esql_utils.esql_db_utils import get_db_tables, get_db_columns_of_table
            db_tables = get_db_tables(db_path)
            enriched_lower = enriched_question.lower()
            for table in db_tables:
                try:
                    columns = get_db_columns_of_table(db_path, table)
                    for col in columns:
                        if col.lower() in enriched_lower:
                            schema_links.append(f"{table}.{col}")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Error extracting schema links from enriched question: {e}")

        schema_links = list(set(schema_links))
        output = self.format_output(schema_links)

        self.log_schema_links(data_logger, schema_links, stage="esql_enriched")

        if update_dataset:
            self.save_output(output, item, file_ext=".json")
            self.dataset.setitem(item, "enriched_question", enriched_question)

        logger.info(f"ESQLQuestionEnrichParser completed | enriched_question: {enriched_question[:100]}...")
        if data_logger:
            data_logger.info(f"{self.NAME}.enriched_question | q={enriched_question[:200]}")
            data_logger.info(f"{self.NAME}.schema_links | links={schema_links}")
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return output