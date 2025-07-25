import os
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict
from loguru import logger
import re
from os import PathLike

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, load_dataset, save_dataset
from core.utils import parse_schema_from_df
from core.db_connect import execute_sql  # Assuming db_connect.py has an execute_sql function


# Note: This implementation adapts ReFoRCE's logic to Squrve's framework.
# We use the LLM for prompt responses and assume execute_sql is available for feedback.

class ReFoRCEGenerator(BaseGenerator):
    OUTPUT_NAME = "pred_sql"
    NAME = "ReFoRCEGenerator"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            use_external: bool = True,
            use_few_shot: bool = True,
            do_column_exploration: bool = True,
            do_self_refinement: bool = True,
            max_iter: int = 5,
            max_try: int = 3,
            csv_max_len: int = 500,
            **kwargs
    ):
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir
        self.use_external = use_external
        self.use_few_shot = use_few_shot
        self.do_column_exploration = do_column_exploration
        self.do_self_refinement = do_self_refinement
        self.max_iter = max_iter
        self.max_try = max_try
        self.csv_max_len = csv_max_len
        self.empty_result = "No data found for the specified query."

    def load_external_knowledge(self, external: Union[str, Path] = None):
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def parse_sql_from_response(self, response: str) -> List[str]:
        # Parse multiple SQL queries from the LLM response, looking for ```sql blocks with optional descriptions
        sqls = []
        matches = re.findall(r'```sql\s*(--Description:.*?\n)?(.*?)```', response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            sql = match[1].strip()
            if sql:
                sqls.append(sql)
        return sqls

    def get_exploration_prompt(self, db_type, schema_str):
        # Improved prompt aligned with original ReFoRCE's exploration prompt
        prompt = f"Write at most 10 {db_type} SQL queries from simple to complex to understand values in related columns.\n"
        prompt += "Each query should be different. Use DISTINCT and LIMIT 20 rows.\n"
        prompt += "Write annotations to describe each SQL in format ```sql\n--Description: \n```.\n"
        prompt += "For string-matching, use ILIKE or LIKE appropriately for {db_type}.\n"  # Add dialect-specific notes
        prompt += "Do not query schema or data types. Focus on SELECT queries only.\n"
        return prompt

    def execute_sqls(self, sqls, db_type, db_path, credential, logger):
        result_dic_list = []
        corrected_count = 0
        for i, sql in enumerate(sqls):
            results = execute_sql(db_type, db_path, sql, credential)
            if isinstance(results, str) and results != self.empty_result and 'ERROR' not in results.upper():
                result_dic_list.append({'sql': sql, 'res': results})
            else:
                corrected_sql = self.self_correct(sql, results, logger, simplify=(results == self.empty_result))
                if corrected_sql:
                    results = execute_sql(db_type, db_path, corrected_sql, credential)
                    if isinstance(results, str) and results != self.empty_result and 'ERROR' not in results.upper():
                        result_dic_list.append({'sql': corrected_sql, 'res': results})
                        corrected_count += 1
                        # Refine remaining SQLs if corrected
                        if i < len(sqls) - 1:
                            refine_prompt = f"Original SQL: {sql}\nCorrected to: {corrected_sql}\nRefine the following SQLs based on this correction:\n" + '\n'.join(sqls[i+1:])
                            response = self.llm.complete(refine_prompt).text
                            refined_sqls = self.parse_sql_from_response(response)
                            if refined_sqls:
                                sqls = sqls[:i+1] + refined_sqls
        return result_dic_list

    def self_correct(self, sql, error, logger, simplify=False):
        # Improved self-correct prompt aligned with original
        prompt = f"Input sql:\n{sql}\nThe error information is:\n{str(error)}\n"
        if simplify:
            prompt += "Since the output is empty, please simplify some conditions in the SQL.\n"
        prompt += "Please correct it and output only one complete SQL query with thinking process in ```sql``` format."
        response = self.llm.complete(prompt).text
        corrected = self.parse_sql_from_response(response)
        if corrected:
            return corrected[0]  # Take the first corrected SQL
        return None

    def exploration(self, question, schema_str, db_type, db_path, credential, logger):
        prompt = self.get_exploration_prompt(db_type, schema_str)
        response = self.llm.complete(prompt).text
        sqls = self.parse_sql_from_response(response)
        if not sqls:
            logger.warning("No SQLs parsed from exploration response.")
            return ""
        pre_info = ""
        results = self.execute_sqls(sqls, db_type, db_path, credential, logger)
        for dic in results:
            pre_info += f"Query:\n{dic['sql']}\nAnswer:\n{dic['res']}\n"
        return pre_info

    def self_refine(self, question, schema_str, pre_info, db_type, db_path, credential, logger):
        iter_count = 0
        generated_sqls = []  # Collect multiple candidates for voting
        error_history = []
        while iter_count < self.max_iter:
            prompt = f"Database schema:\n{schema_str}\n"
            if pre_info:
                prompt += f"Exploration results:\n{pre_info}\n"
            prompt += f"Question: {question}\n"
            prompt += f"Generate one {db_type} SQL query. Think step by step."
            if error_history:
                prompt += "\nAvoid previous errors: " + ', '.join(error_history[-3:])
            response = self.llm.complete(prompt).text
            sqls = self.parse_sql_from_response(response)
            if not sqls:
                iter_count += 1
                continue
            sql = sqls[0]
            executed_result = execute_sql(db_type, db_path, sql, credential)
            error_history.append(str(executed_result))
            if len(error_history) > 3 and all(e == self.empty_result for e in error_history[-4:]):
                logger.info("Early stop due to repeated empty results.")
                break
            if isinstance(executed_result, str) and executed_result != self.empty_result and 'ERROR' not in executed_result.upper():
                # Check for empty columns or nested values
                refine_prompt = f"Input sql:\n{sql}\nError: {executed_result}\nCorrect it and output one SQL."
                if executed_result == self.empty_result:
                    refine_prompt += "\nSimplify conditions as results are empty."
                response = self.llm.complete(refine_prompt).text
                refined_sqls = self.parse_sql_from_response(response)
                if refined_sqls:
                    sql = refined_sqls[0]
                    executed_result = execute_sql(db_type, db_path, sql, credential)
                    error_history.append(str(executed_result))
                    if isinstance(executed_result, str) and executed_result != self.empty_result and 'ERROR' not in executed_result.upper():
                        generated_sqls.append((sql, executed_result))
            iter_count += 1
        if not generated_sqls:
            return None
        # Improved voting: group by similar results
        from collections import defaultdict, Counter
        result_groups = defaultdict(list)
        for sql, res in generated_sqls:
            result_groups[res].append(sql)
        result_counts = Counter({res: len(sqls) for res, sqls in result_groups.items()})
        most_common = result_counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Tie, select random from top
            import random
            top_results = [res for res, count in most_common if count == most_common[0][1]]
            selected_res = random.choice(top_results)
        else:
            selected_res = most_common[0][0]
        return result_groups[selected_res][0]  # Return one SQL from the selected group

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            **kwargs
    ):
        if self.dataset is None or self.llm is None:
            raise ValueError("Dataset and LLM must be provided for ReFoRCEGenerator.")
        row = self.dataset[item]
        question = row['question']
        db_type = row.get('db_type', 'sqlite')  # Default to sqlite if not specified
        db_id = row.get("db_id")
        db_path = Path(self.dataset.db_path) / (
                    db_id + ".sqlite") if self.dataset.db_path and db_type == 'sqlite' else row.get('db_path')
        credential = self.dataset.credential if self.dataset else None

        if self.use_external:
            external = self.load_external_knowledge(row.get("external", None))
            if external:
                question += "\n" + external

        # Process schema
        if isinstance(schema, (str, PathLike)):
            schema = load_dataset(schema)
        if isinstance(schema, pd.DataFrame):
            schema_str = parse_schema_from_df(schema)
        elif isinstance(schema, str):
            schema_str = schema
        else:
            schema_str = str(schema)  # Fallback
        if schema_links:
            schema_str += f"\nSchema Links: {schema_links}"

        pre_info = ""
        if self.do_column_exploration:
            pre_info = self.exploration(question, schema_str, db_type, db_path, credential, logger)

        pred_sql = None
        if self.do_self_refinement:
            pred_sql = self.self_refine(question, schema_str, pre_info, db_type, db_path, credential, logger)

        if pred_sql is None:
            # Fallback generation with logging
            logger.warning("Self-refinement failed, falling back to simple generation.")
            prompt = f"Generate {db_type} SQL for: {question}\nSchema: {schema_str}"
            response = self.llm.complete(prompt).text
            sqls = self.parse_sql_from_response(response)
            pred_sql = sqls[0] if sqls else "/* No SQL generated */"

        if self.is_save:
            instance_id = row.get("instance_id", str(item))
            save_path = Path(self.save_dir) / f"{self.__class__.__name__}_{instance_id}.sql"
            save_dataset(pred_sql, save_path)
            self.dataset.setitem(item, "pred_sql", str(save_path))

        return pred_sql
