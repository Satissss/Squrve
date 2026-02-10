import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
from loguru import logger
import re

from collections import defaultdict
from os import PathLike
from llama_index.core.llms.llm import LLM

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from core.utils import parse_schema_from_df
from core.db_connect import get_sql_exec_result


# Complete ReFoRCE implementation following the original paper and source code
# Includes: Column Exploration, Self-Refinement, Majority Voting, and Consensus Enforcement
@BaseGenerator.register_actor
class ReFoRCEGenerator(BaseGenerator):
    OUTPUT_NAME = "pred_sql"
    NAME = "ReFoRCEGenerator"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            use_external: bool = True,
            use_few_shot: bool = True,
            do_column_exploration: bool = True,
            do_self_refinement: bool = True,
            do_self_consistency: bool = True,
            do_vote: bool = True,
            num_votes: int = 3,
            max_iter: int = 5,
            max_try: int = 3,
            csv_max_len: int = 500,
            temperature: float = 1.0,
            early_stop: bool = True,
            random_vote_for_tie: bool = True,
            model_vote: bool = False,
            final_choose: bool = True,
            db_path: Optional[Union[str, PathLike]] = None,
            credential: Optional[Dict] = None,
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
        self.do_self_consistency = do_self_consistency
        self.do_vote = do_vote
        self.num_votes = num_votes
        self.max_iter = max_iter
        self.max_try = max_try
        self.csv_max_len = csv_max_len
        self.temperature = temperature
        self.early_stop = early_stop
        self.random_vote_for_tie = random_vote_for_tie
        self.model_vote = model_vote
        self.final_choose = final_choose
        self.empty_result = "No data found for the specified query."

        # Safely initialize db_path and credential
        self.db_path = db_path or (self.dataset.db_path if self.dataset else None)
        self.credential = credential or (self.dataset.credential if self.dataset else None)

    def load_external_knowledge(self, external: Union[str, Path] = None):
        """Load external knowledge if available"""
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def parse_sql_from_response(self, response: str) -> List[str]:
        """Parse SQL queries from LLM response, looking for ```sql blocks"""
        sqls = []
        matches = re.findall(r'```sql\s*(--Description:.*?\n)?(.*?)```', response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            sql = match[1].strip()
            if sql and sql.upper().startswith('SELECT'):
                sqls.append(sql)
        return sqls

    def hard_cut(self, text: str, max_len: int = 500) -> str:
        """Truncate text to maximum length"""
        if len(text) > max_len:
            return text[:max_len] + "\n..."
        return text

    def get_exploration_prompt(self, db_type: str, schema_str: str) -> str:
        """Generate exploration prompt for column exploration"""
        prompt = f"Write at most 10 {db_type} SQL queries from simple to complex to understand values in related columns.\n"
        prompt += "Each query should be different. Don't query about any SCHEMA or checking data types. You can write SELECT query only.\n"
        prompt += "Try to use DISTINCT. For each SQL LIMIT 20 rows.\n"
        prompt += "Write annotations to describe each SQL in format ```sql\n--Description: \n```.\n"

        if db_type.lower() == "snowflake":
            prompt += "Use ILIKE for case-insensitive string matching. Ensure all column names are enclosed in double quotes.\n"
        elif db_type.lower() == "bigquery":
            prompt += "Use LOWER() with LIKE for case-insensitive string matching. Enclose identifiers with backticks.\n"
        elif db_type.lower() == "sqlite":
            prompt += "Use LIKE with % wildcards for string matching. Enclose identifiers with double quotes if needed.\n"

        prompt += f"You can only use tables in the provided schema.\n"
        return prompt

    def get_self_refine_prompt(self, question: str, schema_str: str, pre_info: str, db_type: str,
                               format_csv: str = None) -> str:
        """Generate self-refinement prompt"""
        prompt = f"Database schema:\n{schema_str}\n"
        if pre_info:
            prompt += f"Some few-shot examples after column exploration may be helpful:\n{pre_info}\n"

        prompt += f"Task: {question}\n"
        prompt += f"Please think step by step and answer only one complete SQL in {db_type} dialect in ```sql``` format.\n"

        if format_csv:
            prompt += f"Follow the answer format like: {format_csv}.\n"

        prompt += "Here are some useful tips for answering:\n"
        prompt += "When asked something without stating name or id, return both of them.\n"
        prompt += "When asked percentage decrease, you should return a positive value.\n"

        if db_type.lower() == "snowflake":
            prompt += "When using ORDER BY xxx DESC, add NULLS LAST to exclude null records: ORDER BY xxx DESC NULLS LAST.\n"

        return prompt

    def get_self_consistency_prompt(self, question: str, format_csv: str = None) -> str:
        """Generate self-consistency checking prompt"""
        prompt = f"Please check the answer again by reviewing task:\n{question}\n"
        prompt += "Review Relevant Tables and Columns and Possible Conditions and then give the final SQL query.\n"
        prompt += "Don't output other queries. If you think the answer is right, just output the current SQL.\n"

        if format_csv:
            prompt += f"The answer format should be like: {format_csv}\n"

        return prompt

    def execute_sql_safe(self, sql: str, db_type: str, db_path: str, credential: str = None) -> Tuple[bool, str]:
        """Execute SQL safely using Squrve's db_connect module"""
        try:
            # Map parameters per backend requirements
            args = {
                "sql_query": sql,
            }

            if db_type == "sqlite":
                args["db_path"] = db_path
            elif db_type == "snowflake":
                # For Snowflake, the function expects `db_id` instead of `db_path`
                args["db_id"] = db_path
                args["credential_path"] = credential
            elif db_type == "big_query":
                # BigQuery only needs credentials
                args["credential_path"] = credential
            else:
                # Fallback to original mapping
                args["db_path"] = db_path
                if credential is not None:
                    args["credential_path"] = credential

            exec_result = get_sql_exec_result(db_type, **args)

            if isinstance(exec_result, tuple):
                if len(exec_result) == 3:
                    res, err, _ = exec_result
                elif len(exec_result) == 2:
                    res, err = exec_result
                else:
                    res = exec_result[0]
                    err = None
            else:
                res = exec_result
                err = None

            if err:
                return False, str(err)

            if res is None or (isinstance(res, pd.DataFrame) and res.empty):
                return False, self.empty_result

            if isinstance(res, pd.DataFrame):
                csv_str = res.to_csv(index=False)
                return True, self.hard_cut(csv_str, self.csv_max_len)

            return True, self.hard_cut(str(res), self.csv_max_len)

        except Exception as e:
            return False, f"##ERROR## {str(e)}"

    def execute_sqls(self, sqls: List[str], db_type: str, db_path: str, credential: str, logger) -> List[Dict]:
        """Execute multiple SQLs for exploration with error correction"""
        result_dic_list = []
        error_rec = []

        for i, sql in enumerate(sqls):
            if len(result_dic_list) > 10:  # Limit results
                break

            logger.info(f"[Try to execute]\n{sql}\n[Try to execute]")
            success, results = self.execute_sql_safe(sql, db_type, db_path, credential)

            if success and results != self.empty_result:
                result_dic_list.append({'sql': sql, 'res': results})
                logger.info(f"[Successfully executed]\n{sql}\nResults:\n{results[:200]}...\n[Successfully executed]")
            else:
                logger.info(f"[Error occurred]\n{results}\n[Error occurred]")
                error_rec.append(0)

                # Try to correct the SQL
                max_try = self.max_try
                corrected_sql = None

                while max_try > 0:
                    simplify = (results == self.empty_result)
                    corrected_sql = self.self_correct(sql, results, simplify=simplify)

                    if not corrected_sql:
                        break

                    success, results = self.execute_sql_safe(corrected_sql, db_type, db_path, credential)
                    if success and results != self.empty_result:
                        result_dic_list.append({'sql': corrected_sql, 'res': results})
                        error_rec.append(1)
                        logger.info(f"[Successfully corrected]\n{corrected_sql}\n[Successfully corrected]")
                        break

                    max_try -= 1

                if not corrected_sql or max_try == 0:
                    error_rec.append(0)

                # Early termination if too many consecutive errors
                if len(error_rec) > 3 and sum(error_rec[-3:]) == 0:
                    logger.warning("Too many consecutive errors, stopping execution")
                    break

        return result_dic_list

    def self_correct(self, sql: str, error: str, simplify: bool = False) -> Optional[str]:
        """Self-correct SQL based on error feedback"""
        prompt = f"Input sql:\n{sql}\nThe error information is:\n{error}\n"
        if simplify:
            prompt += "Since the output is empty, please simplify some conditions of the past sql.\n"
        prompt += "Please correct it based on previous context and output the thinking process with only one sql query in ```sql``` format. Don't just analyze without SQL or output several SQLs.\n"

        try:
            response = self.llm.complete(prompt).text
            corrected = self.parse_sql_from_response(response)
            if corrected:
                return corrected[0]  # Take the first corrected SQL
        except Exception as e:
            logger.warning(f"Failed to correct SQL: {e}")

        return None

    def exploration(self, question: str, schema_str: str, db_type: str, db_path: str, credential: str, logger) -> Tuple[
        str, str]:
        """Column exploration phase - generate and execute exploration queries"""
        max_try = self.max_try
        pre_info = ""
        response_pre_txt = ""

        while max_try > 0:
            exploration_prompt = f"{schema_str}\nTask: {question}\n"
            exploration_prompt += self.get_exploration_prompt(db_type, schema_str)

            try:
                response = self.llm.complete(exploration_prompt).text
                response_pre_txt = response
                logger.info(f"[Exploration]\n{response[:500]}...\n[Exploration]")

                sqls = self.parse_sql_from_response(response)
                if len(sqls) < 3:
                    logger.warning(f"Too few SQLs generated: {len(sqls)}, retrying...")
                    max_try -= 1
                    continue

                results = self.execute_sqls(sqls, db_type, db_path, credential, logger)

                sql_count = 0
                for dic in results:
                    pre_info += f"Query:\n{dic['sql']}\nAnswer:\n{dic['res']}\n"
                    if isinstance(dic['res'], str):
                        sql_count += 1

                if sql_count == 0:
                    logger.warning("No successful SQL executions, breaking")
                    break

                if len(pre_info) < 100000:  # Limit context length
                    break

                logger.warning("Context too long, retrying with shorter queries")
                pre_info = ""
                max_try -= 1

            except Exception as e:
                logger.error(f"Exploration failed: {e}")
                max_try -= 1

        return pre_info, response_pre_txt

    def self_refine(self, question: str, schema_str: str, pre_info: str, db_type: str, db_path: str, credential: str,
                    logger) -> Optional[str]:
        """Self-refinement phase with iterative SQL generation and consistency checking"""
        iter_count = 0
        results_values = []
        results_tables = []
        error_rec = []

        while iter_count < self.max_iter:
            logger.info(f"Self-refine iteration: {iter_count}")

            # Generate SQL
            self_refine_prompt = self.get_self_refine_prompt(question, schema_str, pre_info, db_type)
            logger.info(f"[Self-refine]\n{self_refine_prompt[:500]}...\n[Self-refine]")

            max_try = self.max_try
            response = None

            while max_try > 0:
                try:
                    response_text = self.llm.complete(self_refine_prompt).text
                    sqls = self.parse_sql_from_response(response_text)
                    if len(sqls) == 1:
                        response = sqls[0]
                        break
                    else:
                        self_refine_prompt = "Please output one SQL only."
                except Exception as e:
                    logger.warning(f"LLM completion failed: {e}")
                max_try -= 1

            if not response:
                logger.error("Failed to generate SQL after retries")
                break

            logger.info(f"[Try to run SQL in self-refine]\n{response}\n[Try to run SQL in self-refine]")

            # Execute the generated SQL
            success, executed_result = self.execute_sql_safe(response, db_type, db_path, credential)
            error_rec.append(str(executed_result))

            # Early stop for repetitive empty results
            if self.early_stop and len(error_rec) > 3:
                if len(set(error_rec[-4:])) == 1 and error_rec[-1] == self.empty_result:
                    logger.info("Early stop: repetitive empty results")
                    break

            if success and executed_result != self.empty_result:
                # SQL executed successfully
                if not self.do_self_consistency:
                    return response

                # Self-consistency check
                self_consistency_prompt = self.get_self_consistency_prompt(question)
                self_consistency_prompt += f"Current answer: \n{self.hard_cut(executed_result, self.csv_max_len)}\n"
                self_consistency_prompt += f"Current sql:\n{response}\n"

                # Check for consistency with previous results
                try:
                    df_result = pd.read_csv(pd.io.common.StringIO(executed_result))

                    # Process results for comparison
                    df_result_copy = df_result.copy()
                    for col in df_result.select_dtypes(include=['float']):
                        df_result_copy[col] = df_result[col].round(2)

                    result_values_str = df_result_copy.to_string()

                    if result_values_str not in results_values:
                        # Check for empty or null columns
                        df_str = df_result.astype(str)
                        empty_columns = df_str.columns[((df_str == "0") | (df_str == "")).all()].tolist()

                        if empty_columns:
                            self_consistency_prompt += f"Empty results in Column {empty_columns}. Please correct them.\n"
                        else:
                            results_values.append(result_values_str)
                            results_tables.append(executed_result)
                    else:
                        # Consistent result found
                        logger.info(
                            f"[Consistent results]\n{self.hard_cut(executed_result, 500)}\n[Consistent results]")
                        return response

                except Exception as e:
                    logger.warning(f"Failed to parse CSV result: {e}")
                    return response

                self_refine_prompt = self_consistency_prompt
            else:
                # SQL failed, provide error feedback
                self_refine_prompt = f"Input sql:\n{response}\nThe error information is:\n{executed_result}\nPlease correct it and output only 1 complete SQL query."

            iter_count += 1

        logger.info(f"Total self-refine iterations: {iter_count}")

        # Return the best result if available
        if results_tables:
            return results_tables[0]  # Return first successful result

        return None

    def generate_multiple_candidates(self, question: str, schema_str: str, pre_info: str, db_type: str, db_path: str,
                                     credential: str, logger, num_candidates: int = 3) -> List[Tuple[str, str]]:
        """Generate multiple SQL candidates for voting"""
        candidates = []

        for i in range(num_candidates):
            logger.info(f"Generating candidate {i + 1}/{num_candidates}")

            if self.do_self_refinement:
                sql = self.self_refine(question, schema_str, pre_info, db_type, db_path, credential, logger)
            else:
                # Simple generation
                prompt = self.get_self_refine_prompt(question, schema_str, pre_info, db_type)
                try:
                    response = self.llm.complete(prompt).text
                    sqls = self.parse_sql_from_response(response)
                    sql = sqls[0] if sqls else None
                except Exception as e:
                    logger.warning(f"Candidate generation failed: {e}")
                    sql = None

            if sql:
                success, result = self.execute_sql_safe(sql, db_type, db_path, credential)
                if success and result != self.empty_result:
                    candidates.append((sql, result))
                    logger.info(f"Candidate {i + 1} generated successfully")
                else:
                    logger.warning(f"Candidate {i + 1} failed to execute: {result}")
            else:
                logger.warning(f"Failed to generate candidate {i + 1}")

        return candidates

    def vote_results(self, candidates: List[Tuple[str, str]], question: str, schema_str: str, logger) -> Optional[str]:
        """Implement majority voting among candidates"""
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0][0]

        # Group candidates by result similarity
        result_groups = defaultdict(list)

        for sql, result in candidates:
            # Use result as key for grouping
            result_groups[result].append(sql)

        # Count votes for each result
        vote_counts = {result: len(sqls) for result, sqls in result_groups.items()}
        sorted_results = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"Vote counts: {vote_counts}")

        # Check for ties
        max_votes = sorted_results[0][1]
        tied_results = [result for result, count in sorted_results if count == max_votes]

        if len(tied_results) > 1:
            if self.random_vote_for_tie:
                import random
                selected_result = random.choice(tied_results)
                logger.info(f"Tie broken randomly, selected: {selected_result[:100]}...")
            elif self.model_vote:
                # Use LLM to break ties
                return self.model_vote_tie_breaker(tied_results, result_groups, question, schema_str, logger)
            else:
                logger.warning("Tie detected but no tie-breaking method enabled")
                selected_result = tied_results[0]
        else:
            selected_result = sorted_results[0][0]

        # Return the first SQL from the winning group
        return result_groups[selected_result][0]

    def model_vote_tie_breaker(self, tied_results: List[str], result_groups: Dict, question: str, schema_str: str,
                               logger) -> Optional[str]:
        """Use LLM to break ties between equally voted results"""
        prompt = f"You are given DB info, task and candidate SQLs with their results. Choose the most correct one.\n"
        prompt += f"Database schema:\n{schema_str}\n"
        prompt += f"Task: {question}\n"
        prompt += f"Here are the candidate SQLs and answers:\n"

        for i, result in enumerate(tied_results):
            sqls = result_groups[result]
            prompt += f"\nCandidate {i + 1}:\n"
            prompt += f"SQL: {sqls[0]}\n"
            prompt += f"Result: {self.hard_cut(result, 1000)}\n"

        prompt += "\nCompare the SQL and results, think step by step and choose the best candidate number (1, 2, etc.).\n"
        prompt += "For results with null or zero values, they tend to be wrong.\n"
        prompt += "Output only the candidate number.\n"

        try:
            response = self.llm.complete(prompt).text
            # Extract candidate number
            import re
            match = re.search(r'\b(\d+)\b', response)
            if match:
                candidate_num = int(match.group(1)) - 1
                if 0 <= candidate_num < len(tied_results):
                    selected_result = tied_results[candidate_num]
                    logger.info(f"Model vote selected candidate {candidate_num + 1}")
                    return result_groups[selected_result][0]
        except Exception as e:
            logger.warning(f"Model voting failed: {e}")

        # Fallback to first candidate
        return result_groups[tied_results[0]][0]

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ):
        """Main execution method following ReFoRCE algorithm"""
        if self.dataset is None or self.llm is None:
            raise ValueError("Dataset and LLM must be provided for ReFoRCEGenerator.")

        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        logger.info(f"ReFoRCEGenerator processing sample {item}")
        row = self.dataset[item]
        question = row['question']
        db_type = row.get('db_type', 'sqlite')
        db_id = row.get("db_id")

        # Handle different database path configurations
        if db_type == 'sqlite':
            db_path = Path(self.db_path) / (db_id + ".sqlite") if self.db_path else row.get('db_path')
        else:
            db_path = db_id or row.get('db_path')

        credential = self.credential or row.get('credential')

        logger.debug(f"Processing question: {question[:100]}... (DB: {db_id}, Type: {db_type})")

        # Load external knowledge if enabled
        if self.use_external:
            external = self.load_external_knowledge(row.get("external", None))
            if external:
                question += "\n" + external
                logger.debug("Loaded external knowledge")

        # Process database schema
        logger.debug("Processing database schema...")
        if isinstance(schema, (str, PathLike)) and Path(schema).exists():
            schema = load_dataset(schema)

        if schema is None:
            instance_schema_path = row.get("instance_schemas")
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
                logger.debug(f"Loaded schema from: {instance_schema_path}")
            else:
                logger.debug("Fetching schema from dataset")
                schema = self.dataset.get_db_schema(item)

            if schema is None:
                raise ValueError("Failed to load a valid database schema for the sample!")

        # Normalize schema format
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            schema_str = parse_schema_from_df(schema)
        elif isinstance(schema, str):
            schema_str = schema
        else:
            schema_str = str(schema)

        if schema_links:
            schema_str += f"\nSchema Links: {schema_links}"

        logger.debug("Database schema processed")

        # Phase 1: Column Exploration
        pre_info = ""
        if self.do_column_exploration:
            logger.info("Starting column exploration phase...")
            pre_info, _ = self.exploration(question, schema_str, db_type, db_path, credential, logger)
            logger.info("Column exploration completed")

        # Phase 2: SQL Generation with Voting
        pred_sql = None

        if self.do_vote and self.num_votes > 1:
            logger.info(f"Starting voting phase with {self.num_votes} candidates...")
            candidates = self.generate_multiple_candidates(
                question, schema_str, pre_info, db_type, db_path, credential, logger, self.num_votes
            )

            if candidates:
                pred_sql = self.vote_results(candidates, question, schema_str, logger)
                logger.info("Voting phase completed")
            else:
                logger.warning("No valid candidates generated, falling back to single generation")

        # Fallback: Single SQL generation
        if pred_sql is None:
            logger.info("Starting single SQL generation...")
            if self.do_self_refinement:
                pred_sql = self.self_refine(question, schema_str, pre_info, db_type, db_path, credential, logger)
            else:
                # Simple generation
                prompt = self.get_self_refine_prompt(question, schema_str, pre_info, db_type)
                try:
                    response = self.llm.complete(prompt).text
                    sqls = self.parse_sql_from_response(response)
                    pred_sql = sqls[0] if sqls else None
                except Exception as e:
                    logger.error(f"Simple generation failed: {e}")

        # Final fallback
        if pred_sql is None:
            logger.warning("All generation methods failed, creating fallback SQL")
            pred_sql = "/* Failed to generate SQL */"

        logger.debug(f"Final SQL: {pred_sql[:100]}...")

        pred_sql = self.save_output(pred_sql, item, row.get("instance_id", str(item)))

        logger.info(f"ReFoRCEGenerator sample {item} processed successfully")
        if data_logger:
            data_logger.info(f"{self.NAME}.final_sql | sql={pred_sql[:200]}")
            data_logger.info(f"{self.NAME}.act end | item={item}")
        return pred_sql
