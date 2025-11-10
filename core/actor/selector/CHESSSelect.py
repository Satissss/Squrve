from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Union, List, Dict, Any
from pathlib import Path
import re
from loguru import logger
import pandas as pd
import json
import ast

from core.actor.selector.BaseSelector import BaseSelector
from core.data_manage import Dataset, single_central_process
from core.utils import parse_schema_from_df, load_dataset, save_dataset
from core.db_connect import execute_sql


@dataclass
class CHESSConfig:
    """Configuration for CHESS selector components"""
    # Database execution settings
    max_execution_time: int = 30  # seconds
    enable_execution_voting: bool = True
    enable_llm_evaluation: bool = True
    # Unit test settings
    ut_unit_test_count: int = 20


class CHESSSelector(BaseSelector):
    """Selector component from CHESS-SQL for choosing the best SQL candidate using execution results and unit tests."""

    NAME = "CHESSSelector"

    REVISE_TEMPLATE = '''You are an expert SQL developer. Revise the following SQL query based on the feedback provided.

Database Schema:
{DATABASE_SCHEMA}

Question: {QUESTION}

Original SQL: {SQL}

Feedback: {FEEDBACK}

Please provide the corrected SQL query. Output only the SQL query without any explanations.'''

    UNIT_TEST_TEMPLATE = '''Generate natural language unit tests for the following SQL query.

Question: {QUESTION}
SQL: {SQL}

Generate {UNIT_TEST_COUNT} unit tests that can be used to validate the SQL query.
Each test should be a natural language question that the SQL should answer correctly.

Return the tests as a Python list of strings.'''

    EVALUATE_TEMPLATE = '''Evaluate the following SQL query based on the unit tests.

Question: {QUESTION}
SQL: {SQL}

Unit Tests:
{UNIT_TESTS}

Evaluate if the SQL query correctly answers the unit tests.
Return a JSON object with 'score' (0-1) and 'feedback' (string).'''

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Any = None,
            config: CHESSConfig = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, **kwargs)
        self.config = config or CHESSConfig()

    def _compare_execution_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare execution results and return voting information"""
        if not results:
            return {"votes": {}, "best_sql": None, "total_votes": 0}
        
        # Count successful executions
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            return {"votes": {}, "best_sql": None, "total_votes": 0}
        
        # Group by result content
        result_groups = {}
        for result in successful_results:
            result_str = str(result["result"])
            if result_str not in result_groups:
                result_groups[result_str] = []
            result_groups[result_str].append(result["sql"])
        
        # Find the most common result
        best_result = max(result_groups.items(), key=lambda x: len(x[1]))
        
        return {
            "votes": result_groups,
            "best_sql": best_result[1][0],  # Return the first SQL that produced this result
            "total_votes": len(successful_results),
            "winning_result": best_result[0],
            "winning_count": len(best_result[1])
        }

    def _generate_unit_tests(self, question: str, sql: str) -> List[str]:
        """Generate unit tests for the SQL query"""
        if not self.config.enable_llm_evaluation or not self.llm:
            return []
            
        template = self.UNIT_TEST_TEMPLATE

        try:
            prompt = template.format(
                QUESTION=question,
                SQL=sql,
                UNIT_TEST_COUNT=self.config.ut_unit_test_count
            )
            
            # Use LLM's own configuration instead of hardcoded temperature
            response = self.llm.complete(prompt).text
            
            # Extract list from response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                tests_str = match.group(0)
                try:
                    tests = ast.literal_eval(tests_str)
                    return tests if isinstance(tests, list) else []
                except:
                    # Fallback to eval if ast.literal_eval fails
                    tests = eval(tests_str)
                    return tests if isinstance(tests, list) else []
            return []
            
        except Exception as e:
            logger.warning(f"Failed to generate unit tests: {e}")
            return []

    def _evaluate_sql(self, question: str, sql: str, unit_tests: List[str]) -> Dict[str, Any]:
        """Evaluate SQL query using unit tests"""
        
        if not unit_tests or not self.config.enable_llm_evaluation or not self.llm:
            return {"score": 0.5, "feedback": "No unit tests available"}
            
        template = self.EVALUATE_TEMPLATE

        try:
            unit_tests_text = "\n".join([f"{i+1}. {test}" for i, test in enumerate(unit_tests)])
            
            prompt = template.format(
                QUESTION=question,
                SQL=sql,
                UNIT_TESTS=unit_tests_text
            )
            
            # Use LLM's own configuration instead of hardcoded temperature
            response = self.llm.complete(prompt).text
            
            # Extract JSON from response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                    return result if isinstance(result, dict) else {"score": 0.5, "feedback": "Invalid evaluation result"}
                except:
                    # Fallback to eval if json.loads fails
                    result = eval(match.group(0))
                    return result if isinstance(result, dict) else {"score": 0.5, "feedback": "Invalid evaluation result"}
            return {"score": 0.5, "feedback": "Could not parse evaluation result"}
            
        except Exception as e:
            logger.warning(f"Failed to evaluate SQL: {e}")
            return {"score": 0.5, "feedback": f"Evaluation failed: {e}"}

    def _select_best_sql(self, candidates: List[Dict[str, Any]], execution_voting: Dict[str, Any], evaluations: List[Dict[str, Any]] = None) -> str:
        """Select the best SQL query based on execution voting and optional LLM evaluations"""
        if not candidates:
            return ""
        
        # If execution voting found a winner, use it
        if execution_voting.get("best_sql"):
            return execution_voting["best_sql"]
        
        # If no execution voting or no clear winner, use LLM evaluations
        if evaluations and self.config.enable_llm_evaluation:
            best_score = -1
            best_sql = candidates[0]["SQL"]
            
            for i, evaluation in enumerate(evaluations):
                if evaluation:
                    score = evaluation.get("score", 0)
                    if score > best_score:
                        best_score = score
                        best_sql = candidates[i]["SQL"]
            
            return best_sql
        
        # Fallback to first candidate
        return candidates[0]["SQL"]

    def _revise_sql(self, question: str, schema: str, sql: str, feedback: str = "") -> str:
        """Revise SQL query based on feedback"""
        
        if not self.llm:
            return sql
            
        template = self.REVISE_TEMPLATE

        try:
            prompt = template.format(
                DATABASE_SCHEMA=schema,
                QUESTION=question,
                SQL=sql,
                FEEDBACK=feedback
            )
            
            # Use LLM's own configuration instead of hardcoded temperature
            response = self.llm.complete(prompt).text
            
            # Extract SQL from response (since output only SQL)
            return response.strip()
                
        except Exception as e:
            logger.warning(f"Failed to revise SQL: {e}")
            return sql

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, Path, List[str], List[Path]] = None,
            data_logger=None,
            **kwargs
    ):
        """Select the best SQL from candidates using execution voting and concurrent evaluation."""
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        row = self.dataset[item]
        question = row['question']
        db_type = row.get('db_type', 'sqlite')
        db_id = row.get('db_id', '')
        credential = self.dataset.credential if hasattr(self.dataset, 'credential') else None

        # Load pred_sql using base class method
        pred_sql = self.load_pred_sql(pred_sql, item)
        if not pred_sql:
            return ""
        if data_logger:
            data_logger.info(f"{self.NAME}.candidates | count={len(pred_sql)}")
            

        candidates = [{"SQL": sql} for sql in pred_sql]

        # Step 1: Execute all SQL candidates concurrently
        execution_results = []
        if self.config.enable_execution_voting:
            logger.debug(f"Executing {len(candidates)} SQL candidates concurrently...")
            with ThreadPoolExecutor() as executor:
                future_to_idx = {
                    executor.submit(self.execute_sql_safe, cand["SQL"], db_type, db_id, credential): i
                    for i, cand in enumerate(candidates)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        result["sql"] = candidates[idx]["SQL"]  # Add SQL to result for reference
                        execution_results.append(result)
                    except Exception as e:
                        execution_results.append({
                            "success": False,
                            "result": None,
                            "error": str(e),
                            "sql": candidates[idx]["SQL"]
                        })

            # Compare execution results
            execution_voting = self._compare_execution_results(execution_results)
            logger.debug(f"Execution voting results: {execution_voting}")
        else:
            execution_voting = {"votes": {}, "best_sql": None, "total_votes": 0}

        # Step 2: Generate unit tests and evaluate with LLM (optional)
        evaluations = []
        if self.config.enable_llm_evaluation and self.llm:
            logger.debug("Generating unit tests and evaluating with LLM...")
            unit_tests = self._generate_unit_tests(question, candidates[0]["SQL"])
            
            # Concurrently evaluate each candidate with LLM
            evaluations = [None] * len(candidates)
            with ThreadPoolExecutor() as executor:
                future_to_idx = {
                    executor.submit(self._evaluate_sql, question, cand["SQL"], unit_tests): i
                    for i, cand in enumerate(candidates)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        evaluations[idx] = future.result()
                    except Exception as e:
                        evaluations[idx] = {"score": 0, "feedback": str(e)}

        # Step 3: Select the best SQL
        best_sql = self._select_best_sql(candidates, execution_voting, evaluations)
        if data_logger:
            data_logger.info(f"{self.NAME}.selected_sql | sql={best_sql}")
            

        # Step 4: Optional SQL revision based on feedback
        if self.config.enable_llm_evaluation and self.llm:
            # Find the evaluation for the best SQL
            best_idx = next((i for i, cand in enumerate(candidates) if cand["SQL"] == best_sql), 0)
            best_eval = evaluations[best_idx] if best_idx < len(evaluations) else None
            
            if best_eval and best_eval.get("feedback"):
                logger.debug("Revising SQL based on feedback...")
                # Load and parse schema for revision
                if schema is None:
                    instance_schema_path = row.get("instance_schemas")
                    if instance_schema_path:
                        schema = load_dataset(instance_schema_path)
                    if schema is None:
                        schema = self.dataset.get_db_schema(item)
                    if schema is None:
                        raise Exception("Failed to load a valid database schema for the sample!")

                if isinstance(schema, dict):
                    schema = single_central_process(schema)
                elif isinstance(schema, list):
                    schema = pd.DataFrame(schema)

                schema_str = parse_schema_from_df(schema)

                revised_sql = self._revise_sql(question, schema_str, best_sql, best_eval["feedback"])
                if revised_sql and revised_sql != best_sql:
                    best_sql = revised_sql
                    logger.debug("SQL revised successfully")

        # Save the result using base class method
        best_sql = self.save_result(best_sql, item, row.get("instance_id"))

        if data_logger:
            data_logger.info(f"{self.NAME}.selected_sql | sql={best_sql}")
            data_logger.info(f"{self.NAME}.act end | item={item}")
            
        return best_sql 