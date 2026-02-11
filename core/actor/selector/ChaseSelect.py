from typing import Union, List, Dict, Any, Tuple
from os import PathLike
from pathlib import Path
from loguru import logger
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.actor.selector.BaseSelect import BaseSelector
from core.data_manage import Dataset
from core.db_connect import get_sql_exec_result_with_time
from core.utils import compare_pandas_table, load_dataset, parse_json_from_str

@BaseSelector.register_actor
class ChaseSelector(BaseSelector):
    """
    Selector that keeps only successfully executed SQL candidates and picks the
    one with the shortest execution time.
    """

    NAME = "ChaseSelector"

    SKILL = """# ChaseSelector

Chase-SQL style selection: execute all candidates, keep only successful runs, group by identical execution results. Per group, take fastest SQL; pairwise LLM comparison (schema + question + SQL + execution result) assigns scores—winner gains loser's group count. Optional majority voting when ambiguous. Advantage: execution result as ground truth, reduces hallucination; drawback: requires DB connectivity, many LLM calls for pairwise comparison.

## Inputs
- `pred_sql`: List of SQL candidates to select from.
- `schema`: DB schema for LLM comparison. If absent, loaded from dataset.

## Output
`pred_sql` (single selected SQL)

## Steps
1. Execute all candidates; keep only successful runs.
2. Group by identical execution result.
3. Per group: pick fastest SQL, use group size as initial score.
4. Pairwise LLM comparison; update scores; optionally use majority voting.
5. Return SQL with highest score.
"""

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Any = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            retry_num: int = 3,
            force_voting: bool = True,
            use_external: bool = True,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, **kwargs)
        self.retry_num = retry_num
        self.force_voting = force_voting
        self.use_external: bool = use_external

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

    def _compare_sql_pair(self, question, schema, sql1, sql2) -> bool:
        """
        Compare two SQL candidates using LLM to determine which is correct.
        
        Args:
            question: The question being asked
            schema: The database schema
            sql1: First SQL candidate dict with 'sql' and 'result' keys
            sql2: Second SQL candidate dict with 'sql' and 'result' keys
        
        Returns:
            True if sql1 is correct, False if sql2 is correct, None if uncertain
        """
        prompt_template = """### Role
You are an expert Database Administrator and SQL Auditor. Your task is to evaluate two candidate SQL queries and determine which one accurately answers the user's question based on the provided Database Schema.

### Database Schema
{DATABASE_SCHEMA}

### Question
{QUESTION}

### Candidate A
**Query:** {CANDIDATE_A_QUERY}
**Execution Result:** {CANDIDATE_A_RESULT}

---

### Candidate B
**Query:** {CANDIDATE_B_QUERY}
**Execution Result:** {CANDIDATE_B_RESULT}

---
### Instructions for Analysis
1. **Semantic Alignment:** Compare the natural language intent of the question with the relational logic of the SQL. Does the query capture the core entity being asked for, including all implicit conditions?
2. **Relational Path Integrity:** Analyze the navigation between tables. Does the query utilize the correct paths (joins/subqueries) to link the data, and does it respect the hierarchical or associative relationships defined in the schema?
3. **Result Consistency:** Evaluate the "Execution Result" against the expected output format. Does the resulting data structure (columns, types, and values) provide a direct and complete answer to the user's request?

### Output Requirement
Return the result strictly in JSON format:
{
  "result": "A or B",
  "reason": "A concise explanation identifying the specific logical flaw in the incorrect query and why the chosen one is superior."
}
        """
        try:
            # Prepare prompt with question and schema
            prompt = prompt_template.replace("{DATABASE_SCHEMA}", str(schema)).replace("{QUESTION}", question)
            prompt = prompt.replace("{CANDIDATE_A_QUERY}", sql1['sql']).replace("{CANDIDATE_B_QUERY}", sql2['sql'])

            # Convert results to string representation
            if isinstance(sql1['result'], pd.DataFrame):
                if not sql1['result'].empty:
                    res1 = str(sql1['result'].head(10).to_dict(orient="records"))
                else:
                    res1 = "Empty result (no rows returned)"
            else:
                res1 = str(sql1['result']) if sql1['result'] is not None else "None"

            if isinstance(sql2['result'], pd.DataFrame):
                if not sql2['result'].empty:
                    res2 = str(sql2['result'].head(10).to_dict(orient="records"))
                else:
                    res2 = "Empty result (no rows returned)"
            else:
                res2 = str(sql2['result']) if sql2['result'] is not None else "None"

            prompt = prompt.replace("{CANDIDATE_A_RESULT}", res1).replace("{CANDIDATE_B_RESULT}", res2)

            # Get LLM response
            res = self.llm.complete(prompt).text
            res = parse_json_from_str(res)
            res = res.get("result", "")
            if "A" in res and "B" not in res:
                return True
            elif "B" in res and "A" not in res:
                return False
            else:
                logger.debug(f"Ambiguous LLM response: {res}")
                return None
        except Exception as e:
            logger.info(f"Error in SQL comparison: {e}")

        return None

    def _select_best_sql(self, question, schema, sqls: List[Dict], max_workers: int = 2):
        """
        Select the best SQL from a list of candidates using pairwise LLM comparison.
        
        Args:
            question: The question being asked
            schema: The database schema
            sqls: List of SQL candidate dictionaries with 'index', 'count', 'score' fields
            max_workers: Maximum number of threads for parallel comparison (default: 2)
        
        Returns:
            Tuple of (best_sql_string, score)
        """
        # Handle single candidate case
        if len(sqls) == 1:
            return sqls[0]['sql'], sqls[0]['score']

        # Build all pairs for comparison
        pairs = []
        for ind, sql1 in enumerate(sqls):
            for sql2 in sqls[ind + 1:]:
                pairs.append((sql1, sql2))

        # Use ThreadPoolExecutor for parallel comparison
        def compare_pair(pair):
            sql1, sql2 = pair
            if self.force_voting or len(sqls) > 3 and sql1['count'] > 1 and sql2['count'] > 1:
                # This supplements the original select method in Chase-SQL.
                # We assume that when a question is highly challenging
                # and the candidate answers are difficult to distinguish,
                # a majority voting strategy is introduced.
                res_dict = {"true": 0, "false": 0}
                for _ in range(max_workers):
                    temp_res = self._compare_sql_pair(question, schema, sql1, sql2)
                    if temp_res is True:
                        res_dict["true"] += 1
                    elif temp_res is False:
                        res_dict["false"] += 1
                res = res_dict['true'] > res_dict['false']
            else:
                res = self._compare_sql_pair(question, schema, sql1, sql2)
            return (sql1['index'], sql2['index'], res, sql1['count'], sql2['count'])

        # Execute comparisons in parallel with max_workers threads
        comparison_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {executor.submit(compare_pair, pair): pair for pair in pairs}
            for future in as_completed(future_to_pair):
                try:
                    result = future.result()
                    comparison_results.append(result)
                except Exception as exc:
                    logger.warning(f"Comparison failed with exception: {exc}")

        # Update scores based on comparison results
        for idx1, idx2, res, count1, count2 in comparison_results:
            if res is None:
                continue
            if res:  # sql1 is correct
                sqls[idx1]['score'] += count2
            else:  # sql2 is correct
                sqls[idx2]['score'] += count1

        # Select the SQL with highest score
        sqls.sort(key=lambda x: x['score'], reverse=True)
        best_sql = sqls[0]

        return best_sql['sql'], best_sql['score']

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, PathLike, List[str], List[PathLike]] = None,
            data_logger=None,
            **kwargs
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        row = self.dataset[item] if self.dataset else {}
        question = row.get('question', '')
        if not question:
            logger.warning(f"{self.NAME} | No question found for item {item}")
            return ""

        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external", None))
            if external_knowledge:
                question += "\n" + external_knowledge
                logger.debug("已加载外部知识")

        db_type = row.get("db_type", "sqlite")
        db_id = row.get("db_id", "")
        credential = getattr(self.dataset, "credential", None)
        row_db_path = row.get("db_path")
        dataset_db_root = getattr(self.dataset, "db_path", None)
        db_path = None
        if row_db_path:
            db_path = Path(row_db_path)
        elif dataset_db_root and db_type == "sqlite" and db_id:
            db_path = Path(dataset_db_root) / f"{db_id}.sqlite"

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

        pred_sql_list = self.load_pred_sql(pred_sql, item)
        if not pred_sql_list:
            return ""

        successful_runs = []
        for sql in pred_sql_list:
            exec_args = self._build_exec_args(db_type, sql, db_id=db_id, db_path=db_path, credential=credential)
            if data_logger:
                data_logger.info(f"{self.NAME}.exec_params | sql={sql} | params={exec_args}")
            try:
                elapsed, exec_result = get_sql_exec_result_with_time(db_type, **exec_args)
            except Exception as exc:
                if data_logger:
                    data_logger.info(f"{self.NAME}.exec_failed | sql={sql} | error={exc}")
                continue

            res, err = self._normalize_exec_result(exec_result)
            if err:
                if data_logger:
                    data_logger.info(f"{self.NAME}.exec_error | sql={sql} | error={err}")
                continue

            successful_runs.append({
                "sql": sql,
                "time_cost": elapsed,
                "result": res
            })

        if not successful_runs:
            logger.warning(f"{self.NAME} | no successful executions for item {item}")
            return ""

        # Group by execution result, select the fastest SQL from the most frequent result group
        result_groups = self._group_by_result(successful_runs)

        if not result_groups:
            logger.warning(f"{self.NAME} | no result groups for item {item}")
            return ""

        # Replace each group with the fastest SQL (since all results in group are identical)
        for ind, group in enumerate(result_groups):
            count = len(group)
            # Find the SQL with minimum execution time in this group
            min_time_res = min(group, key=lambda x: x['time_cost'])
            # Add metadata for scoring
            min_time_res['index'] = ind
            min_time_res['count'] = count
            min_time_res['score'] = count  # Initialize score with frequency count
            result_groups[ind] = min_time_res

        best_sql, score = self._select_best_sql(question, schema, result_groups)

        if data_logger:
            data_logger.info(f"{self.NAME}.best_candidate | details={best_sql} | score={score}")

        best_sql = self.save_result(best_sql, item, row.get("instance_id"))

        if data_logger:
            data_logger.info(f"{self.NAME}.selected_sql | sql={best_sql}")
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return best_sql

    @staticmethod
    def _build_exec_args(
            db_type: str,
            sql: str,
            db_id: str = "",
            db_path: Union[str, Path, None] = None,
            credential: Any = None
    ) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "sql_query": sql,
            "db_path": db_path,
            "db_id": db_id
        }
        credential_path = None
        if isinstance(credential, dict):
            credential_path = credential.get(db_type)
        elif credential:
            credential_path = credential

        if credential_path:
            args["credential_path"] = credential_path
        return args

    @staticmethod
    def _normalize_exec_result(exec_result: Any) -> Tuple[Any, Any]:
        if isinstance(exec_result, tuple):
            if len(exec_result) == 3:
                res, err, _ = exec_result
                return res, err
            if len(exec_result) == 2:
                res, err = exec_result
                return res, err
            if len(exec_result) >= 1:
                return exec_result[0], None
            return None, "Empty result tuple"

        return exec_result, None

    @staticmethod
    def _group_by_result(successful_runs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group SQL runs by their execution results using DataFrame comparison."""
        groups = []
        for run in successful_runs:
            result = run["result"]
            placed = False

            for group in groups:
                group_result = group[0]["result"]
                # Compare DataFrames using compare_pandas_table utility
                if isinstance(result, pd.DataFrame) and isinstance(group_result, pd.DataFrame):
                    if compare_pandas_table(result, group_result, ignore_order=True) == 1:
                        group.append(run)
                        placed = True
                        break
                # Fallback to direct comparison for non-DataFrame results
                elif result == group_result or (pd.isna(result) and pd.isna(group_result)):
                    group.append(run)
                    placed = True
                    break

            if not placed:
                groups.append([run])

        return groups
