from typing import Union, List, Optional, Dict, Any, Literal
from pathlib import Path
from loguru import logger
from llama_index.core.llms.llm import LLM
import re
import concurrent.futures
import pandas as pd

from core.actor.optimizer.BaseOptimize import BaseOptimizer
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from core.db_connect import get_sql_exec_result
from core.utils import sql_clean, parse_schema_from_df, parse_list_from_str, parse_json_from_str


class AdaptiveOptimizer(BaseOptimizer):
    NAME = "AdaptiveOptimizer"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/optimized_sql",
            debug_turn_n: int = 3,
            open_parallel: bool = True,
            max_workers: Optional[int] = None,
            quit_flag: str = "QUIT",
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.debug_turn_n = debug_turn_n
        self.quit_flag = quit_flag

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

    def _decompose_sqls(
            self,
            sqls: str | List[str],
            data_logger=None,
    ):
        # Decompose the SQLs input into small meta SQL list.
        if not sqls:
            if data_logger:
                data_logger.info("No Valid SQLs! Returning empty List.")
            return None

        sqls = [sqls] if isinstance(sqls, str) else sqls

        prompt_template = """# Role
你是一个专业的数据库查询优化器和SQL解析引擎。你的任务是将输入的“完整原始SQL语句”（可能包含一个或多个候选SQL）分解为一组“原子Meta SQL语句”。

# Core Definitions
1. 原子Meta SQL (Atomic Meta SQL)：
    -   一个原子 SQL 应当包含尽可能少的 `WHERE` 条件、`GROUP BY` 维度或 `ORDER BY` 字段。
    -   独立性：每个Meta SQL必须能够单独提交给数据库执行，不依赖其他Meta SQL的执行结果（即无运行时依赖），因此它们可以并行执行。
    -   完备性：所有Meta SQL的结果集组合在一起，必须包含重构原始SQL结果所需的全部数据。
    -   Schema限制：不得引入原始SQL中不存在的表或列。

# Decomposition Rules
请遵循以下逻辑对输入的SQL进行分解：

## 1. 单表拆解 (Single Table Access)：
    如果一个单表查询中包含多个逻辑部分，必须将其拆解为多个独立的 SQL：
* **WHERE 拆解**：
    * 将 `WHERE condition1 AND condition2` 拆解为两个独立的 SQL：`SELECT ... WHERE condition1` 和 `SELECT ... WHERE condition2`。
    * *目的*：确保每个原子 SQL 仅聚焦于一个过滤维度。
* **GROUP BY 拆解**：
    * 将 `GROUP BY col1, col2` 拆解为针对不同维度的聚合查询（如果语义允许）。例如 `SELECT count(*) ... GROUP BY A, B` 应尝试拆解为关注 A 的聚合和关注 B 的聚合。
* **ORDER BY 拆解**：
    * 将 `ORDER BY col1, col2` 拆解为独立的排序查询，除非 col2 的排序强依赖于 col1 的分组上下文。

## 2. Join 拆解 (Join Separation)
* 对于 `TableA JOIN TableB ON ... WHERE A.col=1 AND B.col=2`：
    * 首先拆解为针对 TableA 的查询和针对 TableB 的查询。
    * 接着对拆分后的查询应用“从句裂变规则”（例如，如果 TableA 还有多个 WHERE 条件，继续拆分）。

## 3. 子查询处理 (Subquery & Correlation)
* **标准拆解**：将子查询提取为独立 SQL。
* **相关子查询 (IN/EXISTS)**：
    * 对于 `A WHERE id IN (SELECT id FROM B WHERE condition)`：
    * 生成 Meta SQL 1: `SELECT id FROM B WHERE condition`
    * 生成 Meta SQL 2: `WITH Filtered_B AS (SELECT id FROM B WHERE condition) SELECT * FROM A WHERE id IN (SELECT id FROM Filtered_B)`
    * *注意*：Meta SQL 2 必须包含完整的逻辑闭环，确保其独立可执行。
    
## 4. 多候选输入处理 (Multiple Candidates)：
* 输入可能包含多个语义相同但写法不同的候选SQL。
* 你需要处理列表中的每一个SQL，将它们全部分解。
* **去重合并**：将所有分解得到的 Meta SQL 放入同一个集合中，去除完全重复的字符串，最终返回一个去重后的列表。

# Output Format
* 仅返回一个可解析的 Python List [str]。
* 列表中的元素为字符串格式的 Meta SQL。
* 严禁输出任何Markdown格式（如 ```json ... ```）、解释性文字或代码块标记，仅输出纯文本列表。

# Few-Shot Examples

## Example 1 (Complex Filter Decomposition)
**Input:**
SELECT * FROM users WHERE age > 18 AND city = 'Beijing' AND status = 1

**Output:** [
"SELECT * FROM users WHERE age > 18", 
"SELECT * FROM users WHERE city = 'Beijing'", 
"SELECT * FROM users WHERE status = 1"
]

## Example 2 (Join & Clause Fission)
**Input:**
SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id WHERE u.age > 18 AND o.amount > 100 AND o.status = 'paid'

**Output:** [
"SELECT name FROM users WHERE age > 18", 
"SELECT amount FROM orders WHERE amount > 100", 
"SELECT amount FROM orders WHERE status = 'paid'"
]

## Example 3 (Aggregation Decomposition)
**Input:**
SELECT count(*) FROM logs WHERE date = '2023-01-01' GROUP BY level, server_id

**Output:** [
"SELECT count(*) FROM logs WHERE date = '2023-01-01' GROUP BY level", 
"SELECT count(*) FROM logs WHERE date = '2023-01-01' GROUP BY server_id"
]

## Example 4 (Subquery)
**Input:**
SELECT * FROM products WHERE id IN (SELECT product_id FROM sales WHERE year = 2023)

**Output:** [
"SELECT product_id FROM sales WHERE year = 2023", 
"WITH target_scope AS (SELECT product_id FROM sales WHERE year = 2023) SELECT * FROM products WHERE id IN (SELECT product_id FROM target_scope)"
]

# Task
请处理以下输入的原始SQL语句（列表）：
**Input:**
{sqls}

**Output:** 
"""
        prompt = prompt_template.format(sqls=sqls)

        try:
            decompose_sqls = self.llm.complete(prompt).text
            decompose_sqls = parse_json_from_str(decompose_sqls)
            decompose_sqls = [sql_clean(sql) for sql in decompose_sqls]

            meta_sqls = list(dict.fromkeys(decompose_sqls))
            if not meta_sqls:
                raise ValueError("Failed to parse any meta SQLs from LLM output.")
            sqls.extend(meta_sqls)
            if data_logger:
                data_logger.info(f"{self.NAME}.decompose_sqls | meta_sql_count={len(meta_sqls)}")

            return sqls
        except Exception as e:
            if data_logger:
                data_logger.info(f"Errors when decomposing sqls:{e}")
            return None

    def _get_meta_sql_feedback(
            self,
            sql: str,
            db_id: Optional[str] = None,
            db_path: Optional[Union[str, Path]] = None,
            db_type: str = "sqlite",
            credential: Optional[Dict] = None,
            data_logger=None,
    ):
        # Decompose the input sql and get the final feedback.
        if not sql:
            return None

        final_feedback = []
        meta_sqls = self._decompose_sqls(sql, data_logger)
        if meta_sqls is None:
            # Error when decomposing the sqls
            return None

        for meta_sql in meta_sqls:
            exec_args = self._build_exec_args(db_type, meta_sql, db_id=db_id, db_path=db_path, credential=credential)
            try:
                res, err = get_sql_exec_result(db_type, **exec_args)
            except Exception as exc:
                if data_logger:
                    data_logger.info(f"{self.NAME}.exec_failed | sql={meta_sql} | error={exc}")
                continue
            final_feedback.append({
                "sql": meta_sql,
                "res": res,
                "err": err,
                "status": err is None,
            })

        return final_feedback

    def _refine_syntax_schema_error(
            self,
            question: str,
            sql: str,
            schema: str = None,
            db_type: str = None,
            feedback: List[Dict] = None,
            data_logger=None
    ):
        if not sql or not feedback:
            return None
        if self.llm is None:
            if data_logger:
                data_logger.info("No LLM configured, cannot refine SQL.")
            return None

        def _summarize_feedback() -> str:
            segments = []
            for idx, record in enumerate(feedback):
                label = "ORIGINAL_SQL" if idx == 0 else f"ATOMIC_SQL_{idx}"
                snippet = [
                    f"[{label}]",
                    f"SQL: {sql_clean(record.get('sql', '') or '')}",
                ]
                err_msg = record.get("err")
                if err_msg:
                    snippet.append(f"Error: {err_msg}")
                segments.append("\n".join(snippet))
            return "\n\n".join(segments)

        feedback_txt = _summarize_feedback()
        cleaned_sql = sql_clean(sql)
        prompt = f"""# Instruction
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.

# Question
{question or 'N/A'}

# Database Schema: ({db_type or 'unknown db'})
{schema}

# Original SQL
{cleaned_sql}

# {db_type} errors:
{feedback_txt}

# Constraints
* In SELECT <column>, just select needed columns in the 【Question】 without any unnecessary column or value
* In FROM <table> or JOIN <table>, do not include unnecessary table
* If use max or min func, JOIN <table> FIRST, THEN use SELECT MAX(<column>) or SELECT MIN(<column>)
* If [Value examples] of <column> has 'None' or None, use JOIN <table> or WHERE <column> is NOT NULL is better
* If use ORDER BY <column> ASC|DESC, add GROUP BY <column> before to select distinct values
* Use explicit JOIN syntax instead of implicit comma-separated joins
* Always qualify column names with table aliases when joining multiple tables
* For {db_type}, use appropriate string literal quoting (single quotes for most databases)
* Ensure WHERE clause conditions use correct comparison operators and data types
* When using aggregate functions, include all non-aggregated columns in GROUP BY
* Use table aliases to improve readability and avoid column ambiguity
* Verify subqueries return single values when used in comparison operations
* For date/time comparisons, use proper date functions and formatting for {db_type}

# Output Format
Return ONLY the final executable SQL text. Do not wrap it in code fences or add any commentary.
"""
        try:
            best_sql = self.llm.complete(prompt).text.strip()
            best_sql = sql_clean(best_sql)
        except Exception as exc:
            if data_logger:
                data_logger.info(f"LLM refinement failed: {exc}")
            return None

        if not best_sql:
            return None

        if data_logger:
            data_logger.info(f"{self.NAME}.syntax_refined_sql | sql={best_sql}")
        return best_sql

    def _refine_logic_error(
            self,
            question: str,
            sql: str,
            schema: str = None,
            db_type: str = None,
            feedback: List[Dict] = None,
            data_logger=None
    ):
        if not sql or not feedback:
            return None
        if self.llm is None:
            if data_logger:
                data_logger.info("No LLM configured, cannot refine SQL.")
            return None

        def _summarize_feedback() -> str:
            segments = []
            for idx, record in enumerate(feedback):
                label = "ORIGINAL_SQL" if idx == 0 else f"ATOMIC_SQL_{idx}"
                snippet = [
                    f"[{label}]",
                    f"SQL: {sql_clean(record.get('sql', '') or '')}",
                ]
                res_msg = record.get("res")
                if isinstance(res_msg, pd.DataFrame):
                    res_msg = res_msg.head(5).to_dict(orient="records")
                    snippet.append(f"Query results: {str(res_msg)}")
                segments.append("\n".join(snippet))
            return "\n\n".join(segments)

        feedback_txt = _summarize_feedback()
        cleaned_sql = sql_clean(sql)
        prompt = f"""# Role
You are an elite SQL logic analyst. Your mission: compare the natural-language question with the executable SQL set, reason over their concrete execution results, and fix any logical mistakes without introducing syntax errors.

# Hint
Check whether the original SQL aligns with the query intent. If the logic is correct, exit immediately; if there is a logical error, use the execution results of the original SQL and the atomic SQL to identify the deviation and then generate the optimal SQL.

# Question: 
{question or 'N/A'}

# Database Schema: ({db_type or 'unknown db'})
{schema}

# Original SQL
{cleaned_sql}

# Execution Evidences 
{feedback_txt}

# Task
1. Diagnose whether the original SQL’s execution result actually answers the question; base the diagnosis on the provided evidence.
2. If logic is wrong, draft a corrected SQL that obeys the database type conventions and uses only necessary tables/columns.
3. Validate that the final SQL would return the intended answer using the observed evidence; explain this reasoning to yourself before finalizing (do not output the reasoning).

# Output
- If a logic fix is required: output ONLY the final executable SQL text (no fences, no commentary).
- If no change is needed: output EXACTLY `{self.quit_flag}`.
"""
        try:
            best_sql = self.llm.complete(prompt).text.strip()
            best_sql = sql_clean(best_sql)
        except Exception as exc:
            if data_logger:
                data_logger.info(f"LLM refinement failed: {exc}")
            return None

        if not best_sql:
            return None

        if data_logger:
            data_logger.info(f"{self.NAME}.syntax_refined_sql | sql={best_sql}")
        return best_sql

    def _optimize_single_sql(
            self,
            question: str,
            sql: str,
            schema: str = None,
            db_id: str = None,
            db_path: Union[str, Path] = None,
            db_type: str = "sqlite",
            credential: Optional[Dict] = None,
            data_logger=None,
    ):
        refined_sql = sql
        for turn in range(self.debug_turn_n):
            # get the decomposed meta sqls and execution results.
            feedback = self._get_meta_sql_feedback(sql, db_id, db_path, db_type, credential, data_logger)
            if feedback is None or len(feedback) == 0:
                # exit the debug turns.
                break
            origin_sql_status = feedback[0].get("status")
            try:
                if origin_sql_status:
                    # when the sql is executable, then refine logic module decide whether quit or not.
                    res = self._refine_logic_error(
                        question=question,
                        sql=sql,
                        schema=schema,
                        db_type=db_type,
                        feedback=[row for row in feedback if row.get("status")],
                        data_logger=data_logger
                    )
                    if self.quit_flag in res:
                        break
                else:
                    res = self._refine_syntax_schema_error(
                        question=question,
                        sql=sql,
                        schema=schema,
                        db_type=db_type,
                        feedback=[row for row in feedback if not row.get("status")],
                        data_logger=data_logger
                    )

                if res is None:
                    if data_logger:
                        data_logger.info("No valid refined sql existing, skip this turn.")
                    continue
                refined_sql = res

            except Exception as e:
                if data_logger:
                    data_logger.info(
                        f"An error occurred during the SQL refinement process, skipping this round. The error message is: {e}")
                continue

        return refined_sql

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, Path, List[str], List[Path]] = None,
            data_logger=None,
            **kwargs
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        if self.dataset is None:
            raise ValueError("Dataset is required for LinkAlignOptimizer")

        row = self.dataset[item]
        question = row['question']
        db_type = row['db_type']
        db_id = row.get("db_id")
        db_size = row.get("db_size", -1)
        db_path = Path(self.dataset.db_path) / (
                db_id + ".sqlite") if self.dataset.db_path and db_type == "sqlite" else None
        credential = self.dataset.credential if hasattr(self.dataset, 'credential') else None

        # Load and process schema using base class method
        schema = self.process_schema(schema, item)

        # Load schema_links if not provided
        if schema_links is None:
            schema_links = row.get("schema_links", "None")

        # Load pred_sql using base class method
        sql_list, _ = self.load_pred_sql(pred_sql, item)
        if data_logger:
            data_logger.info(f"{self.NAME}.input_sql_count | count={len(sql_list)}")

        # Decompose SQL into multiple meta-SQL set,
        # Returns the execution results and error messages for all meta-SQLs.
        def process_sql(sql):
            """question: str,
            sql: str,
            schema: str = None,
            db_id: str = None,
            db_path: Union[str, Path] = None,
            db_type: str = "sqlite",
            credential: Optional[Dict] = None,
            data_logger=None,"""
            if db_size > 500 and schema_links:
                filter_schema = schema_links
            else:
                filter_schema = schema
            final_sql = self._optimize_single_sql(
                question, sql, filter_schema, db_id, db_path, db_type, credential, data_logger=data_logger
            )
            return final_sql

        optimized_sqls = []
        if self.open_parallel and len(sql_list) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_sql, sql) for sql in sql_list]
                for future in concurrent.futures.as_completed(futures):
                    optimized_sqls.append(future.result())
        else:
            for sql in sql_list:
                optimized_sqls.append(process_sql(sql))

        # Save results using base class method
        output = self.save_output(optimized_sqls, item, row.get("instance_id"))

        logger.info(f"LinkAlignOptimizer completed processing item {item}")
        if data_logger:
            data_logger.info(f"{self.NAME}.optimized_sql | output={optimized_sqls}")
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return output
