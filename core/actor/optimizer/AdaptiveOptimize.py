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
            debug_turn_n: int = 2,
            open_parallel: bool = True,
            max_workers: Optional[int] = None,
            quit_flag: str = "QUIT",
            skip_logic_refine: bool = False,
            domain: Literal["finance"] = None,
            domain_save_dir="../files/domain",
            top_k: int = 5,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.debug_turn_n = debug_turn_n
        self.quit_flag = quit_flag
        self.skip_logic_refine = skip_logic_refine
        self.domain = domain
        self.domain_save_dir = domain_save_dir
        self.top_k = top_k

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
You are a professional database query optimizer and SQL parsing engine. Your task is to decompose the input "complete raw SQL statement(s)" (which may contain one or more candidate SQLs) into a set of "atomic Meta SQL statements."

# Core Definitions
1. **Atomic Meta SQL:**
    - An atomic SQL should contain as few WHERE conditions, GROUP BY dimensions, or ORDER BY fields as possible.
    - Independence: Each Meta SQL must be executable independently by the database without depending on the execution results of other Meta SQLs (i.e., no runtime dependencies), enabling parallel execution.
    - Completeness: The combination of all Meta SQL result sets must contain all data necessary to reconstruct the original SQL results.
    - Schema Constraint: Must not introduce tables or columns that do not exist in the original SQL.

# Decomposition Rules
Follow the logic below to decompose the input SQL:

## 1. **Single Table Decomposition**:
    If a single-table query contains multiple logical components, it must be decomposed into multiple independent SQLs:
* **WHERE Decomposition**:
    * Decompose `WHERE condition1 AND condition2` into two independent SQLs: `SELECT ... WHERE condition1` and `SELECT ... WHERE condition2`.
    * *Purpose*: Ensure each atomic SQL focuses on a single filtering dimension.
* **GROUP BY Decomposition**:
    * Decompose `GROUP BY col1, col2` into aggregation queries targeting different dimensions (if semantically permissible). For example, `SELECT count(*) ... GROUP BY A, B` should be decomposed into aggregations focusing on A and aggregations focusing on B.
* **ORDER BY Decomposition**:
    * Decompose `ORDER BY col1, col2` into independent sorting queries, unless col2's sorting strongly depends on col1's grouping context.
* **Cross-Clause Combination**: 
    * **Clause Isolation Rule**: When WHERE, GROUP BY, or ORDER BY coexist in the same query, further decompose into atomic SQLs that each contain only ONE type of clause (either WHERE only, or GROUP BY only, or ORDER BY only), ensuring maximum granularity.

## 2. **Join Decomposition**
* For `TableA JOIN TableB ON ... WHERE A.col=1 AND B.col=2`:
    * First decompose into queries targeting TableA and queries targeting TableB.
    * Then apply the "Clause Fission Rule" to the decomposed queries (e.g., if TableA has multiple WHERE conditions, continue splitting).

## 3. Subquery Handling
* **Standard Decomposition**: Extract subqueries as independent SQLs.
* **Correlated Subqueries (IN/EXISTS)**:
    * For `A WHERE id IN (SELECT id FROM B WHERE condition)`:
    * Generate Meta SQL 1: `SELECT id FROM B WHERE condition`
    * Generate Meta SQL 2: `WITH Filtered_B AS (SELECT id FROM B WHERE condition) SELECT * FROM A WHERE id IN (SELECT id FROM Filtered_B)`
    * *Note*: Meta SQL 2 must contain a complete logical closure to ensure independent executability.
    
## 4. Multiple Candidate Input Handling (Multiple Candidates):
* Input may contain multiple candidate SQLs with identical semantics but different syntax.
* You must process each SQL in the list and decompose them all.
* **Deduplication & Merging**: Place all decomposed Meta SQLs into a single set, remove exact duplicate strings, and return a deduplicated list.

# Output Format
* Return only a parseable Python List [str].
* List elements are Meta SQLs in string format.
* Strictly prohibit any Markdown formatting (such as ```json ... ```), explanatory text, or code block markers—output only plain text list.


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
Please process the following input raw SQL statement(s):

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

    def _load_external_domain(self, question: str):
        from rank_bm25 import BM25Okapi
        import jieba
        # For specific domain, like finance. Solving the logic errors need some external knowledge.
        if not self.domain or not self.domain_save_dir:
            return None
        data_path = Path(self.domain_save_dir) / (self.domain + ".json")
        data = load_dataset(data_path)
        if data is None or not isinstance(data, dict):
            logger.info("No Valid External Domain Knowledge available.")
            return None
        try:
            corpus = list(data.keys())
            tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
            tokenized_query = list(jieba.cut(question))

            bm25 = BM25Okapi(tokenized_corpus)

            # 计算分数
            scores = bm25.get_scores(tokenized_query)

            # 获取top_k的索引
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]

            # 返回结果
            return {corpus[i]: data[corpus[i]] for i in top_indices}
        except Exception as e:
            logger.info(f"Errors in BM25 ranking! Error: {e}")
            return None

    def wrap_external_domain(self, external_domain: str | Dict):
        if not external_domain:
            return ""

        if isinstance(external_domain, dict):
            domain_lis = []
            for k, v in external_domain.items():
                domain_lis.append(f"**Name:**  {k}\n**Explanation:**  {v}")
            external_domain = "\n\n".join(domain_lis)

        domain = f"""# {self.domain} knowledge:
**Note:** 
Below are explanations of technical terms that appear in the question. Some technical term explanations may be unrelated to the question—please ignore them.

{external_domain}
"""
        return domain

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
                if record.get("status"):
                    res_msg = record.get("res")
                    if isinstance(res_msg, pd.DataFrame):
                        res_msg = res_msg.head(5).to_dict(orient="records")
                        res_msg = res_msg or "Executable with no syntax errors, but no data was found."
                        snippet.append(f"Query results: {str(res_msg)}")
                else:
                    err_msg = record.get("err")
                    snippet.append(f"Error: {err_msg}")

                segments.append("\n".join(snippet))
            return "\n\n".join(segments)

        feedback_txt = _summarize_feedback()
        cleaned_sql = sql_clean(sql)
        prompt = f"""# Instruction
When executing SQL below, some errors occurred. Please fix the SQL based on the **query**, **database info**, and the **error feedback** from both the **original SQL** and the **atomic SQLs**. 
The atomic SQLs are decomposed parts of the original SQL designed to **isolate and expose specific syntax errors**. Use them to identify and resolve all issues in the original SQL.
Analyze the original SQL error to understand the general syntax issue. Do not address logical errors or query results; other modules will handle those.

# Solve the task step by step:

### Phase 1: Diagnosis & Isolation
1. **Analyze the Original Error**:
   - Read the [ORIGINAL_SQL] error to understand the high-level syntax error.

2. **Examine each atomic SQL result**:
    - **Success:** The syntax in this specific clause is **CORRECT** for {db_type}. **DO NOT modify this part**; the error lies elsewhere.
    - **Failure:**
        - **Same Error:** The bug is locally isolated to this clause (e.g., a typo).
        - **New Error:** A hidden bug (masked in the original SQL) is revealed.
    - **Action:** You must rewrite the failing logic to comply with **{db_type}** syntax (e.g., replace unsupported functions).

### Phase 2: Systematic Repair
3. **Execute Repair:**
   - **Dialect Translation:** STRICTLY replace incompatible syntax with **{db_type}** alternatives (e.g., for SQLite: replace `EXTRACT(YEAR...)` with `strftime('%Y'...)`).
   - **Syntax Correction:** Fix structural errors (e.g., missing keywords, incorrect operators) identified in the error logs.

4. **Final Verification:**
   - Ensure the rewritten SQL is valid for **{db_type}** and strictly references columns defined in the **Schema**.
   
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
            if data_logger:
                data_logger.info(prompt)
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
            external_domain: str = None,
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
                    res_msg = res_msg or "Executable with no syntax errors, but no data was found."
                    snippet.append(f"Query results: {str(res_msg)}")
                segments.append("\n".join(snippet))
            return "\n\n".join(segments)

        feedback_txt = _summarize_feedback()
        external_domain = self.wrap_external_domain(external_domain) or ""
        cleaned_sql = sql_clean(sql)
        prompt = f"""# Role
You are a Senior SQL Logic Auditor and Data Engineer. Your objective is to verify if a generated SQL query accurately answers a natural language question based on the provided Database Schema and Query Execution Results.

# Solve the task step by step:
### Step 1: Diagnosis based on Atomic Evidence
Analyze the Execution Evidences (where the original SQL is decomposed into atomic sub-queries) and the External Domain Knowledge to identify logical errors.
1.  **Atomic Check(High Priority):** 
    - If a key atomic unit (e.g., a core WHERE filter or JOIN) returns empty, scrutinize its logic. Treat this as a signal of a potential issue, but remember that it could also be due to absent data, not necessarily a logic error. Ask yourself: Does the atomic condition accurately reflect the original query intent? Does it contradict external domain knowledge?
2.  **Domain Consistency:** 
    - An empty final result warrants a check for contradictions between the SQL logic (e.g., in WHERE/JOIN clauses) and the External Domain Knowledge. A logic error is likely only if a clear contradiction exists and the result is empty.
3.  **Verification:** 
    - When all atomic units return data, Ensure JOINs, aggregations, and subquery linkages correctly combine the atomic results to answer the original question. Correct atomic sub-results do not guarantee correct final output—structural assembly must preserve logical intent..
    
**Decision Point:**
* If the SQL is logically sound and results are accurate: **Stop and output `{self.quit_flag}` only.**
* If a logical error exists: **Proceed to Step 2.**

### Step 2: Error Localization & Classification
Reason step-by-step to pinpoint the error. Classify the mistake using this taxonomy:
* **LOGIC_QUERY_MISMATCH:** SQL executes but fails the user's intent (e.g., wrong filter column, missing predicate).
* **LOGIC_VALUE:** Incorrect literals/parameters (e.g., string case mismatch, wrong date format, improper NULL handling).
* **LOGIC_JOIN:** Flawed relationships (e.g., Cartesian product, wrong JOIN type, missing ON condition).
* **LOGIC_AGGREGATION:** Grouping errors (e.g., missing GROUP BY, wrong function `COUNT` vs `SUM`).
* **LOGIC_SUBQUERY:** Subquery structure errors (e.g., correlated subquery misuse, wrong nesting).

*Self-Correction Instruction:* explicit state *which* clause caused the failure based on the Atomic Evidence.

### Step 3: Dialect-Specific Correction
Draft the corrected SQL.
1.  **Fix the Logic:** Address the specific error identified in Step 2.
2.  **Syntax Compliance:** Ensure the syntax is strictly valid for **{db_type}**.
3.  **Minimal Intervention:** Only change the erroneous parts; preserve the correct logic of other clauses.

# Question: 
{question or 'N/A'}

{external_domain}

# Database Schema: ({db_type or 'unknown db'})
{schema}

# Original SQL
{cleaned_sql}

# Execution Evidences 
{feedback_txt}

# Output
- If a logic fix is required:
    * Output ONLY the final optimized SQL text.
    * No explanations, no markdown, no commentary, no prefixes, no suffixes.
- If no change is needed: output EXACTLY `{self.quit_flag}`.
"""
        try:
            if data_logger:
                data_logger.info(prompt)
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
        external_domain = self._load_external_domain(question) or ""

        for turn in range(self.debug_turn_n):
            # get the decomposed meta sqls and execution results.
            feedback = self._get_meta_sql_feedback(refined_sql, db_id, db_path, db_type, credential, data_logger)
            if feedback is None or len(feedback) == 0:
                # exit the debug turns.
                break
            origin_sql_status = feedback[0].get("status")
            try:
                if origin_sql_status:
                    # when the sql is executable, then refine logic module decide whether quit or not.
                    res = self._refine_logic_error(
                        question=question,
                        sql=refined_sql,
                        schema=schema,
                        db_type=db_type,
                        feedback=[row for row in feedback if row.get("status")],
                        external_domain=external_domain,
                        data_logger=data_logger
                    ) if not self.skip_logic_refine else self.quit_flag
                    if self.quit_flag in res:
                        break
                else:
                    res = self._refine_syntax_schema_error(
                        question=question,
                        sql=refined_sql,
                        schema=schema,
                        db_type=db_type,
                        feedback=feedback,
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
            raise ValueError("Dataset is required for AdaptiveOptimizer")

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

        logger.info(f"AdaptiveOptimizer completed processing item {item}")
        if data_logger:
            data_logger.info(f"{self.NAME}.optimized_sql | output={optimized_sqls}")
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return output
