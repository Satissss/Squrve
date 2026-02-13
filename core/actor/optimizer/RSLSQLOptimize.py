import json
import re
import concurrent.futures
from typing import Union, List, Optional, Dict
from pathlib import Path
from os import PathLike
from loguru import logger
import pandas as pd

from core.actor.optimizer.BaseOptimize import BaseOptimizer
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from core.db_connect import get_sql_exec_result
from core.utils import parse_schema_from_df

@BaseOptimizer.register_actor
class RSLSQLOptimizer(BaseOptimizer):
    NAME = "RSLSQLOptimizer"

    SKILL = """# RSLSQLOptimizer

RSLSQLOptimizer refines SQL via RSL-style augmentation and self-correction: key_word_augmentation, condition_augmentation → enrich table_info → self_correction (execute → if empty, LLM fixes; repeat up to debug_turn_n). Advantage: augmentation enriches context; drawback: many LLM calls, depends on DB.

## Inputs
- `schema`: Database schema (str/path/dict/list). If absent, loaded from dataset.
- `schema_links`: If absent, loaded from row.
- `pred_sql`: SQL(s) to optimize. If absent, loaded from dataset.

## Output
`pred_sql` (list of SQL)

## Steps
1. Load schema, schema_links, pred_sql.
2. For each SQL: optimize_single_sql.
3. optimize_single_sql: key_word_augmentation, condition_augmentation → self_correction (execute → if empty fix by LLM; repeat).
4. Optional parallel processing for multiple SQLs.
5. Save and return pred_sql.
"""

    SELF_CORRECTION_PROMPT = '''You are an AI agent responsible for generating the correct SQL statements based on the following information:
- A small number of SQL Q&A pairs: used for reference and learning common query patterns.
- Database structure information: including table names, fields, relationships between tables (such as foreign keys, etc.).
- The first three rows of values in the table: sample data for understanding the content and data distribution of the table.
- User questions: queries or questions in natural language form.
- Query requirements and conditions: specific query requirements and conditions in user questions.
- Tables involved in SQL statements: tables involved in user questions.
- Auxiliary query conditions: additional query conditions that may affect the generation of SQL statements.
- Hint: Information for prompting, this message is very important.

Your main tasks are:

1. Parse user questions:
   - Use natural language processing (NLP) techniques to parse user questions and extract query requirements and conditions.

2. Refer to SQL Q&A pairs:
    - Use the provided SQL Q&A pairs as a reference to understand common query patterns and SQL statement structures.

3. Analyze database structure information:
    - Based on the database structure information, understand the fields and relationships of the table, and build the basic framework of the SQL statement.

4. Check sample data:
    - Analyze the data characteristics based on the first three rows of the table values to help determine how to construct query conditions and filter results.

5. Generate SQL statements:
    - Based on user questions, query requirements and conditions, tables involved, and auxiliary query conditions, build a complete SQL statement.

6. Verification and optimization:
    - Check whether the generated SQL statement is logical and optimize it if necessary.

### Input:
- SQL Q&A pairs: a small number of example SQL Q&A pairs.
- Database structure information: including table names, fields, relationships between tables (such as foreign keys, etc.).
- The first three rows of values in the table: sample data.
- User questions: queries or questions in natural language form.
- Query requirements and conditions: specific query requirements and conditions in user questions.
- Auxiliary query conditions: additional query conditions.
- Hint: Information for prompting, this message is very important.

### Output:
- Return the result in json format, the format is {"sql": "SQL statement that meets the user question requirements"}

### Note:
- Ensure that the SQL statement accurately reflects the query requirements and conditions in the user question.
- Reasonably construct the query logic based on the database structure and sample data.
- When generating SQL statements, consider all the provided information to ensure the correctness and efficiency of the statement.
- If the SQL statement is incorrect or inefficient, make improvements. Ensure that the statement is both efficient and accurate.
- Hint: Information for prompting, this message is very important.
- In the generated SQL statement, table names and field names need to be enclosed in backquotes, such as `table_name`, `column_name`.
- In the generated SQL statement, table names and field names must be correct to ensure the correctness and efficiency of the statement.
'''

    KEY_WORD_AUG_INSTRUCTION = '''
You are an AI tasked with determining whether SQL statements need to use the following keywords or operations based on database structure information, the first three rows of a table, and user questions: `DISTINCT`, fuzzy matching, exact matching, `INTERSECT`, `UNION`, etc. Your main tasks are:

1. Understand user questions:
   - Parse user questions, extract key query requirements, such as whether to remove duplicates, fuzzy matching, exact matching, etc.

2. Analyze database structure information:
    - Based on the provided database structure information, understand table fields and data types, and determine whether it is necessary to use `DISTINCT`, fuzzy matching, or other keywords.

3. Check sample data:
    - Analyze data characteristics based on the first three rows of the table, determine whether duplicate data exists, and whether fuzzy matching is needed.

4. Determine keywords and operations:
    - Based on user questions, database structure, and sample data, determine whether the following keywords are needed:
      - Fuzzy matching (LIKE): Used to match similar strings.
      - Exact matching (=): Used for precise matching.
      - `INTERSECT`: Used to obtain the intersection of two query results.
      - `UNION`: Used to merge two query results.

5. Generate suggestions:
    - Return SQL statement keywords for the user question.

### Input:
- Database structure information: including table names, fields, relationships between tables (such as foreign keys), etc.
- The first three rows of the table: sample data to help understand the table content.
- User questions: queries or questions in natural language form.

### Output:
- Suggested SQL keywords: such as fuzzy matching `LIKE`, exact matching `=`, `INTERSECT`, `UNION`, etc.
- Return the results in json format, in the format {"sql_keywords": ["keyword1", "keyword2", ...]}

### Procedure:
1. Parse user questions: Extract key query requirements from the questions.
2. Analyze database structure information: Understand table fields and data types.
3. Check sample data: Analyze data characteristics to determine whether deduplication or fuzzy matching is needed.
4. Determine keywords and operations: Generate appropriate SQL keywords and operation suggestions based on the above information.
5. Generate results: Output suggested SQL keywords and operations.

### Note:
- Ensure that you understand the query requirements in user questions to accurately suggest SQL keywords and operations.
- Based on database structure and sample data, make reasonable judgments on whether specific SQL keywords or operations are needed.
- If user questions involve multiple query requirements, consider all requirements to generate suggestions.
'''

    CONDITION_AUG_INSTRUCTION = '''
You are an intelligent agent responsible for identifying the conditions in the user's question and clarifying the relationships between these conditions. Your main tasks are:

1. Understand the user's question: parse the user's question and extract all the conditions in the question.
2. Identify conditions:
   - Identify specific conditions from the user's question. For example, "age over 30" or "income over 5000".
3. Generate output:
    - List all identified conditions.

### Input:
- User question: a natural language query or question.

### Output:
- Condition list: all conditions extracted from the user question.

### Operation steps:
1. Parse the user's question: use natural language processing techniques to extract the conditions and relationships in the question.
2. Identify conditions: based on the parsing results, identify all conditions in the user's question.
3. Generate results: form a list of conditions and relationships and return them to the user.
4. Return in json format, format: {"conditions": ["condition1", "condition2", ...]}.

### Note:
- Ensure that all conditions in the user's question are correctly extracted and understood.
- If the user's question contains complex conditions or multiple relationships, please make a reasonable judgment based on the context.
'''

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm=None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/optimized_sql",
            debug_turn_n: int = 5,
            open_parallel: bool = True,
            max_workers: Optional[int] = None,
            db_path: Optional[Union[str, Path]] = None,
            credential: Optional[dict] = None,
            use_external: bool = True,
            use_few_shot: bool = True,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.debug_turn_n = debug_turn_n
        self.use_external = use_external
        self.use_few_shot = use_few_shot

        self.db_path = db_path or (dataset.db_path if dataset else None)
        self.credential = credential or (dataset.credential if dataset else None)

        # Load column meanings
        self.column_meaning = load_dataset("files/datasets/column_meaning.json") or {}

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

    def parse_json_response(self, response):
        """
        Robust JSON parsing with multiple fallback strategies
        """
        if not response or not isinstance(response, str):
            logger.warning(f"Invalid response type: {type(response)}")
            return {"sql": "SELECT 1"}

        # Clean the response
        response = response.strip()

        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from the response
        json_patterns = [
            r'\{.*\}',  # Basic JSON object
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # Try fixing common issues
        fixed_response = response

        # Fix unescaped backslashes
        fixed_response = fixed_response.replace("\\", "\\\\")

        # Fix unescaped quotes in SQL strings
        sql_pattern = r'["\'](SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH).*?["\']'

        def escape_sql_quotes(match):
            sql_content = match.group(0)
            if sql_content.startswith('"') and sql_content.endswith('"'):
                sql_content = sql_content[1:-1].replace('"', '\\"')
                return f'"{sql_content}"'
            elif sql_content.startswith("'") and sql_content.endswith("'"):
                sql_content = sql_content[1:-1].replace("'", "\\'")
                return f"'{sql_content}'"
            return match.group(0)

        fixed_response = re.sub(sql_pattern, escape_sql_quotes, fixed_response, flags=re.IGNORECASE | re.DOTALL)

        # Try parsing the fixed response
        try:
            return json.loads(fixed_response)
        except json.JSONDecodeError:
            pass

        # Try to fix common JSON issues
        try:
            # Remove any trailing commas
            fixed_response = re.sub(r',(\s*[}\]])', r'\1', fixed_response)
            # Fix missing quotes around keys
            fixed_response = re.sub(r'(\w+):', r'"\1":', fixed_response)
            return json.loads(fixed_response)
        except json.JSONDecodeError:
            pass

        # Last resort: try to construct a minimal valid JSON
        try:
            # Look for sql field specifically
            sql_match = re.search(r'"sql"\s*:\s*["\']([^"\']*(?:\\.[^"\']*)*)["\']', response, re.DOTALL)
            if sql_match:
                sql_content = sql_match.group(1)
                # Clean up the SQL content
                sql_content = sql_content.replace('\n', ' ').replace('\r', ' ')
                return {"sql": sql_content}
        except:
            pass

        # Try to extract any SQL-like content
        try:
            sql_match = re.search(
                r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH).*?(?:FROM|INTO|UPDATE|DELETE|CREATE|ALTER|DROP|$)',
                response, re.IGNORECASE | re.DOTALL)
            if sql_match:
                sql_content = sql_match.group(0).strip()
                sql_content = sql_content.replace('\n', ' ').replace('\r', ' ')
                return {"sql": sql_content}
        except:
            pass

        # If all else fails, return a default structure
        logger.warning(f"Failed to parse JSON response: {response[:200]}...")
        return {"sql": "SELECT 1"}

    def get_db_path(self, db_id):
        return Path(self.db_path) / f"{db_id}/{db_id}.sqlite"

    def execute_sql(self, sql, db_id):
        try:
            db_path = self.get_db_path(db_id)
            df, err = get_sql_exec_result("sqlite", sql_query=sql, db_path=db_path)
            if err:
                return 0, 0, f"Error: {err}"
            row_count = len(df) if df is not None else 0
            column_count = len(df.columns) if df is not None and not df.empty else 0
            result = str(df.head(5).to_dict(orient="records")) if df is not None else ""
            return row_count, column_count, result
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            return 0, 0, f"Error: {str(e)}"

    def get_all_tables_from_schema(self, schema_df):
        if schema_df is None or schema_df.empty:
            return []
        return schema_df['table_name'].unique().tolist()

    def get_table_columns_from_schema(self, schema_df, table_name):
        if schema_df is None or schema_df.empty:
            return []
        table_schema = schema_df[schema_df['table_name'] == table_name]
        return table_schema['column_name'].tolist()

    def get_all_schema_from_df(self, schema_df):
        if schema_df is None or schema_df.empty:
            return []
        schema_list = []
        for _, row in schema_df.iterrows():
            schema_list.append(f"{row['table_name']}.{row['column_name']}")
        return schema_list

    def get_simple_ddl_from_schema(self, schema_df, tables=None, columns=None):
        if schema_df is None or schema_df.empty:
            return "#\n# ", {}

        if tables is None:
            tables = self.get_all_tables_from_schema(schema_df)

        table_list = {}
        simple_ddl = "#\n# "

        for table in tables:
            if columns:
                col_list = [col.split(".")[1].strip("`") for col in columns if col.split(".")[0] == table]
            else:
                col_list = self.get_table_columns_from_schema(schema_df, table)

            simple_ddl += f"{table}(" + ",".join([f"`{col}`" for col in col_list]) + ")\n# "
            table_list[table] = [f"`{col}`" for col in col_list]

        return simple_ddl.strip(), table_list

    def get_ddl_data_from_schema(self, schema_df, tables, table_list):
        if schema_df is None or schema_df.empty:
            return "# "

        simplified_ddl_data = []
        for table in tables:
            if table not in table_list:
                continue

            col_str = ",".join(table_list[table])
            # 从schema中获取示例数据
            table_schema = schema_df[schema_df['table_name'] == table]
            test = ""
            for _, row in table_schema.iterrows():
                col_name = row['column_name']
                sample_rows = row.get('sample_rows', [])
                if isinstance(sample_rows, list) and len(sample_rows) > 0:
                    vals = [str(sample_rows[i]) if i < len(sample_rows) else "" for i in
                            range(min(3, len(sample_rows)))]
                else:
                    vals = ["", "", ""]
                test += f"{col_name}[{','.join(vals)}],"

            if test:
                simplified_ddl_data.append(f"{table}({test[:-1]})")

        ddls_data = "# " + ";\n# ".join(simplified_ddl_data) + ";\n# "
        return ddls_data

    def get_foreign_key_from_schema(self, schema_df, tables=None):
        if schema_df is None or schema_df.empty:
            return "#\n# "

        if tables is None:
            tables = self.get_all_tables_from_schema(schema_df)

        # 这里需要根据实际的schema结构来提取外键信息
        # 由于用户提供的schema格式中没有明确的外键信息，我们返回空的外键信息
        foreign_str = "#\n# "
        return foreign_str.strip()

    def get_explanation_from_schema(self, schema_df, tables, columns):
        if schema_df is None or schema_df.empty:
            return ""

        explanation = ""
        columns_lower = [col.replace("`", "").lower() for col in columns]

        for _, row in schema_df.iterrows():
            table_name = row['table_name']
            col_name = row['column_name']
            col_desc = row.get('column_descriptions', '')

            if table_name in tables and f"{table_name}.`{col_name}`".lower() in columns_lower:
                if col_desc:
                    explanation += f"# {table_name}.{col_name}: {col_desc}\n"

        return explanation

    def extract_from_text(self, text, db_schema):
        pred = []
        text_lower = text.lower()
        for item in db_schema:
            try:
                if '.' not in item:
                    continue
                table, column = item.lower().split('.', 1)  # Split only on first occurrence
                if table == 'sqlite_sequence':
                    continue
                if column in text_lower:
                    pred.append(item)
            except (ValueError, AttributeError):
                # Skip items that don't have the expected format
                continue
        pred = list(set(pred))
        tables = list(set([p.split('.')[0] for p in pred if '.' in p]))
        columns = [p.replace('.', '.`') + '`' for p in pred if '.' in p]
        return {"tables": tables, "columns": columns}

    def key_word_augmentation(self, table_info, question, evidence):
        prompt = table_info.strip() + '\n\n### definition: ' + evidence + "\n### Question: " + question + "\n\nReturn your answer in JSON format as specified."
        response = self.llm.complete(self.KEY_WORD_AUG_INSTRUCTION + "\n" + prompt).text
        return self.parse_json_response(response)

    def condition_augmentation(self, question):
        prompt = question + "\n\nReturn your answer in JSON format as specified."
        response = self.llm.complete(self.CONDITION_AUG_INSTRUCTION + "\n" + prompt).text
        return self.parse_json_response(response)

    def self_correction(self, table_info, pre_sql, db_id):
        prompt = self.SELF_CORRECTION_PROMPT + '\n' + table_info + '\n\nReturn your answer in JSON format as {"sql": "your sql"}.'
        num = 0
        while num < self.debug_turn_n:
            try:
                row_count, column_count, result = self.execute_sql(pre_sql, db_id)
            except Exception as e:
                logger.error(f"SQL execution error: {e}")
                break
            if num > 0:
                prompt += f"\n### Buggy SQL: {pre_sql.strip()}\n### The result of the buggy SQL is [{result.strip()}]. Please fix the SQL to get the correct result."
            response = self.llm.complete(prompt).text
            sql_dict = self.parse_json_response(response)
            pre_sql = sql_dict['sql'].strip()
            if row_count > 0 or column_count > 0:
                break
            num += 1
        return pre_sql.replace('\n', ' ')

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
            item: Optional[int] = None
    ) -> str:
        try:
            # Since schema is already a string format, we need to work with it directly
            # For RSLSQLOptimizer, we'll use a simplified approach that doesn't require DataFrame conversion

            # Extract table and column information from the schema string
            table_info_aug = schema  # Use the schema string directly

            # Get evidence and example from dataset if available
            evidence = ""
            example = ""
            if self.dataset and item is not None:
                row = self.dataset[item]
                if row:
                    evidence = row.get('evidence', '')
                    if self.use_external:
                        external_knowledge = self.load_external_knowledge(row.get("external", None))
                        if external_knowledge:
                            evidence = evidence + "\n" + external_knowledge if evidence else external_knowledge
                            logger.debug("已加载外部知识")
                    example = load_dataset(row.get('reasoning_examples', '')) if self.use_few_shot else ''

            word_aug = self.key_word_augmentation(table_info_aug, question, evidence)
            cond_aug = self.condition_augmentation(question)
            table_info = table_info_aug + f'\n### sql_keywords: {word_aug.get("sql_keywords", [])}\n### conditions: {cond_aug.get("conditions", [])}'

            # Ensure db_id is not None for SQL execution
            if db_id is None:
                logger.warning("db_id is None, skipping SQL execution in self-correction")
                optimized_sql = sql
            else:
                optimized_sql = self.self_correction(table_info, sql, db_id)
        except Exception as e:
            logger.error(f"Error in optimizing SQL: {e}")
            optimized_sql = sql

        return optimized_sql

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List, pd.DataFrame] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, Path, List[str], List[Path]] = None,
            data_logger=None,
            **kwargs
    ):
        logger.info(f"RSLSQLOptimizer processing item {item}")
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        if self.dataset is None:
            raise ValueError("Dataset is required for RSLSQLOptimizer")

        row = self.dataset[item]
        question = row['question']
        db_type = row.get('db_type', 'sqlite')
        db_id = row.get('db_id')
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
            
        def process_sql(sql):
            return self.optimize_single_sql(
                sql, question, schema, db_type, schema_links, db_id, db_path, credential, item=item
            )

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
        output = self.save_output(optimized_sqls, item, row.get("instance_id", item))

        logger.info(f"RSLSQLOptimizer completed processing item {item}")
        if data_logger:
            data_logger.info(f"{self.NAME}.optimized_sql | output={optimized_sqls}")
            data_logger.info(f"{self.NAME}.act end | item={item}")
            
        return output
