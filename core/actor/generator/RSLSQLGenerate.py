from os import PathLike
from typing import Union, Optional, Dict, Any
import json
import pandas as pd
from pathlib import Path
from loguru import logger
import re

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, load_dataset, single_central_process
from core.utils import save_dataset
from core.db_connect import get_sql_exec_result

# Prompts from Instruction.py
TABLE_AUG_INSTRUCTION = '''
You are an intelligent agent responsible for identifying the database tables involved based on the user's questions and database structure information. Your main tasks are:

1. Understand user questions: parse user questions and extract keywords and intentions.
2. Obtain database structure information: Based on the provided database structure information, understand all tables and their relationships.
3. Identify relevant tables:
   - Based on the keywords and intentions in the user's questions, identify directly related tables.
   - Consider the situation of intermediate tables, such as connection tables or cross tables, which may involve the tables in the user's questions.
4. Generate a list of tables: Integrate directly related tables and intermediate tables to form the final list of tables.
5. Return the results in json format, the format is {"tables": ["table1", "table2", ...],"columns":["table1.`column1`","table2.`column2`",...]}

### Input:
- Database structure information: including table names, fields, and relationships between tables (such as foreign keys, etc.).
- User questions: queries or questions in natural language form.

### Output:
- List of database tables involved: including directly related tables and intermediate tables.

### Operation steps:
1. Parse user questions: extract keywords and intentions from the questions.
2. Identify key tables: preliminarily identify the direct tables related to the user's questions.
3. Check intermediate tables: Based on the database structure information, identify intermediate tables related to the direct tables.
4. Integrate the results: integrate direct tables and intermediate tables to form the final list of tables.
5. Output the results: return all table lists involved in the user's questions. Select the top 15 columns most relevant to the question for each table.

### Note:
- Ensure that all possible intermediate tables are considered, especially tables involving many-to-many relationships.
- Ensure that the output table list is unique and without duplicates.
'''

SQL_GENERATION_INSTRUCTION = '''
You are a smart agent responsible for generating the correct SQL statements based on the following information:
- A small number of SQL Q&A pairs: used for reference and learning common query patterns.
- Database structure information: including table names, fields, relationships between tables (such as foreign keys, etc.).
- The first three rows of values in the table: sample data for understanding the content and data distribution of the table.
- User questions: natural language queries or questions.
- Query requirements and conditions: specific query requirements and conditions in user questions.
- Tables involved in SQL statements: tables involved in user questions.
- Auxiliary query conditions: additional query conditions provided, which may affect the generation of SQL statements.
- definition: Information for prompts, this message is very important.

Your main tasks are:

1. Parse user questions:
   - Use natural language processing (NLP) techniques to parse user questions and extract query requirements and conditions.

2. Refer to SQL Q&A pairs:
    - Use the provided SQL Q&A pairs as a reference to understand common query patterns and SQL statement structures.

3. Analyze database structure information:
    - Based on the database structure information, understand the fields and relationships of the table, and build the basic framework of the SQL statement.

4. Check sample data:
    - Analyze the data characteristics based on the first three rows of the table, which helps to determine how to construct query conditions and filter results.

5. Generate SQL statements:
    - Based on user questions, query requirements and conditions, tables involved, and auxiliary query conditions, construct complete SQL statements.

6. Verification and optimization:
    - Check whether the generated SQL statement is logical and optimize it if necessary.

### Input:
- SQL Q&A pairs: a small number of example SQL Q&A pairs.
- Database structure information: including table names, fields, relationships between tables (such as foreign keys, etc.).
- The first three rows of values in the table: sample data.
- User questions: natural language queries or questions.
- Query requirements and conditions: specific query requirements and conditions in user questions.
- Auxiliary query conditions: additional query conditions.
- definition: Information for prompts, this message is very important.

### Output:
- Return the result in json format, the format is {"sql": "SQL statement that meets the user's question requirements"}

### Operation steps:
1. Parse user questions: extract query requirements and conditions from the questions.
2. Refer to SQL Q&A pairs: understand common query patterns and SQL statement structures.
3. Analyze database structure information: build the basic framework of the SQL statement.
4. Check sample data: determine query conditions and filter results.
5. Generate SQL statements: construct complete SQL statements.
6. Verification and optimization: ensure the logical correctness of the SQL statement and optimize it.

### Note:
- Ensure that the SQL statement accurately reflects the query requirements and conditions in the user questions.
- Reasonably construct query logic based on database structure and sample data.
- When generating SQL statements, consider all the information provided to ensure the correctness and efficiency of the statements.
- If the user question involves complex query requirements, please consider all requirements and conditions to generate SQL statements.

### The most important thing is to remember:
- definition: Information for prompts, this message is very important.
- In the generated SQL statement, table names and field names need to be enclosed in backticks, such as `table_name`, `column_name`.
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

BINARY_PROMPT = '''{table_info}

### Select the best SQL query to answer the  question:

{candidate_sql}

Your answer should be returned by json format.
{{
    "sql": "...",# your SQL query
}}
'''


class RSLSQLGenerator(BaseGenerator):
    NAME = "RSLSQLGenerator"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm=None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            use_external: bool = True,
            use_few_shot: bool = True,
            db_path: Optional[Union[str, PathLike]] = None,
            credential: Optional[dict] = None,
            **kwargs
    ):
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir
        self.use_external = use_external
        self.use_few_shot = use_few_shot

        # 安全地初始化 db_path 和 credential，检查 dataset 是否为 None
        if db_path is not None:
            self.db_path = db_path
        elif dataset is not None:
            self.db_path = dataset.db_path
        else:
            self.db_path = None

        if credential is not None:
            self.credential = credential
        elif dataset is not None:
            self.credential = dataset.credential
        else:
            self.credential = None

        # Load column meanings
        self.column_meaning = load_dataset("files/datasets/column_meaning.json") or {}

    @property
    def name(self):
        return self.NAME

    def _get_db_path(self, row, db_id, db_type):
        """
        Get database path/identifier based on db_type.
        Database-agnostic approach that works with get_sql_exec_result.
        """
        if db_type == "sqlite":
            # For SQLite, we need the actual file path
            if self.db_path:
                db_path = Path(self.db_path)
                if db_path.is_file():
                    return str(db_path)
                else:
                    return str(db_path / f"{db_id}.sqlite")
            else:
                return row.get('path_db', f"{db_id}.sqlite")
        elif db_type == "big_query":
            # For BigQuery, db_path is not used, only credential_path matters
            return row.get('project_id', db_id)
        elif db_type == "snowflake":
            # For Snowflake, db_path is actually the database name
            return row.get('database_name', db_id)
        else:
            # For other database types, return the db_id as identifier
            logger.debug(f"Using db_id as path for unsupported database type: {db_type}")
            return db_id

    def try_direct_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """Attempt direct JSON parsing."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def try_extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON using regex patterns."""
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
        return None

    def try_fix_common_issues(self, response: str) -> Optional[Dict[str, Any]]:
        """Fix common JSON issues and attempt parsing."""
        fixed_response = response.replace("\\", "\\\\")

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

        # Additional fixes
        fixed_response = re.sub(r',(\s*[}\]])', r'\1', fixed_response)
        fixed_response = re.sub(r'(\w+):', r'"\1":', fixed_response)

        try:
            return json.loads(fixed_response)
        except json.JSONDecodeError:
            return None

    def try_extract_sql_field(self, response: str) -> Dict[str, Any]:
        """Extract SQL from 'sql' field as fallback."""
        sql_match = re.search(r'"sql"\s*:\s*["\']([^"\']*(?:\\.[^"\']*)*)["\']', response, re.DOTALL)
        if sql_match:
            sql_content = sql_match.group(1).replace('\n', ' ').replace('\r', ' ')
            return {"sql": sql_content}
        return {"sql": "SELECT 1", "tables": [], "columns": [], "sql_keywords": [], "conditions": []}

    def try_extract_any_sql(self, response: str) -> Dict[str, Any]:
        """Extract any SQL-like content as last resort."""
        sql_match = re.search(
            r'(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH).*?(?:FROM|INTO|UPDATE|DELETE|CREATE|ALTER|DROP|$)',
            response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql_content = sql_match.group(0).strip().replace('\n', ' ').replace('\r', ' ')
            return {"sql": sql_content}
        return {"sql": "SELECT 1", "tables": [], "columns": [], "sql_keywords": [], "conditions": []}

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Robust JSON parsing with multiple fallback strategies."""
        if not response or not isinstance(response, str):
            logger.warning(f"Invalid response type: {type(response)}")
            return {"sql": "SELECT 1", "tables": [], "columns": [], "sql_keywords": [], "conditions": []}

        response = response.strip()

        # Try direct parsing
        result = self.try_direct_parse(response)
        if result:
            return result

        # Try extracting JSON
        result = self.try_extract_json(response)
        if result:
            return result

        # Try fixing common issues
        result = self.try_fix_common_issues(response)
        if result:
            return result

        # Try extracting sql field
        result = self.try_extract_sql_field(response)
        if result.get("sql") != "SELECT 1":
            return result

        # Try extracting any SQL
        result = self.try_extract_any_sql(response)
        if result.get("sql") != "SELECT 1":
            return result

        logger.warning(f"Failed to parse JSON response: {response[:200]}...")
        return {"sql": "SELECT 1", "tables": [], "columns": [], "sql_keywords": [], "conditions": []}

    def execute_sql(self, sql, db_id, db_type, row=None):
        """Execute SQL query and return row count, column count, and partial results."""
        try:
            db_path = self._get_db_path(row or {}, db_id, db_type)
            sql_args = {
                "db_type": db_type,
                "sql_query": sql,
                "db_path": db_path,
                "db_id": db_id,
                "credential_path": self.credential if self.credential else None
            }
            df, err = get_sql_exec_result(**sql_args)
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
        """从schema DataFrame中获取所有表名"""
        if schema_df is None or schema_df.empty:
            return []
        return schema_df['table_name'].unique().tolist()

    def get_table_columns_from_schema(self, schema_df, table_name):
        """从schema DataFrame中获取指定表的列名"""
        if schema_df is None or schema_df.empty:
            return []
        table_schema = schema_df[schema_df['table_name'] == table_name]
        return table_schema['column_name'].tolist()

    def get_all_schema_from_df(self, schema_df):
        """从schema DataFrame中获取所有schema信息"""
        if schema_df is None or schema_df.empty:
            return []
        schema_list = []
        for _, row in schema_df.iterrows():
            schema_list.append(f"{row['table_name']}.{row['column_name']}")
        return schema_list

    def get_simple_ddl_from_schema(self, schema_df, tables=None, columns=None):
        """从schema DataFrame生成简单的DDL"""
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

    def get_ddl_data_from_schema(self, schema_df, tables, table_list, db_id=None, db_type="sqlite"):
        """从schema DataFrame生成DDL数据信息"""
        if schema_df is None or schema_df.empty:
            return "# "

        simplified_ddl_data = []
        for table in tables:
            if table not in table_list:
                continue

            col_str = ",".join(table_list[table])

            # First try to get sample data from schema if available
            table_schema = schema_df[schema_df['table_name'] == table]
            has_sample_data = False

            if not table_schema.empty:
                # Check if we have sample_rows in schema
                sample_rows = table_schema.iloc[0].get('sample_rows', [])
                if isinstance(sample_rows, list) and len(sample_rows) > 0:
                    has_sample_data = True
                    test = ""
                    for _, row in table_schema.iterrows():
                        col_name = row['column_name']
                        sample_vals = row.get('sample_rows', [])
                        if isinstance(sample_vals, list) and len(sample_vals) > 0:
                            vals = [str(sample_vals[i]) if i < len(sample_vals) else "" for i in
                                    range(min(3, len(sample_vals)))]
                        else:
                            vals = ["", "", ""]
                        test += f"{col_name}[{','.join(vals)}],"

                    if test:
                        simplified_ddl_data.append(f"{table}({test[:-1]})")
                        continue

            # If no sample data in schema and db_id provided, query database
            if not has_sample_data and db_id is not None:
                try:
                    sql = f"SELECT {col_str} FROM `{table}` LIMIT 3"
                    sql_args = {
                        "db_type": db_type,
                        "sql_query": sql,
                        "db_path": self._get_db_path({}, db_id, db_type),
                        "db_id": db_id,
                        "credential_path": self.credential
                    }
                    df, err = get_sql_exec_result(**sql_args)
                    if err or df is None:
                        continue

                    col_names = df.columns.tolist()
                    data_rows = df.values.tolist()
                    test = ""
                    for idx, col in enumerate(col_names):
                        vals = [str(data_rows[r][idx]) if r < len(data_rows) else "" for r in range(3)]
                        test += f"{col}[{','.join(vals)}],"

                    if test:
                        simplified_ddl_data.append(f"{table}({test[:-1]})")
                except Exception as e:
                    logger.warning(f"Failed to get sample data for table {table}: {e}")
                    continue

        ddls_data = "# " + ";\n# ".join(simplified_ddl_data) + ";\n# "
        return ddls_data

    def get_foreign_key_from_schema(self, schema_df, tables=None):
        """从schema DataFrame中提取外键信息"""
        if schema_df is None or schema_df.empty:
            return "#\n# "

        if tables is None:
            tables = self.get_all_tables_from_schema(schema_df)

        # 这里需要根据实际的schema结构来提取外键信息
        # 由于用户提供的schema格式中没有明确的外键信息，我们返回空的外键信息
        foreign_str = "#\n# "
        return foreign_str.strip()

    def get_explanation_from_schema(self, schema_df, tables, columns):
        """从schema DataFrame中获取列描述信息"""
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

    def table_column_selection(self, table_info, question, evidence):
        prompt = table_info.strip() + '\n\n### definition: ' + evidence + "\n### Question: " + question + "\n\nReturn your answer in JSON format as specified."
        response = self.llm.complete(TABLE_AUG_INSTRUCTION + "\n" + prompt).text
        return self.parse_json_response(response)

    def preliminary_sql_gen(self, table_info, table_column, example, question, evidence):
        table_info += f'### tables: {table_column["tables"]}\n'
        table_info += f'### columns: {table_column["columns"]}\n'
        prompt = example.strip() + "\n\n### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n### definition: ' + evidence + "\n### Question: " + question + "\n\nReturn your answer in JSON format as {\"sql\": \"your sql\"}."
        response = self.llm.complete(SQL_GENERATION_INSTRUCTION + "\n" + prompt).text
        return self.parse_json_response(response)['sql'].replace('\n', ' ')

    def extract_from_text(self, text, db_schema):
        """Extract schema links from text (evidence or SQL) using column name matching."""
        pred = []
        if not text or not db_schema:
            return {"tables": [], "columns": []}
        
        text_lower = text.lower()
        for item in db_schema:
            try:
                if '.' not in item:
                    continue
                parts = item.split('.')
                if len(parts) < 2:
                    continue
                table, column = parts[0].lower(), '.'.join(parts[1:]).lower()
                if table == 'sqlite_sequence':
                    continue
                # More flexible column matching
                column_clean = column.strip('`').strip()
                if column_clean in text_lower:
                    pred.append(item)
            except (ValueError, AttributeError) as e:
                logger.debug(f"Skipping malformed schema item {item}: {e}")
                continue
        
        pred = list(set(pred))
        tables = list(set([p.split('.')[0] for p in pred if '.' in p]))
        columns = []
        for p in pred:
            if '.' in p:
                table_part, col_part = p.split('.', 1)
                if not col_part.startswith('`'):
                    col_part = '`' + col_part
                if not col_part.endswith('`'):
                    col_part = col_part + '`'
                columns.append(f"{table_part}.{col_part}")
        return {"tables": tables, "columns": columns}

    def merge_schema_links(self, sl_sql, sl_llm, sl_hint):
        tables = list(set(sl_llm['tables'] + sl_sql['tables'] + sl_hint['tables']))
        columns = list(set(sl_llm['columns'] + sl_sql['columns'] + sl_hint['columns']))
        return {"tables": [t.lower() for t in tables], "columns": [c.lower() for c in columns]}

    def filter_schema_links(self, schema_links, db_schema):
        """Filter schema links to ensure they exist in the actual database schema."""
        if not schema_links or not db_schema:
            return {"tables": [], "columns": []}
        
        pred = []
        db_schema_lower = [s.lower() for s in db_schema if s]
        
        for col in schema_links.get('columns', []):
            if not col:
                continue
            col_clean = col.replace('`', '').lower().strip()
            for sch in db_schema:
                if sch and sch.lower() == col_clean:
                    pred.append(sch)
                    break
        
        pred = list(set(pred))
        tables = list(set([p.split('.')[0] for p in pred if '.' in p and p.split('.')[0]]))
        columns = []
        for p in pred:
            if '.' in p:
                table_part, col_part = p.split('.', 1)
                if not col_part.startswith('`'):
                    col_part = '`' + col_part
                if not col_part.endswith('`'):
                    col_part = col_part + '`'
                columns.append(f"{table_part}.{col_part}")
        
        return {"tables": tables, "columns": columns}

    def key_word_augmentation(self, table_info, question, evidence):
        prompt = table_info.strip() + '\n\n### definition: ' + evidence + "\n### Question: " + question + "\n\nReturn your answer in JSON format as specified."
        response = self.llm.complete(KEY_WORD_AUG_INSTRUCTION + "\n" + prompt).text
        return self.parse_json_response(response)

    def condition_augmentation(self, question):
        prompt = question + "\n\nReturn your answer in JSON format as specified."
        response = self.llm.complete(CONDITION_AUG_INSTRUCTION + "\n" + prompt).text
        return self.parse_json_response(response)

    def sql_generation_aug(self, table_info, table_aug, word_aug, cond_aug, example, question, evidence):
        table_info += f'\n### sql_keywords: {word_aug["sql_keywords"]}\n'
        table_info += f'### tables: {table_aug["tables"]}\n'
        table_info += f'### columns: {table_aug["columns"]}\n'
        table_info += f'### conditions: {cond_aug["conditions"]}'
        prompt = example.strip() + '\n\n### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n' + table_info.strip() + '\n\n### definition: ' + evidence + "\n### Question: " + question + "\n\nReturn your answer in JSON format as {\"sql\": \"your sql\"}."
        response = self.llm.complete(SQL_GENERATION_INSTRUCTION + "\n" + prompt).text
        return self.parse_json_response(response)['sql'].replace('\n', ' ')

    def binary_selection(self, table_info, sql1, re1, sql2, re2):
        candidate_sql = f"### sql1: {sql1} \n### result1: {re1} \n### sql2: {sql2} \n### result2: {re2}"
        prompt = table_info + "\n\n### Select the best SQL query to answer the question:\n" + candidate_sql + "\n\nReturn your answer in JSON format as {\"sql\": \"your selected or new sql\"}."
        response = self.llm.complete(BINARY_PROMPT.format(table_info=table_info, candidate_sql=candidate_sql)).text
        return self.parse_json_response(response)['sql'].replace('\n', ' ')

    def self_correction(self, table_info, pre_sql, db_id, db_type, row=None):
        prompt = SELF_CORRECTION_PROMPT + '\n' + table_info + '\n\nReturn your answer in JSON format as {"sql": "your sql"}.'
        num = 0
        while num < 5:
            try:
                row_count, column_count, result = self.execute_sql(pre_sql, db_id, db_type, row)
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

    def act(
            self,
            item,
            schema=None,
            schema_links=None,
            **kwargs
    ):
        """Generate predicted SQL for the given dataset item using RSLSQL methodology."""
        logger.info(f"RSLSQLGenerator processing item {item}")
        row = self.dataset[item]
        question = row['question']
        db_id = row['db_id']
        db_type = row.get('db_type', 'sqlite')
        evidence = row.get('evidence', '') or (load_dataset(row.get('external', '')) if self.use_external else '')
        example = load_dataset(row.get('reasoning_examples', '')) if self.use_few_shot else ''
        evidence = evidence or ''
        example = example or ''

        # Load and process schema - 参考DINSQLGenerate的实现
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

        # Normalize schema type
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            schema_df = schema
        else:
            raise ValueError("Invalid schema format")

        logger.debug("Database schema processed")

        # Step 1: Preliminary SQL - 使用schema DataFrame而不是直接访问数据库
        try:
            simple_ddl, _ = self.get_simple_ddl_from_schema(schema_df)
            ddl_data = self.get_ddl_data_from_schema(schema_df, self.get_all_tables_from_schema(schema_df),
                                                     {t: self.get_table_columns_from_schema(schema_df, t) for t in
                                                      self.get_all_tables_from_schema(schema_df)}, db_id, db_type)
            foreign_key = self.get_foreign_key_from_schema(schema_df)
            table_info = f'### Sqlite SQL tables, with their properties:\n{simple_ddl}\n### Here are some data information about database references.\n{ddl_data}\n### Foreign key information of Sqlite SQL tables, used for table joins:\n{foreign_key}'
            table_column = self.table_column_selection(table_info, question, evidence)

            pre_sql = self.preliminary_sql_gen(table_info, table_column, example, question, evidence)
        except Exception as e:
            logger.error(f"Error in preliminary SQL generation: {e}")
            # Fallback to a simple SQL
            pre_sql = "SELECT 1"
            table_column = {"tables": [], "columns": []}

        # Bidirectional Schema Linking
        try:
            db_schema = self.get_all_schema_from_df(schema_df)
            sl_hint = self.extract_from_text(evidence, db_schema)
            sl_sql = self.extract_from_text(pre_sql, db_schema)
            sl_llm = table_column
            schema_links = self.merge_schema_links(sl_sql, sl_llm, sl_hint)
            schema_links = self.filter_schema_links(schema_links, db_schema)
        except Exception as e:
            logger.error(f"Error in schema linking: {e}")
            schema_links = {"tables": [], "columns": []}

        # Step 2: Information Augmentation
        try:
            simple_ddl, table_list = self.get_simple_ddl_from_schema(schema_df, schema_links['tables'],
                                                                     schema_links['columns'])
            ddl_data = self.get_ddl_data_from_schema(schema_df, schema_links['tables'], table_list, db_id, db_type)
            foreign_key = self.get_foreign_key_from_schema(schema_df, schema_links['tables'])
            explanation = self.get_explanation_from_schema(schema_df, schema_links['tables'], schema_links['columns'])
            table_info_aug = f'### Sqlite SQL tables, with their properties:\n{simple_ddl}\n### Here are some data information about database references.\n{ddl_data}\n### Foreign key information of Sqlite SQL tables, used for table joins:\n{foreign_key}\n### The meaning of every column:\n#\n{explanation.strip()}\n#\n'

            table_aug = self.table_column_selection(table_info_aug, question, evidence)  # Similar to table_augmentation
            word_aug = self.key_word_augmentation(table_info_aug, question, evidence)
            cond_aug = self.condition_augmentation(question)
            sql2 = self.sql_generation_aug(table_info_aug, table_aug, word_aug, cond_aug, example, question, evidence)
        except Exception as e:
            logger.error(f"Error in information augmentation: {e}")
            # Fallback values
            table_aug = {"tables": [], "columns": []}
            word_aug = {"sql_keywords": []}
            cond_aug = {"conditions": []}
            sql2 = "SELECT 1"

        # Step 3: Binary Selection
        try:
            r1, c1, re1 = self.execute_sql(pre_sql, db_id, db_type, row)
            r2, c2, re2 = self.execute_sql(sql2, db_id, db_type, row)
            selected_sql = self.binary_selection(table_info_aug, pre_sql, re1, sql2, re2)
        except Exception as e:
            logger.error(f"Error in binary selection: {e}")
            selected_sql = pre_sql

        # Step 4: Self Correction
        try:
            pred_sql = self.self_correction(
                table_info_aug + f'\n### sql_keywords: {word_aug["sql_keywords"]}\n### conditions: {cond_aug["conditions"]}',
                selected_sql,
                db_id,
                db_type,
                row
            )
        except Exception as e:
            logger.error(f"Error in self correction: {e}")
            pred_sql = selected_sql

        if self.is_save:
            instance_id = row.get("instance_id", item)
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            save_path = save_path / f"{self.NAME}_{instance_id}.sql"

            save_dataset(pred_sql, new_data_source=save_path)
            self.dataset.setitem(item, "pred_sql", str(save_path))

        return pred_sql
