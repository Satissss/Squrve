from typing import Union, Optional, List, Dict
from os import PathLike
import json
import re
from loguru import logger
import pandas as pd
from pathlib import Path
from core.actor.parser.BaseParse import BaseParser, parallel_slice_parse
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from llama_index.core.llms.llm import LLM

@BaseParser.register_actor
class RSLSQLBiDirParser(BaseParser):
    NAME = "RSLSQLBiDirParser"

    SKILL = """# RSLSQLBiDirParser

RSL-SQL bidirectional schema linking: forward (LLM selects tables + top-15 columns) and reverse (LLM generates preliminary SQL; extract links from SQL). Merges three sources—LLM selection, evidence text extraction, SQL extraction—then filters by db_schema. Uses sample data and evidence in prompts. Advantage: SQL-as-reverse-signal enriches links; drawback: requires preliminary SQL generation (extra LLM call).

## Inputs
- `schema`: DB schema. If absent, loaded from dataset.

## Output
`schema_links` (dict: `{tables, columns}`)

## Steps
1. Table/column selection: LLM identifies tables and top-15 columns per table.
2. Preliminary SQL generation: LLM generates SQL from selection + examples.
3. Extract links from evidence text and from generated SQL.
4. Merge and filter against db_schema.
5. Return `schema_links`.
"""

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
6. Verification and optimization: ensure the logical correctness of the SQL statement.

### Note:
- Ensure that the SQL statement accurately reflects the query requirements and conditions in the user questions.
- Reasonably construct query logic based on database structure and sample data.
- When generating SQL statements, consider all the information provided to ensure the correctness and efficiency of the statements.
    
    ### The most important thing is to remember:
    - definition: Information for prompts, this message is very important.
    - In the generated SQL statement, table names and field names need to be enclosed in backticks, such as `table_name`, `column_name`.
    - In the generated SQL statement, table names and field names must be correct to ensure the correctness and efficiency of the statement.
'''

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Union[LLM, List[LLM]] = None,
            output_format: str = "list",  # output in `list` or `str`
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/schema_links",
            use_external: bool = False,
            **kwargs
    ):
        super().__init__(dataset, llm, output_format, is_save, save_dir, use_external, **kwargs)

    def parse_json_response(self, response):
        """
        Robust JSON parsing with multiple fallback strategies
        """
        if not response or not isinstance(response, str):
            logger.warning(f"Invalid response type: {type(response)}")
            return {"sql": "SELECT 1", "tables": [], "columns": [], "sql_keywords": [], "conditions": []}

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
        return {"sql": "SELECT 1", "tables": [], "columns": [], "sql_keywords": [], "conditions": []}

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
                # Try to get sample data from different possible column names
                sample_rows = row.get('sample_rows', []) or row.get('sample_data', []) or []
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

        # Assuming no explicit FK info, return empty
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
            # Try different possible column names for descriptions
            col_desc = row.get('column_descriptions', '') or row.get('description', '') or row.get('col_desc', '')

            if table_name in tables and f"{table_name}.`{col_name}`".lower() in columns_lower:
                if col_desc:
                    explanation += f"# {table_name}.{col_name}: {col_desc}\n"

        return explanation

    def extract_from_text(self, text, db_schema):
        """Extract schema links from text (evidence or SQL) using flexible column name matching.
        Reproduces the improved behavior in RSLSQLGenerator.
        """
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
                table_part, column_part = parts[0].lower(), '.'.join(parts[1:]).lower()
                if table_part == 'sqlite_sequence':
                    continue
                column_clean = column_part.strip('`').strip()
                # Flexible matching: full column or any underscore-separated token present
                if column_clean in text_lower or any(word in text_lower for word in column_clean.split('_')):
                    pred.append(item)
            except (ValueError, AttributeError):
                continue

        pred = list(set(pred))
        tables = list(set([p.split('.')[0] for p in pred if '.' in p]))
        columns = []
        for p in pred:
            if '.' in p:
                t, c = p.split('.', 1)
                c_fmt = c
                if not c_fmt.startswith('`'):
                    c_fmt = '`' + c_fmt
                if not c_fmt.endswith('`'):
                    c_fmt = c_fmt + '`'
                columns.append(f"{t}.{c_fmt}")
        return {"tables": tables, "columns": columns}

    def merge_schema_links(self, sl_sql, sl_llm, sl_hint):
        tables = list(set(sl_llm['tables'] + sl_sql['tables'] + sl_hint['tables']))
        columns = list(set(sl_llm['columns'] + sl_sql['columns'] + sl_hint['columns']))
        return {"tables": [t.lower() for t in tables], "columns": [c.lower() for c in columns]}

    def filter_schema_links(self, schema_links, db_schema):
        pred = []
        db_schema_lower = [s.lower() for s in db_schema]
        for col in schema_links['columns']:
            col_lower = col.replace('`', '').lower()
            for idx, sch_lower in enumerate(db_schema_lower):
                if sch_lower == col_lower:
                    pred.append(db_schema[idx])
                    break
        pred = list(set(pred))
        tables = list(set([p.split('.')[0] for p in pred if '.' in p]))
        columns = []
        for p in pred:
            t, c = p.split('.', 1)
            columns.append(f"{t}.`{c}`")
        return {"tables": tables, "columns": columns}

    def table_column_selection(self, table_info, question, evidence):
        prompt = table_info.strip() + '\n\n### definition: ' + evidence + "\n### Question: " + question + "\n\nReturn your answer in JSON format as specified."
        response = self.llm.complete(self.TABLE_AUG_INSTRUCTION + "\n" + prompt).text
        return self.parse_json_response(response)

    def preliminary_sql_gen(self, table_info, table_column, example, question, evidence):
        table_info += f'### tables: {table_column["tables"]}\n'
        table_info += f'### columns: {table_column["columns"]}\n'
        prompt = example.strip() + "\n\n### Answer the question by sqlite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.\n" + table_info.strip() + '\n\n### definition: ' + evidence + "\n### Question: " + question + "\n\nReturn your answer in JSON format as {\"sql\": \"your sql\"}."
        response = self.llm.complete(self.SQL_GENERATION_INSTRUCTION + "\n" + prompt).text
        return self.parse_json_response(response)['sql'].replace('\n', ' ')

    @parallel_slice_parse
    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List, pd.DataFrame] = None,
            data_logger=None,
            update_dataset=True,
            **kwargs
    ):
        try:
            if data_logger:
                data_logger.info(f"{self.NAME}.act start | item={item}")

            row = self.dataset[item]
            question = row['question']
            evidence = row.get('evidence', '') or (load_dataset(row.get('external', '')) if self.use_external else '')
            example = load_dataset(row.get('reasoning_examples', ''))
            evidence = evidence or ''
            example = example or ''

            # Use base class method to process schema
            schema_df = self.process_schema(item, schema)

            # Preliminary SQL generation for reverse linking
            try:
                simple_ddl, _ = self.get_simple_ddl_from_schema(schema_df)
                ddl_data = self.get_ddl_data_from_schema(schema_df, self.get_all_tables_from_schema(schema_df),
                                                         {t: self.get_table_columns_from_schema(schema_df, t) for t in
                                                          self.get_all_tables_from_schema(schema_df)})
                foreign_key = self.get_foreign_key_from_schema(schema_df)
                table_info = '### Sqlite SQL tables, with their properties:\n' + simple_ddl + '\n### Here are some data information about database references.\n' + ddl_data + '\n### Foreign key information of Sqlite SQL tables, used for table joins:\n' + foreign_key
                table_column = self.table_column_selection(table_info, question, evidence)
                self.log_schema_links(data_logger, table_column.get('columns', []), stage="selected.columns")
                self.log_schema_links(data_logger, table_column.get('tables', []), stage="selected.tables")

                pre_sql = self.preliminary_sql_gen(table_info, table_column, example, question, evidence)
                if data_logger:
                    data_logger.info(f"{self.NAME}.preliminary_sql | {pre_sql}")

            except Exception as e:
                if data_logger:
                    data_logger.error(f"{self.NAME}.preliminary_sql_gen error | {e}")
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
                self.log_schema_links(data_logger, schema_links.get('columns', []), stage="final.columns")
                self.log_schema_links(data_logger, schema_links.get('tables', []), stage="final.tables")
            except Exception as e:
                if data_logger:
                    data_logger.error(f"{self.NAME}.schema_linking error | {e}")
                schema_links = {"tables": [], "columns": []}

            # Use base class method to save output
            if update_dataset:
                self.save_output(schema_links, item)

            if data_logger:
                data_logger.info(f"{self.NAME}.act end | item={item}")

            return schema_links

        except Exception as e:
            logger.error(f"Error in RSLSQLBiDirParser.act(): {e}")
            # Return empty schema links as fallback
            return {"tables": [], "columns": []}

    def merge_results(self, results: List):
        if not results:
            logger.info("Input results empty!")

        merge_result = {}
        for row in results:
            if not isinstance(row, dict):
                raise TypeError(f"Each row must be a dict, but got {type(row)}: {row}")
            if "tables" not in merge_result:
                merge_result["tables"] = row.get("tables", [])
            else:
                merge_result["tables"] += row.get("tables", [])

            if "columns" not in merge_result:
                merge_result["columns"] = row.get("columns", [])
            else:
                merge_result["columns"] += row.get("columns", [])

        return merge_result
