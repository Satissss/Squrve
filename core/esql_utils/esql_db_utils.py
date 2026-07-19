import sqlite3
import random
import logging
import re
import time
import nltk
from nltk.tokenize import word_tokenize
import difflib
from rank_bm25 import BM25Okapi
import sqlglot
from sqlglot import parse, parse_one, expressions
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.qualify_columns import qualify_columns
from sqlglot.expressions import Select
from func_timeout import func_timeout, FunctionTimedOut
from typing import Any, Union, List, Dict, Optional

try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, PermissionError):
    try:
        nltk.download('punkt', quiet=True)
    except (PermissionError, Exception):
        pass


def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            if fetch == "all":
                return cursor.fetchall()
            elif fetch == "one":
                return cursor.fetchone()
            elif fetch == "random":
                samples = cursor.fetchmany(10)
                return random.choice(samples) if samples else []
            elif isinstance(fetch, int):
                return cursor.fetchmany(fetch)
            else:
                raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', or an integer.")
    except Exception as e:
        logging.error(f"Error in execute_sql: {e}\n db_path: {db_path}\n SQL: {sql}")
        raise e


def get_db_tables(db_path):
    try:
        tables = execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';")
        db_tables = [table_name_tuple[0] for table_name_tuple in tables if table_name_tuple[0] != "sqlite_sequence"]
        return db_tables
    except Exception as e:
        logging.error(f"Error in get_db_tables: {e}")
        raise e


def get_db_columns_of_table(db_path: str, table_name: str) -> List[str]:
    try:
        table_info_rows = execute_sql(db_path, f"PRAGMA table_info(`{table_name}`);")
        columns_of_table = [row[1] for row in table_info_rows]
        return columns_of_table
    except Exception as e:
        logging.error(f"Error in get_table_all_columns: {e}\nTable: {table_name}")
        raise e


def isTableInDB(db_path: str, table_name: str) -> bool:
    db_tables = get_db_tables(db_path)
    if table_name in db_tables:
        return True
    else:
        return False


def isColumnInTable(db_path: str, table_name: str, column_name: str) -> bool:
    columns_of_table = get_db_columns_of_table(db_path, table_name)
    if column_name in columns_of_table:
        return True
    else:
        return False


def get_original_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    db_schema_dict = {}
    for table_name_tuple in tables:
        table_name = table_name_tuple[0]
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_statement = cursor.fetchone()[0]
        db_schema_dict[table_name] = create_statement
    cursor.close()
    conn.close()
    db_schema = " "
    for table, create_table_statement in db_schema_dict.items():
        db_schema = db_schema + create_table_statement + "\n"
    return db_schema


def clean_db_schema(db_schema: str) -> str:
    lines = db_schema.split('\n')
    cleaned_lines = []
    current_statement = []
    for index in range(len(lines)):
        line = lines[index]
        line = line.strip()
        if not line:
            continue
        if "CREATE TABLE" in line:
            line = line + " ("
            cleaned_lines.append(line)
            continue
        if line[0] == '(':
            continue
        if line[0] == ')':
            cleaned_lines.append(line)
            continue
        if "primary key" in line.lower():
            cleaned_lines[-1] = cleaned_lines[-1] + 'primary key,'
            continue
        line = line.replace('AUTOINCREMENT', '')
        line = line.replace('DEFAULT 0', '')
        line = line.replace('NOT NULL', '')
        line = line.replace('NULL', '')
        line = line.replace('UNIQUE', '')
        line = line.replace('ON UPDATE', '')
        line = line.replace('ON DELETE', '')
        line = line.replace('CASCADE', '')
        line = line.replace('autoincrement', '')
        line = line.replace('default 0', '')
        line = line.replace('not null', '')
        line = line.replace('null', '')
        line = line.replace('unique', '')
        line = line.replace('on update', '')
        line = line.replace('on delete', '')
        line = line.replace('cascade', '')
        line = re.sub(r'\s*,', ',', line)
        line = re.sub(r'`([^`]+)`\s+(\w+)', r'`\1` \2', line)
        line = re.sub(r'(\w+)\s+(\w+)', r'\1 \2', line)
        cleaned_lines.append(line)
    cleaned_db_schema = '\n'.join(cleaned_lines)
    return cleaned_db_schema


def get_schema(db_path: str) -> str:
    original_db_schema = get_original_schema(db_path)
    db_schema = clean_db_schema(original_db_schema)
    return db_schema


def get_schema_dict(db_path: str) -> Dict:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]
    db_schema_dict = {}
    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        table_info = cursor.fetchall()
        db_schema_dict[table_name] = {col_item[1]: col_item[2] for col_item in table_info}
    cursor.close()
    conn.close()
    return db_schema_dict


def get_schema_tables_and_columns_dict(db_path: str) -> Dict:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]
    db_schema_dict = {}
    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        table_info = cursor.fetchall()
        db_schema_dict[table_name] = [col_item[1] for col_item in table_info]
    cursor.close()
    conn.close()
    return db_schema_dict


def clean_sql(sql: str) -> str:
    clean_sql = sql.replace('\n', ' ').replace('"', "`").replace('\"', "`")
    return clean_sql


def compare_sqls_outcomes(db_path: str, predicted_sql: str, ground_truth_sql: str) -> int:
    try:
        predicted_res = execute_sql(db_path, predicted_sql)
        ground_truth_res = execute_sql(db_path, ground_truth_sql)
        return int(set(predicted_res) == set(ground_truth_res))
    except Exception as e:
        logging.critical(f"Error comparing SQL outcomes: {e}")
        raise e


def compare_sqls(db_path: str, predicted_sql: str, ground_truth_sql: str, meta_time_out: int = 30) -> Dict[str, Union[int, str]]:
    predicted_sql = clean_sql(predicted_sql)
    try:
        res = func_timeout(meta_time_out, compare_sqls_outcomes, args=(db_path, predicted_sql, ground_truth_sql))
        error = "incorrect answer" if res == 0 else "--"
    except FunctionTimedOut:
        logging.warning("Comparison timed out.")
        error = "timeout"
        res = 0
    except Exception as e:
        logging.error(f"Error in compare_sqls: {e}")
        error = str(e)
        res = 0
    return {'exec_res': res, 'exec_err': error}


def extract_sql_tables(db_path: str, sql: str) -> List[str]:
    db_tables = get_db_tables(db_path)
    try:
        parsed_tables = list(parse_one(sql, read='sqlite').find_all(expressions.Table))
        tables_in_sql = [str(table.name) for table in parsed_tables if str(table.name) in [db_table.lower() for db_table in db_tables]]
        tables_in_sql = list(set(tables_in_sql))
        return tables_in_sql
    except Exception as e:
        logging.critical(f"Error in extract_sql_tables: {e}\n")
        raise e


def extract_sql_tables_with_aliases(db_path: str, sql: str) -> List[str]:
    db_tables = get_db_tables(db_path)
    try:
        parsed_tables = list(parse_one(sql, read='sqlite').find_all(expressions.Table))
        tables_w_aliases = [{"table_name": str(table.name), "table_alias": str(table.alias)} for table in parsed_tables if str(table.name) in [db_table.lower() for db_table in db_tables]]
        tables_w_aliases = [table_alias_dict for table_alias_dict in tables_w_aliases if table_alias_dict['table_alias'] != '']
        return tables_w_aliases
    except Exception as e:
        logging.warning(f"Error in extract_sql_tables_with_aliases: \n\tError{e} \n\t{sql}")
        raise


def replace_alias_with_table_names_in_sql(db_path: str, sql: str) -> str:
    try:
        tables_w_aliases = extract_sql_tables_with_aliases(db_path, sql)
        for table_dict in tables_w_aliases:
            table_name = table_dict['table_name']
            table_alias = table_dict['table_alias']
            sql = sql.replace(f"{table_alias}.", f"{table_name}.")
        return sql
    except Exception as e:
        logging.warning(f"Error in replace_alias_with_table_names_in_sql: {e}")
        return sql


def extract_sql_columns(db_path: str, sql: str) -> Dict[str, List[str]]:
    try:
        sql = replace_alias_with_table_names_in_sql(db_path, sql)
        db_tables = get_db_tables(db_path)
        parsed_columns = list(parse_one(sql, read='sqlite').find_all(expressions.Column))
        sql_schema_dict = {}
        for column in parsed_columns:
            if column.table:
                table_name = str(column.table)
                if table_name in [db_table.lower() for db_table in db_tables]:
                    if table_name not in sql_schema_dict:
                        sql_schema_dict[table_name] = []
                    sql_schema_dict[table_name].append(str(column.name))
        for table_name in sql_schema_dict:
            sql_schema_dict[table_name] = list(set(sql_schema_dict[table_name]))
        return sql_schema_dict
    except Exception as e:
        logging.warning(f"Error in extract_sql_columns: {e}")
        return {}


def generate_schema_from_schema_dict(db_path: str, schema_dict: Dict[str, List[str]]) -> str:
    try:
        schema = get_schema(db_path)
        lines = schema.split('\n')
        filtered_lines = []
        for line in lines:
            if "CREATE TABLE" in line:
                for table_name in schema_dict:
                    if table_name.lower() in line.lower():
                        filtered_lines.append(line)
                        break
            else:
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)
    except Exception as e:
        logging.warning(f"Error in generate_schema_from_schema_dict: {e}")
        return schema


def collect_possible_conditions(db_path: str, sql: str) -> List[Dict[str, Union[str, Dict]]]:
    OP_MAP = {
        'EQ': '=', 'GT': '>', 'LT': '<', 'GTE': '>=', 'LTE': '<=', 'NEQ': '!=',
        'Is': 'IS', 'NullSafeEQ': 'IS NOT DISTINCT FROM', 'NullSafeNEQ': 'IS DISTINCT FROM',
    }
    try:
        sql = replace_alias_with_table_names_in_sql(db_path, sql)
        parsed = parse_one(sql, read='sqlite')
        where_clause = parsed.find(expressions.Where)
        if not where_clause:
            return []
        conditions = []
        for condition in where_clause.walk():
            cls_name = condition.__class__.__name__
            op_str = OP_MAP.get(cls_name)
            if op_str and isinstance(condition.this, expressions.Column) and isinstance(condition.expression, expressions.Literal):
                table_name = str(condition.this.table) if condition.this.table else ""
                column_name = str(condition.this.name)
                value = str(condition.expression)
                if table_name and column_name:
                    similar_values = _find_similar_values(db_path, table_name, column_name, value)
                    conditions.append({
                        "table": table_name,
                        "column": column_name,
                        "op": op_str,
                        "value": value,
                        "similar_values": similar_values
                    })
        return conditions
    except Exception as e:
        logging.warning(f"Error in collect_possible_conditions: {e}")
        return []


def _find_similar_values(db_path: str, table_name: str, column_name: str, value: str) -> Dict[str, Dict[str, List[str]]]:
    try:
        db_tables = get_db_tables(db_path)
        similar_values = {}
        for table in db_tables:
            if table.lower() == table_name.lower():
                continue
            columns = get_db_columns_of_table(db_path, table)
            for col in columns:
                try:
                    distinct_values = execute_sql(db_path, f"SELECT DISTINCT `{col}` FROM `{table}` LIMIT 10", fetch="all")
                    similar_col_values = []
                    for row in distinct_values:
                        row_val = str(row[0]) if row[0] is not None else ""
                        if row_val and value.lower() in row_val.lower():
                            similar_col_values.append(row_val)
                    if similar_col_values:
                        if table not in similar_values:
                            similar_values[table] = {}
                        similar_values[table][col] = similar_col_values
                except Exception:
                    continue
        return similar_values
    except Exception as e:
        logging.warning(f"Error in _find_similar_values: {e}")
        return {}