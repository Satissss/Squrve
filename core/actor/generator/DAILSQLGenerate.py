import collections
import re
import string
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from os import PathLike
from typing import Union, Dict, List, Optional
from pathlib import Path
import pandas as pd
import abc

from loguru import logger
from sql_metadata import Parser

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)
from core.db_connect import get_sql_exec_result

nltk.download('stopwords', quiet=True)


# Utility Functions
def jaccard_similarity(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0


def sql_normalization(sql):
    sql = sql.strip()

    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])
        return s

    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()
            if char == "'":
                in_quotation = not in_quotation
        return out_s

    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    def double2single(s):
        return s.replace('"', "'")

    def add_asc(s):
        pattern = re.compile(
            r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")
        return s

    def sql_split(s):
        while "  " in s:
            s = s.replace("  ", " ")
        s = s.strip()
        i = 0
        toks = []
        while i < len(s):
            tok = ""
            if s[i] == "'":
                tok += s[i]
                i += 1
                while i < len(s) and s[i] != "'":
                    tok += s[i]
                    i += 1
                if i < len(s):
                    tok += s[i]
                    i += 1
            else:
                while i < len(s) and s[i] != " ":
                    tok += s[i]
                    i += 1
                while i < len(s) and s[i] == " ":
                    i += 1
            toks.append(tok)
        return toks

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if f"t{i}" in tables_aliases:
                new_tables_aliases[f"t{i}"] = tables_aliases[f"t{i}"]
        table_names = [tok.split('.')[0] for tok in sql_split(s) if '.' in tok]
        for table_name in table_names:
            if table_name in tables_aliases:
                new_tables_aliases[table_name] = tables_aliases[table_name]
        tables_aliases = new_tables_aliases

        new_s = []
        pre_tok = ""
        for tok in sql_split(s):
            if tok in tables_aliases:
                if pre_tok == 'as':
                    new_s = new_s[:-1]
                elif pre_tok != tables_aliases[tok]:
                    new_s.append(tables_aliases[tok])
            elif '.' in tok:
                split_toks = tok.split('.')
                for i in range(len(split_toks)):
                    if len(split_toks[i]) > 2 and split_toks[i][0] == "'" and split_toks[i][-1] == "'":
                        split_toks[i] = split_toks[i].replace("'", "").lower()
                    if split_toks[i] in tables_aliases:
                        split_toks[i] = tables_aliases[split_toks[i]]
                new_s.append('.'.join(split_toks))
            else:
                new_s.append(tok)
            pre_tok = tok

        s = new_s
        new_s = [s[i] for i in range(len(s)) if s[i] != "as" and (i == 0 or s[i - 1] != "as")]
        new_s = ' '.join(new_s)
        return new_s

    processing_func = lambda x: remove_table_alias(add_asc(lower(white_space_fix(double2single(remove_semicolon(x))))))
    return processing_func(sql.strip())


def sql2skeleton(sql, db_schema):
    sql = sql_normalization(sql)

    table_names_original, table_dot_column_names_original, column_names_original = [], [], []
    column_names_original.append("*")
    for table_id, table_name_original in enumerate(db_schema["table_names_original"]):
        table_names_original.append(table_name_original.lower())
        table_dot_column_names_original.append(table_name_original + ".*")
        for column_id_and_name in db_schema["column_names_original"]:
            column_id = column_id_and_name[0]
            column_name_original = column_id_and_name[1]
            table_dot_column_names_original.append(table_name_original.lower() + "." + column_name_original.lower())
            column_names_original.append(column_name_original.lower())

    parsed_sql = Parser(sql)
    new_sql_tokens = []
    for token in parsed_sql.tokens:
        if token.value in table_names_original:
            new_sql_tokens.append("_")
        elif token.value in column_names_original or token.value in table_dot_column_names_original:
            new_sql_tokens.append("_")
        elif token.value.startswith("'") and token.value.endswith("'"):
            new_sql_tokens.append("_")
        elif token.value.isdigit():
            new_sql_tokens.append("_")
        elif isNegativeInt(token.value):
            new_sql_tokens.append("_")
        elif isFloat(token.value):
            new_sql_tokens.append("_")
        else:
            new_sql_tokens.append(token.value.strip())

    sql_skeleton = " ".join(new_sql_tokens)
    sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace(" on _ = _", "")
    pattern3 = re.compile(r'_ (?:join _ ?)+')
    sql_skeleton = re.sub(pattern3, "_ ", sql_skeleton)

    while ("_ , _" in sql_skeleton):
        sql_skeleton = sql_skeleton.replace("_ , _", "_")

    ops = ["=", "!=", ">", ">=", "<", "<="]
    for op in ops:
        if "_ {} _".format(op) in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("_ {} _".format(op), "_")
    while ("where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton):
        if "where _ and _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
        if "where _ or _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ or _", "where _")

    while "  " in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("  ", " ")

    split_skeleton = sql_skeleton.split(" ")
    for i in range(2, len(split_skeleton)):
        if split_skeleton[i - 2] == "order" and split_skeleton[i - 1] == "by" and split_skeleton[i] != "_":
            split_skeleton[i] = "_"
    sql_skeleton = " ".join(split_skeleton)

    return sql_skeleton


def mask_question_with_schema_linking(data_jsons, mask_tag='<mask>', value_tag=''):
    mask_questions = []
    for data_json in data_jsons:
        sc_link = data_json["sc_link"]
        cv_link = data_json["cv_link"]
        q_col_match = sc_link["q_col_match"]
        q_tab_match = sc_link["q_tab_match"]
        num_date_match = cv_link["num_date_match"]
        cell_match = cv_link["cell_match"]
        question_for_copying = data_json["question_for_copying"]
        q_col_match, q_tab_match, cell_match = match_shift(q_col_match, q_tab_match, cell_match)

        def mask(question_toks, mask_ids, tag):
            new_question_toks = []
            for id, tok in enumerate(question_toks):
                if id in mask_ids:
                    new_question_toks.append(tag)
                else:
                    new_question_toks.append(tok)
            return new_question_toks

        num_date_match_ids = [int(match.split(',')[0]) for match in num_date_match]
        cell_match_ids = [int(match.split(',')[0]) for match in cell_match]
        value_match_q_ids = num_date_match_ids + cell_match_ids
        question_toks = mask(question_for_copying, value_match_q_ids, value_tag)

        q_col_match_ids = [int(match.split(',')[0]) for match in q_col_match]
        q_tab_match_ids = [int(match.split(',')[0]) for match in q_tab_match]
        schema_match_q_ids = q_col_match_ids + q_tab_match_ids
        question_toks = mask(question_toks, schema_match_q_ids, mask_tag)
        mask_questions.append(" ".join(question_toks))

    return mask_questions


def get_sql_for_database(path_db, db_type='sqlite', credential=None, db_id=None):
    """
    Get CREATE TABLE SQL statements for database using get_sql_exec_result.
    Supports multiple database types through unified interface.
    """
    try:
        # Get table names first
        if db_type == 'sqlite':
            table_names_query = "SELECT name FROM sqlite_master WHERE type='table'"
        elif db_type == 'big_query':
            # BigQuery uses INFORMATION_SCHEMA
            table_names_query = "SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_type='BASE TABLE'"
        elif db_type == 'snowflake':
            # Snowflake uses INFORMATION_SCHEMA
            table_names_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
        else:
            # Default to standard SQL
            table_names_query = "SELECT table_name FROM information_schema.tables WHERE table_type='BASE TABLE'"

        # Execute query to get table names
        result, error = get_sql_exec_result(
            db_type=db_type,
            sql_query=table_names_query,
            db_path=str(path_db) if path_db else None,
            credential_path=credential,
            db_id=db_id
        )

        if error or result is None:
            logger.warning(f"Failed to get table names: {error}")
            return []

        # Extract table names from result
        if hasattr(result, 'values'):
            table_names = [row[0] for row in result.values]
        elif isinstance(result, list):
            table_names = [row[0] if isinstance(row, (list, tuple)) else str(row) for row in result]
        else:
            table_names = []

        sqls = []
        for table_name in table_names:
            if db_type == 'sqlite':
                # For SQLite, get the CREATE statement
                create_sql_query = f"SELECT sql FROM sqlite_master WHERE tbl_name='{table_name}'"
                result, error = get_sql_exec_result(
                    db_type=db_type,
                    sql_query=create_sql_query,
                    db_path=str(path_db) if path_db else None,
                    credential_path=credential,
                    db_id=db_id
                )
                if not error and result is not None:
                    if hasattr(result, 'values') and len(result.values) > 0:
                        sql_statement = result.values[0][0]
                        if sql_statement:
                            sqls.append(sql_statement)
            else:
                # For other databases, construct CREATE statement from INFORMATION_SCHEMA
                # This is a simplified version - in practice you might want more detailed schema info
                sqls.append(f"-- Table: {table_name} (Schema details not available for {db_type})")

        return sqls

    except Exception as e:
        logger.error(f"Failed to get SQL for database: {e}")
        return []


# Enums
class REPR_TYPE:
    CODE_REPRESENTATION = "SQL"
    TEXT_REPRESENTATION = "TEXT"
    OPENAI_DEMOSTRATION = "NUMBERSIGN"
    BASIC = "BASELINE"
    ALPACA_SFT = "INSTRUCTION"
    OPENAI_DEMOSTRATION_WFK = "NUMBERSIGNWFK"
    BASIC_WOFK = "BASELINEWOFK"
    TEXT_REPRESENTATION_WFK = "TEXTWFK"
    ALPACA_SFT_WFK = "INSTRUCTIONWFK"
    OPENAI_DEMOSTRATION_WORULE = "NUMBERSIGNWORULE"
    CODE_REPRESENTATION_WRULE = "SQLWRULE"
    ALPACA_SFT_WRULE = "INSTRUCTIONWRULE"
    TEXT_REPRESENTATION_WRULE = "TEXTWRULE"
    CODE_REPRESENTATION_COT = "SQLCOT"
    TEXT_REPRESENTATION_COT = "TEXTCOT"
    OPENAI_DEMOSTRATION_COT = "NUMBERSIGNCOT"
    ALPACA_SFT_COT = "INSTRUCTIONCOT"
    CBR = "CBR"


class EXAMPLE_TYPE:
    ONLY_SQL = "ONLYSQL"
    QA = "QA"
    COMPLETE = "COMPLETE"
    QAWRULE = "QAWRULE"
    OPENAI_DEMOSTRATION_QA = "NUMBERSIGNQA"
    BASIC_QA = "BASELINEQA"


class SELECTOR_TYPE:
    COS_SIMILAR = "COSSIMILAR"
    RANDOM = "RANDOM"
    EUC_DISTANCE = "EUCDISTANCE"
    EUC_DISTANCE_THRESHOLD = "EUCDISTANCETHRESHOLD"
    EUC_DISTANCE_SKELETON_SIMILARITY_THRESHOLD = "EUCDISSKLSIMTHR"
    EUC_DISTANCE_QUESTION_MASK = "EUCDISQUESTIONMASK"
    EUC_DISTANCE_PRE_SKELETON_SIMILARITY_THRESHOLD = "EUCDISPRESKLSIMTHR"
    EUC_DISTANCE_PRE_SKELETON_SIMILARITY_PLUS = "EUCDISPRESKLSIMPLUS"
    EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD = "EUCDISMASKPRESKLSIMTHR"
    EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD_SHIFT = "EUCDISMASKPRESKLSIMTHRSHIFT"


# Linking Functions
STOPWORDS = set(stopwords.words('english'))
PUNKS = set(string.punctuation)

CELL_EXACT_MATCH_FLAG = "EXACTMATCH"
CELL_PARTIAL_MATCH_FLAG = "PARTIALMATCH"
COL_PARTIAL_MATCH_FLAG = "CPM"
COL_EXACT_MATCH_FLAG = "CEM"
TAB_PARTIAL_MATCH_FLAG = "TPM"
TAB_EXACT_MATCH_FLAG = "TEM"


def compute_schema_linking(question, column, table):
    def _to_tokens(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [t for t in re.split(r"[\s_]+", value.strip()) if t]
        return [str(value)]

    def partial_match(x_list, y_list):
        x_str = " ".join(x_list).strip().lower()
        y_str = " ".join(_to_tokens(y_list)).strip().lower()
        if not x_str or x_str in STOPWORDS or x_str in PUNKS:
            return False
        return re.search(rf"\b{re.escape(x_str)}\b", y_str) is not None

    def exact_match(x_list, y_list):
        x_str = " ".join(x_list).strip().lower()
        y_str = " ".join(_to_tokens(y_list)).strip().lower()
        return x_str == y_str

    q_col_match = dict()
    q_tab_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        col_id2list[col_id] = _to_tokens(col_item)

    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = _to_tokens(tab_item)

    # 5-gram
    n = 5
    while n > 0:
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n]
            n_gram = " ".join(n_gram_list)
            if len(n_gram.strip()) == 0:
                continue
            # exact match case
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        q_col_match[f"{q_id},{col_id}"] = COL_EXACT_MATCH_FLAG
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        q_tab_match[f"{q_id},{tab_id}"] = TAB_EXACT_MATCH_FLAG

            # partial match case
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{col_id}" not in q_col_match:
                            q_col_match[f"{q_id},{col_id}"] = COL_PARTIAL_MATCH_FLAG
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{tab_id}" not in q_tab_match:
                            q_tab_match[f"{q_id},{tab_id}"] = TAB_PARTIAL_MATCH_FLAG
        n -= 1
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}


def compute_cell_value_linking(tokens, schema_dict, db_type, db_path, db_id, credential=None):
    """
    Compute cell value linking using the unified database connection approach.
    Supports multiple database types through get_sql_exec_result factory method.
    Follows the original DAIL-SQL approach but uses database-agnostic queries.
    """

    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_match(word, column, table, db_type, db_path, db_id, credential, exact=False, invalid_tables_cache=None, qualify_table_fn=None):
        """
        Use get_sql_exec_result for database queries across different database types.
        Supports SQLite, BigQuery, Snowflake, etc.
        Implements both exact and partial matching like the original DAIL-SQL.
        """
        # Escape single quotes in word to prevent SQL injection
        escaped_word = word.replace("'", "''")

        # Construct appropriate SQL for different database types
        if exact:
            # Exact match: word matches entire cell content (with optional spaces)
            if db_type == "big_query":
                like_conditions = [
                    f"`{column}` = '{escaped_word}'",
                    f"`{column}` = ' {escaped_word}'",
                    f"`{column}` = '{escaped_word} '",
                    f"`{column}` = ' {escaped_word} '"
                ]
                sql_query = f"SELECT `{column}` FROM `{table}` WHERE ({' OR '.join(like_conditions)}) LIMIT 5"
            elif db_type == "snowflake":
                # Use UPPER identifiers and qualify table name to avoid case-sensitive quoted object issues
                col_ident = str(column).upper()
                tbl_ident = str(table).upper()
                if qualify_table_fn:
                    tbl_ident = qualify_table_fn(tbl_ident)
                like_conditions = [
                    f"{col_ident} ILIKE '{escaped_word}'",
                    f"{col_ident} ILIKE ' {escaped_word}'",
                    f"{col_ident} ILIKE '{escaped_word} '",
                    f"{col_ident} ILIKE ' {escaped_word} '"
                ]
                sql_query = f"SELECT {col_ident} FROM {tbl_ident} WHERE ({' OR '.join(like_conditions)}) LIMIT 5"
            else:
                # SQLite and others
                like_conditions = [
                    f"{column} LIKE '{escaped_word}'",
                    f"{column} LIKE ' {escaped_word}'",
                    f"{column} LIKE '{escaped_word} '",
                    f"{column} LIKE ' {escaped_word} '"
                ]
                sql_query = f"SELECT {column} FROM {table} WHERE ({' OR '.join(like_conditions)}) LIMIT 5"
        else:
            # Partial match: word appears anywhere in cell content
            if db_type == "big_query":
                like_conditions = [
                    f"`{column}` LIKE '{escaped_word} %'",
                    f"`{column}` LIKE '% {escaped_word}'",
                    f"`{column}` LIKE '% {escaped_word} %'",
                    f"`{column}` LIKE '{escaped_word}'"
                ]
                sql_query = f"SELECT `{column}` FROM `{table}` WHERE ({' OR '.join(like_conditions)}) LIMIT 5"
            elif db_type == "snowflake":
                col_ident = str(column).upper()
                tbl_ident = str(table).upper()
                if qualify_table_fn:
                    tbl_ident = qualify_table_fn(tbl_ident)
                like_conditions = [
                    f"{col_ident} ILIKE '{escaped_word} %'",
                    f"{col_ident} ILIKE '% {escaped_word}'",
                    f"{col_ident} ILIKE '% {escaped_word} %'",
                    f"{col_ident} ILIKE '{escaped_word}'"
                ]
                sql_query = f"SELECT {col_ident} FROM {tbl_ident} WHERE ({' OR '.join(like_conditions)}) LIMIT 5"
            else:
                # SQLite and others
                like_conditions = [
                    f"{column} LIKE '{escaped_word} %'",
                    f"{column} LIKE '% {escaped_word}'",
                    f"{column} LIKE '% {escaped_word} %'",
                    f"{column} LIKE '{escaped_word}'"
                ]
                sql_query = f"SELECT {column} FROM {table} WHERE ({' OR '.join(like_conditions)}) LIMIT 5"

        try:
            # Use unified database connection approach - consistent with LinkAlignGenerator
            exec_args = {
                "db_type": db_type,
                "sql_query": sql_query,
                "db_path": str(db_path) if db_path else None,
                "credential_path": credential,
                "db_id": db_id
            }
            result, error = get_sql_exec_result(**exec_args)

            if error:
                # Cache invalid tables to avoid repeated probing on Snowflake
                if invalid_tables_cache is not None and db_type == "snowflake":
                    lowered = str(error).lower()
                    if ("does not exist" in lowered) or ("not authorized" in lowered) or ("sql compilation error" in lowered):
                        try:
                            tbl_for_cache = qualify_table_fn(str(table).upper()) if qualify_table_fn else str(table).upper()
                            invalid_tables_cache.add(tbl_for_cache)
                        except Exception:
                            pass
                logger.debug(f"Database query failed: {error}")
                return False

            # Check if result has data - handle different result types
            if result is None:
                return False
            elif isinstance(result, pd.DataFrame):
                return not result.empty
            elif isinstance(result, str):
                return "No data found" not in result and "exception" not in result.lower()
            elif isinstance(result, (list, tuple)):
                return len(result) > 0
            else:
                return bool(result)

        except Exception as e:
            logger.debug(f"Database query failed for {db_type}: {e}")
            return False

    num_date_match = {}
    cell_match = {}

    # Process columns from schema_dict
    column_names = schema_dict.get('column_names_original', [])
    column_types = schema_dict.get('column_types', [])
    table_names = schema_dict.get('table_names_original', [])

    logger.debug(f"Starting cell value linking for {len(column_names)} columns in {db_type} database")

    # Snowflake-specific preparation: schema qualification and failure cache
    table_schema_map = {}
    invalid_tables_cache = set()
    default_sf_schema = None

    def _load_credentials_dict(cred):
        try:
            if isinstance(cred, dict):
                return cred.get("snowflake", cred)
            if isinstance(cred, (str, Path)):
                return load_dataset(cred)
        except Exception:
            return None
        return None

    total_probes_remaining = None
    if db_type == "snowflake":
        # Global safety budget to avoid excessive remote calls
        total_probes_remaining = 500
        cred_dict = _load_credentials_dict(credential)
        if isinstance(cred_dict, dict):
            default_sf_schema = cred_dict.get("schema") or cred_dict.get("SCHEMA")

        # Map table -> schema via INFORMATION_SCHEMA to avoid unqualified references
        try:
            candidate_tables = [str(t).upper() for t in schema_dict.get('table_names_original', [])]
            if candidate_tables:
                in_list = ", ".join([f"'{t}'" for t in candidate_tables])
                info_sql = f"SELECT TABLE_NAME, TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME IN ({in_list})"
                result, error = get_sql_exec_result(
                    db_type=db_type,
                    sql_query=info_sql,
                    credential_path=credential,
                    db_id=db_id,
                    db_path=str(db_path) if db_path else None
                )
                if not error and result is not None:
                    rows = []
                    if hasattr(result, 'values'):
                        rows = result.values
                    elif isinstance(result, list):
                        rows = result
                    for r in rows:
                        try:
                            tname = str(r[0]).upper()
                            tschema = str(r[1]).upper()
                            table_schema_map[tname] = tschema
                        except Exception:
                            continue
        except Exception:
            pass

    def qualify_table_fn_snowflake(table_upper: str) -> str:
        if '.' in table_upper:
            return table_upper
        schema_part = table_schema_map.get(table_upper) or (str(default_sf_schema).upper() if default_sf_schema else None)
        if schema_part:
            return f"{schema_part}.{table_upper}"
        return table_upper

    for col_id, (table_id, col_name) in enumerate(column_names):

        if table_id >= len(table_names):
            continue

        table_name = table_names[table_id]
        col_type = column_types[col_id] if col_id < len(column_types) else 'text'

        match_q_ids = []
        remaining_probes = 50 if db_type == "snowflake" else 10000
        for q_id, word in enumerate(tokens):
            if len(word.strip()) == 0:
                continue
            if word in STOPWORDS or word in PUNKS:
                continue

            num_flag = isnumber(word)
            if num_flag:
                # Check if column is numeric or time type - database agnostic
                numeric_types = ["number", "int", "integer", "float", "decimal", "numeric", "double"]
                time_types = ["time", "date", "datetime", "timestamp"]

                col_type_lower = col_type.lower()
                if any(t in col_type_lower for t in numeric_types):
                    num_date_match[f"{q_id},{col_id}"] = "NUMBER"
                elif any(t in col_type_lower for t in time_types):
                    num_date_match[f"{q_id},{col_id}"] = "TIME"
            else:
                if db_type == "snowflake":
                    qualified_tbl = qualify_table_fn_snowflake(str(table_name).upper())
                    if qualified_tbl in invalid_tables_cache:
                        continue
                    if remaining_probes <= 0 or (total_probes_remaining is not None and total_probes_remaining <= 0):
                        continue
                # Check cell value match using the unified query
                if db_word_match(
                    word,
                    col_name,
                    table_name,
                    db_type,
                    db_path,
                    db_id,
                    credential,
                    exact=False,
                    invalid_tables_cache=invalid_tables_cache,
                    qualify_table_fn=(qualify_table_fn_snowflake if db_type == "snowflake" else None),
                ):
                    match_q_ids.append(q_id)
                if db_type == "snowflake":
                    remaining_probes -= 1
                    if total_probes_remaining is not None:
                        total_probes_remaining -= 1

        # Process consecutive matches for exact matching
        f = 0
        while f < len(match_q_ids):
            t = f + 1
            while t < len(match_q_ids) and match_q_ids[t] == match_q_ids[t - 1] + 1:
                t += 1
            q_f, q_t = match_q_ids[f], match_q_ids[t - 1] + 1
            words = [token for token in tokens[q_f: q_t]]

            # Try exact match first
            if db_word_match(
                ' '.join(words),
                col_name,
                table_name,
                db_type,
                db_path,
                db_id,
                credential,
                exact=True,
                invalid_tables_cache=invalid_tables_cache,
                qualify_table_fn=(qualify_table_fn_snowflake if db_type == "snowflake" else None),
            ):
                for q_id in range(q_f, q_t):
                    cell_match[f"{q_id},{col_id}"] = CELL_EXACT_MATCH_FLAG
            else:
                for q_id in range(q_f, q_t):
                    cell_match[f"{q_id},{col_id}"] = CELL_PARTIAL_MATCH_FLAG
            f = t

    logger.debug(f"Cell value linking completed: {len(num_date_match)} numeric matches, {len(cell_match)} cell matches")
    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
    return cv_link


def match_shift(q_col_match, q_tab_match, cell_match):
    q_id_to_match = collections.defaultdict(list)
    for match_key in q_col_match.keys():
        q_id = int(match_key.split(',')[0])
        c_id = int(match_key.split(',')[1])
        type_ = q_col_match[match_key]
        q_id_to_match[q_id].append((type_, c_id))
    for match_key in q_tab_match.keys():
        q_id = int(match_key.split(',')[0])
        t_id = int(match_key.split(',')[1])
        type_ = q_tab_match[match_key]
        q_id_to_match[q_id].append((type_, t_id))
    relevant_q_ids = list(q_id_to_match.keys())

    priority = []
    for q_id in q_id_to_match.keys():
        q_id_to_match[q_id] = list(set(q_id_to_match[q_id]))
        priority.append((len(q_id_to_match[q_id]), q_id))
    priority.sort()
    matches = []
    new_q_col_match, new_q_tab_match = dict(), dict()
    for _, q_id in priority:
        if not list(set(matches) & set(q_id_to_match[q_id])):
            exact_matches = []
            for match in q_id_to_match[q_id]:
                if match[0] in [COL_EXACT_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                    exact_matches.append(match)
            if exact_matches:
                res = exact_matches
            else:
                res = q_id_to_match[q_id]
            matches.extend(res)
        else:
            res = list(set(matches) & set(q_id_to_match[q_id]))
        for match in res:
            type_, c_t_id = match
            if type_ in [COL_PARTIAL_MATCH_FLAG, COL_EXACT_MATCH_FLAG]:
                new_q_col_match[f'{q_id},{c_t_id}'] = type_
            if type_ in [TAB_PARTIAL_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                new_q_tab_match[f'{q_id},{c_t_id}'] = type_

    new_cell_match = dict()
    for match_key in cell_match.keys():
        q_id = int(match_key.split(',')[0])
        if q_id in relevant_q_ids:
            continue
        # if cell_match[match_key] == CELL_EXACT_MATCH_FLAG:
        new_cell_match[match_key] = cell_match[match_key]

    return new_q_col_match, new_q_tab_match, new_cell_match


# Inline from utils.post_process
def process_duplication(sql):
    return sql.strip().split("/*")[0]


# Inline get_tables from utils.utils
class SqliteTable(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def get_tables(path_db, db_type='sqlite', credential=None, db_id=None):
    """
    Get table information using get_sql_exec_result for database abstraction.
    Supports multiple database types.
    """
    try:
        # Get table names using the unified approach
        table_names = get_table_names_unified(path_db, db_type, credential, db_id=db_id)

        res = list()
        for table_name in table_names:
            # Get schema information
            schema = get_table_schema_unified(table_name, path_db, db_type, credential, db_id=db_id)

            # Create table object
            res.append(
                SqliteTable(
                    name=table_name,
                    schema=schema,
                    data=None  # Data is not loaded by default
                )
            )

        return res

    except Exception as e:
        logger.error(f"Failed to get tables: {e}")
        return []


def get_table_names_unified(path_db, db_type='sqlite', credential=None, db_id=None):
    """
    Get table names using get_sql_exec_result for database abstraction.
    """
    try:
        # Construct appropriate query for different database types
        if db_type == 'sqlite':
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        elif db_type == 'big_query':
            query = "SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_type='BASE TABLE'"
        elif db_type == 'snowflake':
            query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
        else:
            query = "SELECT table_name FROM information_schema.tables WHERE table_type='BASE TABLE'"

        result, error = get_sql_exec_result(
            db_type=db_type,
            sql_query=query,
            db_path=str(path_db) if path_db else None,
            credential_path=credential,
            db_id=db_id
        )

        if error or result is None:
            logger.warning(f"Failed to get table names: {error}")
            return []

        # Extract table names from result
        if hasattr(result, 'values'):
            table_names = [row[0] for row in result.values]
        elif isinstance(result, list):
            table_names = [row[0] if isinstance(row, (list, tuple)) else str(row) for row in result]
        else:
            table_names = []

        return table_names

    except Exception as e:
        logger.error(f"Failed to get table names: {e}")
        return []


def get_table_schema_unified(table_name, path_db, db_type='sqlite', credential=None, db_id=None):
    """
    Get table schema (column names) using get_sql_exec_result for database abstraction.
    """
    try:
        # Construct appropriate query for different database types
        if db_type == 'sqlite':
            query = f'PRAGMA table_info("{table_name}")'
        elif db_type == 'big_query':
            query = f"SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE table_name='{table_name}'"
        elif db_type == 'snowflake':
            query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{table_name.upper()}'"
        else:
            query = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}'"

        result, error = get_sql_exec_result(
            db_type=db_type,
            sql_query=query,
            db_path=str(path_db) if path_db else None,
            credential_path=credential,
            db_id=db_id
        )

        if error or result is None:
            logger.warning(f"Failed to get schema for table {table_name}: {error}")
            return []

        # Extract column names from result
        if db_type == 'sqlite':
            # For SQLite PRAGMA, column names are in the second column (index 1)
            if hasattr(result, 'values'):
                schema = [row[1] for row in result.values]
            elif isinstance(result, list):
                schema = [row[1] if len(row) > 1 else str(row) for row in result]
            else:
                schema = []
        else:
            # For other databases, column names are in the first column
            if hasattr(result, 'values'):
                schema = [row[0] for row in result.values]
            elif isinstance(result, list):
                schema = [row[0] if isinstance(row, (list, tuple)) else str(row) for row in result]
            else:
                schema = []

        return schema

    except Exception as e:
        logger.error(f"Failed to get schema for table {table_name}: {e}")
        return []


# Prompt Classes
class BasicPrompt(object):
    def __init__(self, *args, **kwargs):
        pass

    def format_target(self, example):
        return self.format_question(example) + "\nSELECT "

    def format_question(self, example):
        raise NotImplementedError()

    def get_extra_info(self, db_id):
        return None


class SQLPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        # Handle different database types for schema extraction
        db_type = example.get('db_type', 'sqlite')
        path_db = example.get('path_db')
        credential = example.get('credential_path')
        db_id = example.get('db_id')

        if path_db:
            try:
                sqls = get_sql_for_database(path_db, db_type, credential, db_id=db_id)
                if sqls:
                    prompt_info = self.template_info.format("\n\n".join(sqls))
                else:
                    raise ValueError("No SQL schemas retrieved")
            except Exception as e:
                logger.warning(f"Failed to get SQL schema from database: {e}")
                # Fallback to table-based schema representation
                tables_info = []
                for table in example.get("tables", []):
                    table_sql = f"CREATE TABLE {table.name} ({', '.join(table.schema)});"
                    tables_info.append(table_sql)
                prompt_info = self.template_info.format("\n\n".join(tables_info))
        else:
            # For cases when path_db is not available
            tables_info = []
            for table in example.get("tables", []):
                table_sql = f"CREATE TABLE {table.name} ({', '.join(table.schema)});"
                tables_info.append(table_sql)
            prompt_info = self.template_info.format("\n\n".join(tables_info))

        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])
        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]
        return "\n\n".join(prompt_components)


class TextPrompt(BasicPrompt):
    template_info = "Given the following database schema:\n{}"
    template_question = "Answer the following: {}"

    def format_question(self, example):
        schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in example["tables"]])
        prompt_info = self.template_info.format(schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])
        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]
        return "\n".join(prompt_components)


class NumberSignPrompt(BasicPrompt):
    template_info = "### Complete sqlite SQL query only and with no explanation\n### SQLite SQL tables, with their properties:\n#\n{}\n#"
    template_question = "### {}"

    def format_question(self, example):
        schemas = "\n".join([f"# {_.name}({', '.join(_.schema)})" for _ in example["tables"]])
        prompt_info = self.template_info.format(schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])
        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]
        return "\n".join(prompt_components)


class BaselinePrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_target(self, example):
        return self.format_question(example) + "\nA: SELECT "

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class InstructionPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class TextWithForeignKeyPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class NumberSignWithForeignKeyPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class BaselineWithoutForeignKeyPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class InstructionWithForeignKeyPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class SQLWithRulePrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class TextWithRulePrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class NumberSignWithoutRulePrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class InstructionWithRulePrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class SQLCOTPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_target(self, example):
        return self.format_question(example) + "\nA: SELECT "

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class TextCOTPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_target(self, example):
        return self.format_question(example) + "\nA: SELECT "

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class NumberSignCOTPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_target(self, example):
        return self.format_question(example) + "\nA: SELECT "

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class InstructionCOTPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_target(self, example):
        return self.format_question(example) + "\nA: SELECT "

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


class CBRPrompt(BasicPrompt):
    template_info = "/* Given the following database schema: */\n{}"
    template_question = "/* Answer the following: {} */"

    def format_target(self, example):
        return self.format_question(example) + "\nA: SELECT "

    def format_question(self, example):
        sqls = get_sql_for_database(example['path_db'], db_id=example.get('db_id'))
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)


# Example Format Classes
class SqlExampleStyle(object):
    def get_example_prefix(self):
        return "/* Some SQL examples are provided based on similar problems: */\n"

    def format_example(self, example):
        return example['query']


class QuestionSqlExampleStyle(object):
    def get_example_prefix(self):
        return "/* Some SQL examples are provided based on similar problems: */\n"

    def format_example(self, example):
        template_qa = "/* Answer the following: {} */\n{}"
        return template_qa.format(example['question'], example['query'])


class QuestionSqlWithRuleExampleStyle(object):
    def get_example_prefix(self):
        return "/* Some SQL examples are provided based on similar problems: */\n"

    def format_example(self, example):
        template_qa = "/* Answer the following: {} */\n{}"
        return template_qa.format(example['question'], example['query'])


class CompleteExampleStyle(object):
    def get_example_prefix(self):
        return "/* Some SQL examples are provided based on similar problems: */\n"

    def format_example(self, example):
        return example['query']


class NumberSignQuestionSqlExampleStyle(object):
    def get_example_prefix(self):
        return "/* Some SQL examples are provided based on similar problems: */\n"

    def format_example(self, example):
        return example['query']


class BaselineQuestionSqlExampleStyle(object):
    def get_example_prefix(self):
        return "/* Some SQL examples are provided based on similar problems: */\n"

    def format_example(self, example):
        return example['query']


# ICL Prompt
class BasicICLPrompt(object):
    NUM_EXAMPLE = None
    SEP_EXAMPLE = "\n\n"

    def __init__(self, tokenizer="approx", *args, **kwargs):
        self.tokenizer = tokenizer
        self.example_qualities = []
        self.pattern_similarities = []

    def count_tokens(self, string):
        return len(string.split())

    def record_example_quality(self, examples, target):
        quality_list = []
        for example in examples:
            quality_list.append(jaccard_similarity(example["query_skeleton"], target["query_skeleton"]))
        self.example_qualities.append(quality_list)

    def get_example_quality(self):
        if self.example_qualities:
            return np.mean([num for row in self.example_qualities for num in row])
        else:
            return 1

    def get_example_quality_for_each(self):
        if self.example_qualities:
            return np.mean(self.example_qualities, axis=1)
        else:
            return []

    def record_pattern_similarity(self, examples, target):
        similarity_list = []
        for example in examples:
            similarity_list.append(jaccard_similarity(example["query_skeleton"], target["query_skeleton"]))
        self.pattern_similarities.append(similarity_list)

    def get_pattern_similarity(self):
        if self.pattern_similarities:
            return np.mean(self.pattern_similarities, axis=1)
        else:
            return []

    def format(self, target, max_seq_len, max_ans_len, scope_factor, cross_domain):
        # Ensure required methods are available
        self._ensure_required_methods()

        # Proceed with prompt construction
        suffix = self.format_target(target)[len(self.format_question(target)):]
        prompt_str = ""
        token_cnt = 0

        # Add few-shot examples if k_shot > 0
        if getattr(self, 'NUM_EXAMPLE', 0) > 0:
            examples = self._get_examples_safe(target, self.NUM_EXAMPLE, cross_domain)
            if examples:
                if hasattr(self, 'record_example_quality'):
                    self.record_example_quality(examples, target)
                if hasattr(self, 'record_pattern_similarity'):
                    self.record_pattern_similarity(examples, target)

                formatted_examples = [self.format_example(ex) for ex in examples]
                examples_prompt = self.get_example_prefix() + self.SEP_EXAMPLE.join(
                    formatted_examples) + self.SEP_EXAMPLE
                prompt_str += examples_prompt
                token_cnt += self.count_tokens(examples_prompt)

        # Add the main question
        question_prompt = self.format_question(target)
        prompt_str += question_prompt + suffix
        token_cnt += self.count_tokens(question_prompt) + self.count_tokens(suffix)

        # Truncate if necessary
        if token_cnt > max_seq_len:
            logger.warning(f"Prompt too long ({token_cnt} tokens), truncating...")
            # Simple truncation strategy - keep question and reduce examples
            if getattr(self, 'NUM_EXAMPLE', 0) > 0:
                self.NUM_EXAMPLE = max(0, self.NUM_EXAMPLE - 1)
                return self.format(target, max_seq_len, max_ans_len, scope_factor, cross_domain)

        return {"prompt": prompt_str, "prompt_tokens": token_cnt}

    def _ensure_required_methods(self):
        """Ensure all required methods are available"""
        if not hasattr(self, 'format_question'):
            self.format_question = self._default_format_question
        if not hasattr(self, 'format_target'):
            self.format_target = self._default_format_target
        if not hasattr(self, 'get_example_prefix'):
            self.get_example_prefix = self._default_get_example_prefix
        if not hasattr(self, 'format_example'):
            self.format_example = self._default_format_example

    def _default_format_question(self, ex):
        """Default question formatting"""
        if 'tables' in ex:
            schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in ex["tables"]])
            return f"Given the following database schema:\n{schemas}\n\nAnswer the following: {ex['question']}"
        else:
            return f"Answer the following: {ex['question']}"

    def _default_format_target(self, ex):
        """Default target formatting"""
        return self.format_question(ex) + "\nSELECT "

    def _default_get_example_prefix(self):
        """Default example prefix"""
        return "/* Some SQL examples are provided based on similar problems: */\n"

    def _default_format_example(self, ex):
        """Default example formatting"""
        return f"/* Answer the following: {ex['question']} */\n{ex.get('query', 'SELECT')}"

    def _get_examples_safe(self, target, num_example, cross_domain):
        """Safely get examples with fallback"""
        if hasattr(self, 'get_examples'):
            try:
                return self.get_examples(target, num_example, cross_domain)
            except:
                pass

        # Fallback: return empty list for 0-shot
        return []


# Example Selector Classes
class BasicExampleSelector(object):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.train_json = self.data.get('train_json', [])
        self.db_ids = [d.get('db_id') for d in self.train_json]
        self.train_questions = [d.get('question') for d in self.train_json]

    def get_examples(self, question, num_example, cross_domain=False):
        pass

    def domain_mask(self, question, db_id):
        return [i for i, q in enumerate(self.train_questions) if self.db_ids[i] == db_id and q == question]

    def retrieve_index(self, question, db_id):
        mask = self.domain_mask(question, db_id)
        if mask:
            return mask[0]
        return -1


class RandomExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        random.seed(0)

    def get_examples(self, target, num_example, cross_domain=False):
        indexes = list(range(len(self.train_json)))
        if cross_domain:
            indexes = [i for i in indexes if self.db_ids[i] != target['db_id']]
        selected_indexes = random.sample(indexes, min(num_example, len(indexes)))
        return [self.train_json[i] for i in selected_indexes]


class CosineSimilarExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.train_embeddings = np.random.rand(len(self.train_questions), 768)  # Dummy

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = np.random.rand(1, 768)  # Dummy
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistanceSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.train_embeddings = np.random.rand(len(self.train_questions), 768)  # Dummy

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = np.random.rand(1, 768)  # Dummy
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(d, i) for d, i in zip(distances, range(len(distances)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = []
        for d, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.train_embeddings = np.random.rand(len(self.train_questions), 768)  # Dummy

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = np.random.rand(1, 768)  # Dummy
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(d, i) for d, i in zip(distances, range(len(distances)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = []
        for d, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceSkeletonSimilarityThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.train_embeddings = np.random.rand(len(self.train_questions), 768)  # Dummy

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = np.random.rand(1, 768)  # Dummy
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistanceQuestionMaskSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.train_embeddings = np.random.rand(len(self.train_questions), 768)  # Dummy

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = np.random.rand(1, 768)  # Dummy
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistancePreSkeletonSimilarityThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.train_embeddings = np.random.rand(len(self.train_questions), 768)  # Dummy

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = np.random.rand(1, 768)  # Dummy
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistancePreSkeletonSimilarityPlusSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.train_embeddings = np.random.rand(len(self.train_questions), 768)  # Dummy

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = np.random.rand(1, 768)  # Dummy
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistanceMaskPreSkeletonSimilarityThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.train_embeddings = np.random.rand(len(self.train_questions), 768)  # Dummy

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = np.random.rand(1, 768)  # Dummy
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistanceMaskPreSkeletonSimilarityThresholdShiftSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        self.train_embeddings = np.random.rand(len(self.train_questions), 768)  # Dummy

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = np.random.rand(1, 768)  # Dummy
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


# Prompt Factory
def prompt_factory(repr_type, k_shot, example_format, selector_type):
    repr_cls = get_repr_cls(repr_type)
    class_dict = {
        'name': f"{repr_type}_{k_shot}-SHOT",
        'NUM_EXAMPLE': k_shot
    }
    if k_shot == 0:
        PromptClass = abc.ABCMeta('PromptClass', (repr_cls, BasicICLPrompt), class_dict)
    else:
        example_format_cls = get_example_format_cls(example_format)
        selector_cls = get_example_selector(selector_type)
        class_dict['name'] = f"{repr_type}_{k_shot}-SHOT_{selector_type}_{example_format}-EXAMPLE"
        PromptClass = abc.ABCMeta('PromptClass', (selector_cls, example_format_cls, repr_cls, BasicICLPrompt),
                                  class_dict)
    return PromptClass


def get_repr_cls(repr_type):
    if repr_type == REPR_TYPE.CODE_REPRESENTATION:
        return SQLPrompt
    elif repr_type == REPR_TYPE.TEXT_REPRESENTATION:
        return TextPrompt
    elif repr_type == REPR_TYPE.OPENAI_DEMOSTRATION:
        return NumberSignPrompt
    elif repr_type == REPR_TYPE.BASIC:
        return BaselinePrompt
    elif repr_type == REPR_TYPE.ALPACA_SFT:
        return InstructionPrompt
    elif repr_type == REPR_TYPE.OPENAI_DEMOSTRATION_WFK:
        return NumberSignWithForeignKeyPrompt
    elif repr_type == REPR_TYPE.BASIC_WOFK:
        return BaselineWithoutForeignKeyPrompt
    elif repr_type == REPR_TYPE.TEXT_REPRESENTATION_WFK:
        return TextWithForeignKeyPrompt
    elif repr_type == REPR_TYPE.ALPACA_SFT_WFK:
        return InstructionWithForeignKeyPrompt
    elif repr_type == REPR_TYPE.OPENAI_DEMOSTRATION_WORULE:
        return NumberSignWithoutRulePrompt
    elif repr_type == REPR_TYPE.CODE_REPRESENTATION_WRULE:
        return SQLWithRulePrompt
    elif repr_type == REPR_TYPE.ALPACA_SFT_WRULE:
        return InstructionWithRulePrompt
    elif repr_type == REPR_TYPE.TEXT_REPRESENTATION_WRULE:
        return TextWithRulePrompt
    elif repr_type == REPR_TYPE.CODE_REPRESENTATION_COT:
        return SQLCOTPrompt
    elif repr_type == REPR_TYPE.TEXT_REPRESENTATION_COT:
        return TextCOTPrompt
    elif repr_type == REPR_TYPE.OPENAI_DEMOSTRATION_COT:
        return NumberSignCOTPrompt
    elif repr_type == REPR_TYPE.ALPACA_SFT_COT:
        return InstructionCOTPrompt
    elif repr_type == REPR_TYPE.CBR:
        return CBRPrompt
    else:
        raise ValueError(f"{repr_type} is not supported yet")


def get_example_format_cls(example_format):
    if example_format == EXAMPLE_TYPE.ONLY_SQL:
        return SqlExampleStyle
    elif example_format == EXAMPLE_TYPE.QA:
        return QuestionSqlExampleStyle
    elif example_format == EXAMPLE_TYPE.QAWRULE:
        return QuestionSqlWithRuleExampleStyle
    elif example_format == EXAMPLE_TYPE.COMPLETE:
        return CompleteExampleStyle
    elif example_format == EXAMPLE_TYPE.OPENAI_DEMOSTRATION_QA:
        return NumberSignQuestionSqlExampleStyle
    elif example_format == EXAMPLE_TYPE.BASIC_QA:
        return BaselineQuestionSqlExampleStyle
    else:
        raise ValueError(f"{example_format} is not supported yet")


def get_example_selector(selector_type):
    if selector_type == SELECTOR_TYPE.COS_SIMILAR:
        return CosineSimilarExampleSelector
    elif selector_type == SELECTOR_TYPE.RANDOM:
        return RandomExampleSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE:
        return EuclideanDistanceSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_THRESHOLD:
        return EuclideanDistanceThresholdSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_SKELETON_SIMILARITY_THRESHOLD:
        return EuclideanDistanceSkeletonSimilarityThresholdSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_QUESTION_MASK:
        return EuclideanDistanceQuestionMaskSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_PRE_SKELETON_SIMILARITY_THRESHOLD:
        return EuclideanDistancePreSkeletonSimilarityThresholdSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_PRE_SKELETON_SIMILARITY_PLUS:
        return EuclideanDistancePreSkeletonSimilarityPlusSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD:
        return EuclideanDistanceMaskPreSkeletonSimilarityThresholdSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD_SHIFT:
        return EuclideanDistanceMaskPreSkeletonSimilarityThresholdShiftSelector
    else:
        raise ValueError(f"{selector_type} is not supported yet")


# Similarly for others

# Main Class
class DAILSQLGenerate(BaseGenerator):
    NAME = "DAILSQL"
    OUTPUT_NAME = "pred_sql"

    @property
    def name(self):
        return self.NAME

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm=None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            use_external: bool = True,
            use_few_shot: bool = True,
            prompt_repr=REPR_TYPE.TEXT_REPRESENTATION,
            k_shot=0,
            example_type=EXAMPLE_TYPE.QA,
            selector_type=SELECTOR_TYPE.RANDOM,
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
        self.prompt_repr = prompt_repr
        self.k_shot = k_shot
        self.example_type = example_type
        self.selector_type = selector_type

        # 安全地初始化 db_path 和 credential，检查 dataset 是否为 None
        if db_path is not None:
            self.db_path = db_path
        elif self.dataset is not None:
            self.db_path = self.dataset.db_path
        else:
            self.db_path = None

        if credential is not None:
            self.credential = credential
        elif self.dataset is not None:
            self.credential = self.dataset.credential
        else:
            self.credential = None

        # Initialize prompt system
        self.prompt = None
        if self.dataset and self.use_few_shot:
            try:
                self.prompt = prompt_factory(self.prompt_repr, self.k_shot, self.example_type, self.selector_type)(
                    data=self._build_dataset_adapter(), tokenizer="approx")
                logger.debug(f"Initialized DAIL-SQL prompt system: {self.prompt_repr}, {self.k_shot}-shot")
            except Exception as e:
                logger.warning(f"Failed to initialize DAIL-SQL prompt system: {e}, using fallback")
                self.prompt = None

    def act(self, item, schema=None, schema_links=None, **kwargs):
        """Main DAIL-SQL generation method following the Generator interface"""
        logger.info(f"DAILSQLGenerator 开始处理样本 {item}")

        # Validate inputs
        is_valid, error_msg = self._validate_inputs(item, schema)
        if not is_valid:
            logger.error(f"输入验证失败: {error_msg}")
            raise ValueError(error_msg)

        try:
            # Get data row
            row = self.dataset[item]
            question = row['question']
            db_type = row['db_type']
            db_id = row["db_id"]
            db_path = self._get_db_path(row, db_id, db_type)
            logger.debug(f"处理问题: {question[:100]}... (数据库: {db_id}, 类型: {db_type})")

            # Load external knowledge if available
            if self.use_external:
                external_knowledge = self._load_external_knowledge(row.get("external", None))
                if external_knowledge:
                    question += "\n" + external_knowledge
                    logger.debug("已加载外部知识")

            # Load and process schema
            logger.debug("开始处理数据库模式...")
            if isinstance(schema, (str, PathLike)) and Path(schema).exists():
                schema = load_dataset(schema)

            if schema is None:
                instance_schema_path = row.get("instance_schemas")
                if instance_schema_path:
                    schema = load_dataset(instance_schema_path)
                    logger.debug(f"从实例模式路径加载模式: {instance_schema_path}")
                else:
                    logger.debug("从数据集获取数据库模式")
                    schema = self.dataset.get_db_schema(item)

                if schema is None:
                    raise ValueError("Failed to load a valid database schema for the sample!")

            # Normalize schema type
            if isinstance(schema, dict):
                schema = single_central_process(schema)
            elif isinstance(schema, list):
                schema = pd.DataFrame(schema)

            if isinstance(schema, pd.DataFrame):
                schema_str = parse_schema_from_df(schema)
                schema_dict = self._build_compatible_schema_dict(schema)
            else:
                raise ValueError("Invalid schema format")

            logger.debug("数据库模式处理完成")

            # Build target object for DAIL-SQL format
            target = {
                'question': question,
                'db_id': db_id,
                'db_type': db_type,  # Include database type for database-agnostic processing
                'path_db': db_path,
                'tables': self._get_tables_from_schema(schema_dict),
                'query': row.get('query', 'SELECT'),
                'column_names_original': schema_dict.get('column_names_original', []),
                'table_names_original': schema_dict.get('table_names_original', []),
                'query_skeleton': self._get_query_skeleton(row.get('query', 'SELECT'), schema_dict),
                'credential_path': self.credential  # Include credential path for non-SQLite databases
            }

            # Compute schema linking if not provided
            if schema_links is None:
                logger.debug("开始计算模式链接...")
                question_toks = question.split()

                # Schema column/table linking using original DAIL-SQL approach
                sc_link = compute_schema_linking(
                    question_toks,
                    [col[1] for col in schema_dict['column_names_original']],
                    schema_dict['table_names_original']
                )

                # Cell value linking with database queries using get_sql_exec_result
                cv_link = compute_cell_value_linking(
                    question_toks,
                    schema_dict,
                    db_type,
                    db_path,
                    db_id,
                    self.credential
                )

                # Apply match shifting as in original DAIL-SQL
                q_col_match, q_tab_match, cell_match = match_shift(
                    sc_link['q_col_match'],
                    sc_link['q_tab_match'],
                    cv_link['cell_match']
                )

                target['sc_link'] = {'q_col_match': q_col_match, 'q_tab_match': q_tab_match}
                target['cv_link'] = {'num_date_match': cv_link['num_date_match'], 'cell_match': cell_match}
                target['question_for_copying'] = question_toks
                logger.debug("模式链接计算完成")
            else:
                # Use provided schema links
                if isinstance(schema_links, (str, PathLike)):
                    schema_links = load_dataset(schema_links)
                target.update(schema_links)

            # Format prompt using DAIL-SQL prompt system
            logger.debug("开始格式化提示...")
            if hasattr(self, 'prompt') and self.prompt:
                try:
                    prompt_data = self.prompt.format(
                        target=target,
                        max_seq_len=2048,
                        max_ans_len=200,
                        scope_factor=100,
                        cross_domain=True
                    )
                    prompt = prompt_data['prompt']
                    logger.debug(f"使用 DAIL-SQL 提示系统，token 数: {prompt_data.get('prompt_tokens', 0)}")
                except Exception as e:
                    logger.warning(f"DAIL-SQL 提示系统失败，使用回退提示: {e}")
                    prompt = self._build_fallback_prompt(target, schema_str)
            else:
                # Fallback to basic prompt if prompt system not initialized
                prompt = self._build_fallback_prompt(target, schema_str)

            # Generate SQL using LLM
            logger.debug("开始生成 SQL...")
            try:
                res = self.llm.complete(prompt)
                res_text = res.text if hasattr(res, 'text') else str(res)
            except Exception as e:
                logger.error(f"LLM 生成失败: {e}")
                return "SELECT 1"  # Return a valid SQL as fallback

            # Post-process SQL with database type information
            sql = self._post_process_sql(res_text, db_type)

            # Save result if required
            if self.is_save:
                self._save_result(sql, row, item)

            logger.debug(f"生成的 SQL: {sql[:100]}...")
            logger.info(f"DAILSQLGenerator 样本 {item} 处理完成")
            return sql

        except Exception as e:
            logger.error(f"DAILSQLGenerator act 错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "SELECT 1"  # Return a valid SQL as fallback

    def _build_compatible_schema_dict(self, schema_df: pd.DataFrame) -> Dict:
        """将 DataFrame 格式的 schema 转换为 DAIL-SQL 兼容的字典格式"""
        schema_dict = {
            'column_names_original': [],
            'table_names_original': [],
            'column_types': [],
            'connection': None  # 暂时设为 None，实际使用时可能需要数据库连接
        }

        # 获取所有唯一的表名
        table_names = schema_df['table_name'].unique().tolist()
        schema_dict['table_names_original'] = table_names

        # 构建 column_names_original 格式: [(table_id, column_name), ...]
        # 其中 table_id 是表名在 table_names_original 中的索引
        for _, row in schema_df.iterrows():
            table_name = row['table_name']
            column_name = row['column_name']
            table_id = table_names.index(table_name)
            schema_dict['column_names_original'].append((table_id, column_name))
            schema_dict['column_types'].append(row.get('column_types', 'text'))

        return schema_dict

    def _get_tables_from_schema(self, schema_dict: Dict) -> List:
        """从 schema 字典中提取表信息，用于构建 tables 对象"""
        tables = []
        table_names = schema_dict.get('table_names_original', [])

        for table_name in table_names:
            # 获取该表的所有列
            table_columns = []
            for table_id, col_name in schema_dict.get('column_names_original', []):
                if table_id < len(table_names) and table_names[table_id] == table_name:
                    table_columns.append(col_name)

            # 创建 SqliteTable 对象
            table_obj = SqliteTable(
                name=table_name,
                schema=table_columns,
                data=None
            )
            tables.append(table_obj)

        return tables

    def _simplified_schema_linking(self, question_toks, schema_dict):
        """简化的 schema linking 实现，避免复杂的数据库连接"""
        q_col_match = {}
        q_tab_match = {}

        # 获取列名和表名
        columns = [c[1] for c in schema_dict['column_names_original']]
        tables = schema_dict['table_names_original']

        # 简单的字符串匹配
        for q_id, word in enumerate(question_toks):
            word_lower = word.lower()

            # 检查列名匹配
            for col_id, col_name in enumerate(columns):
                if word_lower in col_name.lower() or col_name.lower() in word_lower:
                    q_col_match[f"{q_id},{col_id}"] = COL_PARTIAL_MATCH_FLAG
                    if word_lower == col_name.lower():
                        q_col_match[f"{q_id},{col_id}"] = COL_EXACT_MATCH_FLAG

            # 检查表名匹配
            for tab_id, tab_name in enumerate(tables):
                if word_lower in tab_name.lower() or tab_name.lower() in word_lower:
                    q_tab_match[f"{q_id},{tab_id}"] = TAB_PARTIAL_MATCH_FLAG
                    if word_lower == tab_name.lower():
                        q_tab_match[f"{q_id},{tab_id}"] = TAB_EXACT_MATCH_FLAG

        return q_col_match, q_tab_match

    def _get_db_path(self, row, db_id, db_type):
        """
        Get database path/identifier based on db_type.
        Consistent with LinkAlignGenerator approach.
        """
        if db_type == "sqlite":
            return Path(self.db_path) / (db_id + ".sqlite") if self.db_path else None
        elif db_type in ["big_query", "snowflake"]:
            # For cloud databases, db_path might be db_id or connection string
            return db_id
        else:
            return self.db_path

    def _get_query_skeleton(self, query, schema_dict):
        """Get SQL query skeleton"""
        if not query or not query.strip().upper().startswith('SELECT'):
            return query
        try:
            return sql2skeleton(query, schema_dict)
        except:
            return query

    def _build_fallback_prompt(self, target, schema_str):
        """
        Build a basic fallback prompt when DAIL-SQL prompt system is not available.
        Database-agnostic prompt construction.
        """
        db_type = target.get('db_type', 'sqlite')

        # Add database-specific instructions if needed
        db_instruction = ""
        if db_type == "big_query":
            db_instruction = "/* Note: Use BigQuery SQL syntax with backticks for table/column names */\n"
        elif db_type == "snowflake":
            db_instruction = "/* Note: Use Snowflake SQL syntax */\n"
        elif db_type == "sqlite":
            db_instruction = "/* Note: Use SQLite SQL syntax */\n"

        prompt = f"""{db_instruction}Given the following database schema:
{schema_str}

Answer the following: {target['question']}
SELECT """
        return prompt

    def _post_process_sql(self, sql_text, db_type='sqlite'):
        """
        Post-process generated SQL for different database types.
        Database-agnostic SQL cleaning and validation.
        """
        # Clean up the SQL text
        sql = " ".join(sql_text.replace("\n", " ").split())
        sql = process_duplication(sql)

        # Ensure SQL starts with SELECT
        if not sql.upper().startswith('SELECT'):
            if sql.startswith(' '):
                sql = 'SELECT' + sql
            else:
                sql = 'SELECT ' + sql

        # Database-specific SQL adjustments
        if db_type == "big_query":
            # BigQuery specific adjustments
            sql = self._adjust_sql_for_bigquery(sql)
        elif db_type == "snowflake":
            # Snowflake specific adjustments
            sql = self._adjust_sql_for_snowflake(sql)
        elif db_type == "sqlite":
            # SQLite specific adjustments
            sql = self._adjust_sql_for_sqlite(sql)

        return sql

    def _adjust_sql_for_bigquery(self, sql):
        """Adjust SQL for BigQuery syntax"""
        # BigQuery uses backticks for identifiers
        # This is a basic adjustment - more sophisticated parsing might be needed
        return sql

    def _adjust_sql_for_snowflake(self, sql):
        """Adjust SQL for Snowflake syntax"""
        # Snowflake uses double quotes for case-sensitive identifiers
        # This is a basic adjustment - more sophisticated parsing might be needed
        return sql

    def _adjust_sql_for_sqlite(self, sql):
        """Adjust SQL for SQLite syntax"""
        # SQLite uses standard SQL syntax
        return sql

    def _load_external_knowledge(self, external_path):
        """Load external knowledge if available"""
        if not external_path:
            return None
        try:
            external = load_dataset(external_path)
            if external and len(external) > 50:
                external = "####[External Prior Knowledge]:\n" + external
                return external
        except Exception as e:
            logger.warning(f"加载外部知识失败: {e}")
        return None

    def _save_result(self, sql, row, item):
        """Save the generated SQL result"""
        instance_id = row.get("instance_id", item)
        save_path = Path(self.save_dir)
        if self.dataset and self.dataset.dataset_index:
            save_path = save_path / str(self.dataset.dataset_index)
        save_path = save_path / f"{self.NAME}_{instance_id}.sql"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(sql, new_data_source=save_path)
        if self.dataset:
            self.dataset.setitem(item, "pred_sql", str(save_path))
        logger.debug(f"SQL 已保存到: {save_path}")

    def _build_dataset_adapter(self):
        """Build a dataset adapter for DAIL-SQL prompt system"""

        class DatasetAdapter:
            def __init__(self, dataset):
                self.dataset = dataset
                self.train_json = []
                self.db_ids = []
                self.train_questions = []

                # Build training data if available
                if hasattr(dataset, 'data') and dataset.data:
                    for i, item in enumerate(dataset.data):
                        if isinstance(item, dict):
                            # Convert Squrve format to DAIL-SQL format
                            adapted_item = {
                                'question': item.get('question', ''),
                                'query': item.get('query', 'SELECT'),
                                'db_id': item.get('db_id', 'default'),
                                'path_db': item.get('path_db', ''),
                                'tables': []  # Will be populated as needed
                            }
                            self.train_json.append(adapted_item)
                            self.db_ids.append(adapted_item['db_id'])
                            self.train_questions.append(adapted_item['question'])

            def get_train_json(self):
                return self.train_json

            def get_test_json(self):
                return []  # Not used for generation

        return DatasetAdapter(self.dataset)

    def _validate_inputs(self, item, schema=None):
        """Validate input parameters"""
        if self.dataset is None:
            return False, "Dataset not initialized"

        if self.llm is None:
            return False, "LLM not initialized"

        try:
            row = self.dataset[item]
            if 'question' not in row:
                return False, "Data sample missing 'question' field"
            if 'db_id' not in row:
                return False, "Data sample missing 'db_id' field"
            if 'db_type' not in row:
                return False, "Data sample missing 'db_type' field"
        except Exception as e:
            return False, f"Cannot access data sample: {e}"

        return True, ""


# Define process_duplication from existing
def process_duplication(sql):
    return sql.strip().split("/*")[0]


# Add helper functions for sql2skeleton
def isNegativeInt(string):
    return string.startswith("-") and string[1:].isdigit()


def isFloat(string):
    if string.startswith("-"):
        string = string[1:]
    s = string.split(".")
    if len(s) > 2:
        return False
    for s_i in s:
        if not s_i.isdigit():
            return False
    return True
