import collections
import re
import string
import numpy as np
import random
import sqlite3
import nltk
from nltk.corpus import stopwords
from os import PathLike
from typing import Union, Dict, List, Optional
from pathlib import Path
import pandas as pd

from loguru import logger
from sql_metadata import Parser

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)

import abc

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


def get_sql_for_database(path_db):
    # Full
    con = sqlite3.connect(str(path_db))
    cur = con.cursor()
    table_names = [row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    sqls = []
    for name in table_names:
        sql = cur.execute(f"SELECT sql FROM sqlite_master WHERE tbl_name='{name}'").fetchone()
        if sql:
            sqls.append(sql[0])
    con.close()
    return sqls


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
    def partial_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str in STOPWORDS or x_str in PUNKS:
            return False
        if re.match(rf"\b{re.escape(x_str)}\b", y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str == y_str:
            return True
        else:
            return False

    q_col_match = dict()
    q_tab_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0:
            continue
        col_id2list[col_id] = col_item

    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item

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


def compute_cell_value_linking(tokens, schema):
    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_partial_match(word, column, table, db_conn):
        cursor = db_conn.cursor()

        p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or " \
                f"{column} like '% {word} %' or {column} like '{word}'"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except Exception as e:
            return False

    def db_word_exact_match(word, column, table, db_conn):
        cursor = db_conn.cursor()

        p_str = f"select {column} from {table} where {column} like '{word}' or {column} like ' {word}' or " \
                f"{column} like '{word} ' or {column} like ' {word} '"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except Exception as e:
            return False

    num_date_match = {}
    cell_match = {}

    for col_id, column in enumerate(schema.columns):
        if col_id == 0:
            assert column.orig_name == "*"
            continue
        match_q_ids = []
        for q_id, word in enumerate(tokens):
            if len(word.strip()) == 0:
                continue
            if word in STOPWORDS or word in PUNKS:
                continue

            num_flag = isnumber(word)
            if num_flag:  # TODO refine the date and time match
                if column.type in ["number", "time"]:
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
            else:
                ret = db_word_partial_match(word, column.orig_name, column.table.orig_name, schema.connection)
                if ret:
                    # print(word, ret)
                    match_q_ids.append(q_id)
        f = 0
        while f < len(match_q_ids):
            t = f + 1
            while t < len(match_q_ids) and match_q_ids[t] == match_q_ids[t - 1] + 1:
                t += 1
            q_f, q_t = match_q_ids[f], match_q_ids[t - 1] + 1
            words = [token for token in tokens[q_f: q_t]]
            ret = db_word_exact_match(' '.join(words), column.orig_name, column.table.orig_name, schema.connection)
            if ret:
                for q_id in range(q_f, q_t):
                    cell_match[f"{q_id},{col_id}"] = CELL_EXACT_MATCH_FLAG
            else:
                for q_id in range(q_f, q_t):
                    cell_match[f"{q_id},{col_id}"] = CELL_PARTIAL_MATCH_FLAG
            f = t

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


def get_tables(path_db):
    if not Path(path_db).exists():
        raise RuntimeError(f"{path_db} not exists")

    # init sqlite connection
    connection = sqlite3.connect(str(path_db))
    cur = connection.cursor()

    # extract table information
    # table_info = parse_db(path_db, cur)  # Assuming parse_db is not needed or inline if necessary
    table_names = get_table_names(cur=cur)

    res = list()
    for table_name in table_names:
        # schema
        schema = [_[1] for _ in cur.execute(f'PRAGMA table_info("{table_name}")')]

        # data
        data = None

        # append table
        res.append(
            SqliteTable(
                name=table_name,
                schema=schema,
                data=data,
                # table_info=table_info.get(table_name, dict())
            )
        )

    cur.close()
    return res


def get_table_names(cur):
    table_names = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [_[0] for _ in table_names]
    return table_names


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
        sqls = get_sql_for_database(example["path_db"])
        prompt_info = self.template_info.format("\n\n".join(sqls))
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        sqls = get_sql_for_database(example['path_db'])
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
        # Explicitly define format_question if not available
        if not hasattr(self, 'format_question'):
            def format_question(ex):
                sqls = get_sql_for_database(ex['path_db'])
                prompt_info = "/* Given the following database schema: */\n{}".format("\n\n".join(sqls))
                prompt_extra = self.get_extra_info(ex['db_id']) if hasattr(self, 'get_extra_info') else ""
                prompt_question = "/* Answer the following: {} */".format(ex['question'])
                components = [prompt_info]
                if prompt_extra:
                    components.append(prompt_extra)
                components.append(prompt_question)
                return "\n\n".join(components)
            self.format_question = format_question
        # Explicitly define format_target if not available
        if not hasattr(self, 'format_target'):
            def format_target(ex):
                return self.format_question(ex) + "\nSELECT "
            self.format_target = format_target
        # Explicitly define get_example_prefix if not available
        if not hasattr(self, 'get_example_prefix'):
            def get_example_prefix():
                return "/* Some SQL examples are provided based on similar problems: */\n"
            self.get_example_prefix = get_example_prefix
        # Explicitly define format_example if not available
        if not hasattr(self, 'format_example'):
            def format_example(ex):
                return "/* Answer the following: {} */\n{}".format(ex['question'], ex['query'])
            self.format_example = format_example
        # Proceed with prompt construction
        suffix = self.format_target(target)[len(self.format_question(target)):]
        prompt_str = ""
        token_cnt = 0
        if self.NUM_EXAMPLE > 0:
            if not hasattr(self, 'get_examples'):
                def get_examples(t, n, cd):
                    # Fallback to random selection
                    import random
                    indexes = list(range(len(self.train_json)))
                    if cd:
                        indexes = [i for i in indexes if self.db_ids[i] != t['db_id']]
                    selected = random.sample(indexes, min(n, len(indexes)))
                    return [self.train_json[i] for i in selected]
                self.get_examples = get_examples
            examples = self.get_examples(target, self.NUM_EXAMPLE, cross_domain)
            if hasattr(self, 'record_example_quality'):
                self.record_example_quality(examples, target)
            if hasattr(self, 'record_pattern_similarity'):
                self.record_pattern_similarity(examples, target)
            formatted_examples = [self.format_example(ex) for ex in examples]
            examples_prompt = self.get_example_prefix() + self.SEP_EXAMPLE.join(formatted_examples) + self.SEP_EXAMPLE
            prompt_str += examples_prompt
            token_cnt += self.count_tokens(examples_prompt)
        question_prompt = self.format_question(target)
        prompt_str += question_prompt + suffix
        token_cnt += self.count_tokens(question_prompt) + self.count_tokens(suffix)
        # TODO: Implement truncation if prompt exceeds max_seq_len
        return {"prompt": prompt_str, "prompt_tokens": token_cnt}


# Example Selector Classes
class BasicExampleSelector(object):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.train_json = self.data.get('train_json', [])
        self.db_ids = [d.get('db_id') for d in self.train_json]
        self.train_questions = [d['question'] for d in self.train_json]

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

    def get_examples(self, target, num_example, cross_domain=False):
        # Use Jaccard similarity based on question text instead of embeddings
        similarities = []
        for i, train_question in enumerate(self.train_questions):
            sim = jaccard_similarity(target['question'], train_question)
            similarities.append(sim)
        
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

    def get_examples(self, target, num_example, cross_domain=False):
        # Use simple word-based distance instead of embeddings
        distances = []
        target_words = set(target['question'].lower().split())
        for train_question in self.train_questions:
            train_words = set(train_question.lower().split())
            # Calculate simple set-based distance
            distance = len(target_words.symmetric_difference(train_words))
            distances.append(distance)
        
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

    def get_examples(self, target, num_example, cross_domain=False):
        # Use simple word-based distance with threshold
        distances = []
        target_words = set(target['question'].lower().split())
        for train_question in self.train_questions:
            train_words = set(train_question.lower().split())
            distance = len(target_words.symmetric_difference(train_words))
            distances.append(distance)
        
        pairs = [(d, i) for d, i in zip(distances, range(len(distances)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = []
        for d, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            # Apply threshold (e.g., distance < 10)
            if d < 10:  # Simple threshold
                top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceSkeletonSimilarityThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

    def get_examples(self, target, num_example, cross_domain=False):
        # Use Jaccard similarity on question text with threshold
        similarities = []
        for train_question in self.train_questions:
            sim = jaccard_similarity(target['question'], train_question)
            similarities.append(sim)
        
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            # Apply similarity threshold
            if s > 0.1:  # Simple threshold
                top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistanceQuestionMaskSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

    def get_examples(self, target, num_example, cross_domain=False):
        # Use Jaccard similarity on question text
        similarities = []
        for train_question in self.train_questions:
            sim = jaccard_similarity(target['question'], train_question)
            similarities.append(sim)
        
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

    def get_examples(self, target, num_example, cross_domain=False):
        # Use Jaccard similarity on question text with threshold
        similarities = []
        for train_question in self.train_questions:
            sim = jaccard_similarity(target['question'], train_question)
            similarities.append(sim)
        
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            # Apply similarity threshold
            if s > 0.1:  # Simple threshold
                top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistancePreSkeletonSimilarityPlusSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

    def get_examples(self, target, num_example, cross_domain=False):
        # Use Jaccard similarity on question text with enhanced selection
        similarities = []
        for train_question in self.train_questions:
            sim = jaccard_similarity(target['question'], train_question)
            similarities.append(sim)
        
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            # Enhanced selection with higher threshold
            if s > 0.15:  # Higher threshold for better quality
                top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistanceMaskPreSkeletonSimilarityThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

    def get_examples(self, target, num_example, cross_domain=False):
        # Use Jaccard similarity on question text with mask-based threshold
        similarities = []
        for train_question in self.train_questions:
            sim = jaccard_similarity(target['question'], train_question)
            similarities.append(sim)
        
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            # Mask-based threshold
            if s > 0.1:  # Threshold for masked similarity
                top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break
        return [self.train_json[index] for (index, s) in top_pairs]


class EuclideanDistanceMaskPreSkeletonSimilarityThresholdShiftSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

    def get_examples(self, target, num_example, cross_domain=False):
        # Use Jaccard similarity on question text with shifted threshold
        similarities = []
        for train_question in self.train_questions:
            sim = jaccard_similarity(target['question'], train_question)
            similarities.append(sim)
        
        pairs = [(s, i) for s, i in zip(similarities, range(len(similarities)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = []
        for s, index in pairs_sorted:
            if cross_domain and self.train_json[index]['db_id'] == target['db_id']:
                continue
            if self.train_json[index]['question'] == target['question']:
                continue
            # Shifted threshold for better selection
            if s > 0.05:  # Lower threshold for more examples
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

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            use_external: bool = True,
            use_few_shot: bool = True,
            prompt_repr=REPR_TYPE.TEXT_REPRESENTATION,
            k_shot=0,
            example_type=EXAMPLE_TYPE.QA,
            selector_type=SELECTOR_TYPE.RANDOM,
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
        self.prompt_repr = prompt_repr
        self.k_shot = k_shot
        self.example_type = example_type
        self.selector_type = selector_type

        self.db_path = db_path or (dataset.db_path if dataset else None)
        self.credential = credential or (dataset.credential if dataset else None)

        if self.dataset:
            self.prompt = prompt_factory(self.prompt_repr, self.k_shot, self.example_type, self.selector_type)(
                data=self.dataset, tokenizer="approx")

        # Import necessary libraries for database operations
        from core.db_connect import get_sql_exec_result

    def act(self, item, schema=None, schema_links=None, **kwargs):
        try:
            if isinstance(item, int):
                row = self.dataset[item]
            else:
                row = item
                
            question = row.get('question', '')
            db_id = row.get('db_id', 'default_db')
            db_type = row.get('db_type', 'sqlite')
            
            logger.debug(f"Processing question: {question[:100]}... (DB: {db_id}, Type: {db_type})")

            # Load and process schema
            logger.debug("Processing database schema...")
            if isinstance(schema, (str, PathLike)):
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

            # Normalize schema type - 参考 DINSQLGenerate 的处理方式
            if isinstance(schema, dict):
                schema = single_central_process(schema)
            elif isinstance(schema, list):
                schema = pd.DataFrame(schema)

            if isinstance(schema, pd.DataFrame):
                schema_str = parse_schema_from_df(schema)
                # 为了兼容原有的 DAIL-SQL 逻辑，需要构建兼容的 schema 字典
                schema_dict = self._build_compatible_schema_dict(schema)
            else:
                raise ValueError("Invalid schema format")

            logger.debug("Database schema processed")

            # 构建 target 对象，使用处理后的 schema
            path_db = row.get('path_db', '')
            target = {
                'question': question,
                'db_id': db_id,
                'path_db': path_db,
                'tables': self._get_tables_from_schema(schema_dict),
                'query': row.get('query', 'SELECT'),
                'column_names_original': schema_dict.get('column_names_original', []),
                'table_names_original': schema_dict.get('table_names_original', []),
            }

            # Compute schema_links if not provided
            if schema_links is None:
                question_toks = target['question'].split()
                # 使用简化的 schema linking 实现
                q_col_match, q_tab_match = self._simplified_schema_linking(question_toks, schema_dict)
                
                # 简化的 cell value linking - 只处理数字和日期
                cv_link = {"num_date_match": {}, "cell_match": {}}
                for q_id, word in enumerate(question_toks):
                    # 检查是否为数字
                    try:
                        float(word)
                        # 找到匹配的数字类型列
                        for col_id, (table_id, col_name) in enumerate(schema_dict['column_names_original']):
                            col_type = schema_dict['column_types'][col_id] if col_id < len(schema_dict['column_types']) else 'text'
                            if 'number' in col_type.lower() or 'int' in col_type.lower() or 'float' in col_type.lower():
                                cv_link['num_date_match'][f"{q_id},{col_id}"] = 'NUMBER'
                    except ValueError:
                        pass
                
                schema_links = {
                    'sc_link': {'q_col_match': q_col_match, 'q_tab_match': q_tab_match}, 
                    'cv_link': cv_link
                }

            prompt_data = self.prompt.format(target, max_seq_len=2048, max_ans_len=200, scope_factor=100,
                                             cross_domain=True)

            prompt = prompt_data['prompt']

            res = self.llm.complete(prompt)  # Assume returns object with text
            res_text = res.text if hasattr(res, 'text') else str(res)

            sql = " ".join(res_text.replace("\n", " ").split())
            sql = process_duplication(sql)  # Define if not
            if not sql.upper().startswith('SELECT'):
                sql = 'SELECT ' + sql

            if self.is_save:
                instance_id = row.get("instance_id", item)
                save_path = Path(self.save_dir)
                save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
                save_path = save_path / f"{self.name}_{instance_id}.sql"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_dataset(sql, new_data_source=save_path)
                self.dataset.setitem(item, "pred_sql", str(save_path))

            return sql
        except Exception as e:
            logger.error(f"DAILSQL act error: {e}")
            return "SELECT"

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
