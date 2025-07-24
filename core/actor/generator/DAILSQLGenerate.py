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

import sql_metadata
from sql_metadata import Parser

from core.actor.generator.BaseGenerate import BaseGenerator
from core import logger

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
        pattern = re.compile(r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
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
        new_s = [s[i] for i in range(len(s)) if s[i] != "as" and (i == 0 or s[i-1] != "as")]
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
        if split_skeleton[i-2] == "order" and split_skeleton[i-1] == "by" and split_skeleton[i] != "_":
            split_skeleton[i] = "_"
    sql_skeleton = " ".join(split_skeleton)

    return sql_skeleton

def mask_question_with_schema_linking(data_jsons, mask_tag='<mask>', value_tag='<unk>'):
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
            if num_flag:    # TODO refine the date and time match
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
        sqls = get_sql_for_database(example['path_db'])
        prompt_info = self.template_info.format("\n\n".join(sqls))
        prompt_extra = self.get_extra_info(example['db_id'])
        prompt_question = self.template_question.format(example['question'])
        components = [prompt_info] if not prompt_extra else [prompt_info, prompt_extra]
        components.append(prompt_question)
        return "\n\n".join(components)

# Add all other prompt classes similarly from PromptReprTemplate.py

# Example Format Classes
class SqlExampleStyle(object):
    def get_example_prefix(self):
        return "/* Some SQL examples are provided based on similar problems: */\n"

    def format_example(self, example):
        return example['query']

# Add all from ExampleFormatTemplate.py

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

    # Full from PromptICLTemplate.py, adapting count_tokens

# Example Selector Classes
class BasicExampleSelector(object):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.train_json = data['train_json']  # Assume data provides this
        # ...

# Add all from ExampleSelectorTemplate.py
# For bert_model, note: Requires sentence_transformers. If not, skip or use simple version.

# Prompt Factory
def prompt_factory(repr_type, k_shot, example_format, selector_type):
    repr_cls = get_repr_cls(repr_type)
    if k_shot == 0:
        class PromptClass(repr_cls, BasicICLPrompt):
            name = f"{repr_type}_{k_shot}-SHOT"
            NUM_EXAMPLE = k_shot
    else:
        example_format_cls = get_example_format_cls(example_format)
        selector_cls = get_example_selector(selector_type)
        class PromptClass(selector_cls, example_format_cls, repr_cls, BasicICLPrompt):
            name = f"{repr_type}_{k_shot}-SHOT_{selector_type}_{example_format}-EXAMPLE"
            NUM_EXAMPLE = k_shot
    return PromptClass

# Add get_repr_cls, get_example_format_cls, get_example_selector from prompt_builder.py

def get_repr_cls(repr_type):
    if repr_type == REPR_TYPE.CODE_REPRESENTATION:
        return SQLPrompt
    # ... all cases

# Similarly for others

# Main Class
class DAILSQLGenerate(BaseGenerator):
    OUTPUT_NAME = "pred_sql"

    def __init__(self, llm, dataset, prompt_repr=REPR_TYPE.TEXT_REPRESENTATION, k_shot=0, example_type=EXAMPLE_TYPE.QA, selector_type=SELECTOR_TYPE.RANDOM, save_dir=None, is_save=False):
        self.llm = llm
        self.dataset = dataset  # Assume provides get_train_json, etc.
        self.prompt_repr = prompt_repr
        self.k_shot = k_shot
        self.example_type = example_type
        self.selector_type = selector_type
        self.save_dir = save_dir
        self.is_save = is_save

        self.prompt = prompt_factory(self.prompt_repr, self.k_shot, self.example_type, self.selector_type)(data= self.dataset, tokenizer="approx")

    def act(self, item, schema=None, schema_links=None, **kwargs):
        try:
            target = {
                'question': item.get('question', ''),
                'db_id': item.get('db_id', 'default_db'),
                'path_db': schema if isinstance(schema, str) else schema.get('path_db', 'default.db') if schema else 'default.db',
                'tables': get_tables(target['path_db']),
                'query': item.get('query', 'SELECT'),
                'column_names_original': schema.get('column_names_original', []) if schema else [],
                'table_names_original': schema.get('table_names_original', []) if schema else [],
                # Add other needed keys with defaults
            }

            if schema is None:
                logger.warning("Schema not provided, using dummy schema")
                # Define dummy

            # Compute schema_links if not provided
            if schema_links is None:
                question_toks = target['question'].split()
                columns = [c[1] for c in schema['column_names_original']]
                tables = schema['table_names_original']
                sc_link = compute_schema_linking(question_toks, columns, tables)
                cv_link = compute_cell_value_linking(question_toks, schema)  # Assume schema has connection
                sc_link['q_col_match'], sc_link['q_tab_match'], cv_link['cell_match'] = match_shift(sc_link['q_col_match'], sc_link['q_tab_match'], cv_link['cell_match'])
                schema_links = {'sc_link': sc_link, 'cv_link': cv_link}

            # Add to target if needed

            prompt_data = self.prompt.format(target, max_seq_len=2048, max_ans_len=200, scope_factor=100, cross_domain=True)

            prompt = prompt_data['prompt']

            res = self.llm.complete(prompt)  # Assume returns object with text
            res_text = res.text if hasattr(res, 'text') else str(res)

            sql = " ".join(res_text.replace("\n", " ").split())
            sql = process_duplication(sql)  # Define if not
            if not sql.upper().startswith('SELECT'):
                sql = 'SELECT ' + sql

            if self.is_save:
                instance_id = item.get('instance_id', 'unknown')
                save_path = Path(self.save_dir) / f"{self.NAME}_{instance_id}.sql"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    f.write(sql)
                # Update dataset if method exists

            return sql
        except Exception as e:
            logger.error(f"DAILSQL act error: {e}")
            return "SELECT"

# Define process_duplication from existing
def process_duplication(sql):
    return sql.strip().split("/*")[0]

# Add any missing definitions

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