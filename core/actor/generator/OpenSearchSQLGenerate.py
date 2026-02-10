import pandas as pd
import re
import os
from os import PathLike
import chardet
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from typing import Optional
from pathlib import Path
import time
import numpy as np
from loguru import logger

from core.actor.generator.BaseGenerate import BaseGenerator
from core.utils import load_dataset, save_dataset
from core.actor.generator.sql_debug import sql_debug_by_feedback
from core.db_connect import get_sql_exec_result

# Prompts from all_prompt.py and prompts.py
EXTRACT_PROMPT = """/* Some extract examples are provided based on similar problems: */
/* Answer the following: Please give the name of the course in which most numbers of the students got an A. Also, list the full name of the students who got an A in this course. most number of students got an A refers MAX(COUNT(student_id WHERE grade = 'A')); full name = f_name, l_name; got an A refers to grade = 'A'; */
#reason: The question requires display in order: "name of the course", "full name"."A" is filtering condition.
#columns: course.name, student.f_name, student.l_name, registration.grade, registration.student_id
#values: "A"

/* Answer the following:How much more votes for episode 1 than for episode 5? more votes refers to SUBTRACT(SUM(votes when episode = 1), SUM(votes when episode = 5)) */
#reason: The question requires display in order: "How much more vote". The definition of "more vote" is SUBTRACT(SUM(votes when episode = 1), SUM(votes when episode = 5)). 1, 5 are filtering conditions.
#columns: Episode.episode, Vote.votes
#values: "1", "5"

/* Answer the following: What is the average score of the movie "The Fall of Berlin" in 2019? Average score refers to Avg(rating_score); */
#reason: The question requires display in order: "average score". Average score is Avg(rating_score), "The Fall of Berlin",2019 are filtering conditions.
#columns: ratings.rating_score, ratings.rating_id, ratings.rating_timestamp_utc, movies.movie_title
#values: "The Fall of Berlin", "2019"

/* Answer the following: How many distinct orders were there in 2003 when the quantity ordered was less than 30? "year(orderDate) = '2003'; quantityOrdered < 30;" */
#reason:  The question requires display in order: "How many distinct orders"." in 2003", "less than 30" are filtering conditions.
#columns: orderdetails.orderNumber, orderdetails.quantityOrdered, orders.orderDate
#values: "30", "2003"

{fewshot}

/* Database schema */
{db_info}

Attention:
1. if the question have when\\where\\which, pay attention to pick table.column related to time, location and name in #columns
2. Please answer the question in the following format without any other content:
```
#reason: Analysis of which columns and values might be relevant to the question. Note that when dealing with questions about time, who, which, what, etc., you should keep column related to time, names, and locations in the #column.(format: the question query xxx, the related column include table.column, the values include values)
#columns: The top 10 columns relevant to the question( format: table.column_1, table.column_2 ...)
#values: Potential filter values that the question might query(format: "value1", "value2" ...)
```
/* Answer the following: {query} */
"""

NOUN_PROMPT = """Please extract all nouns and phrases from the following sentence, separating the results directly with a comma( format: "noun_1", "noun_2","phrases" ):
{raw_question}"""

SELECT_PROMPT = """现在我们定义一个问句的语法原子单元如下:
Q: 询问词: 如 calculate\\ Include\\ List\\ List out\\ List all\\ give\\ state\\ Name\\ In which\\ How many\\  which\\ what\\ who\\ when\\ provide\\ Tally\\ identify\\ Find\\ mention\\ write等
J: 判断词： 如 Do\\ Did\\ If\\ Is\\ Are等
I: 查询内容: 查询的主体内容, 如: name, ID, date, location, item, biggest city.
C: 条件句: 通过介词和连词引入的查询的要求或属性, 如大于、等于、排序、聚合等. 介词和连词有: of\\ have\\ with\\ that\\ by. 条件句的形式例子有: with condition\\ have condition\\ of attribute\\ that was condition


一个问题通过这些原子串联起来。常见的串联方式有
QIC(询问句): List the student with score more than 80: Q: 'List' I: 'the student' C: 'with score more than 80'
JC(判断句): State if Tom is a Cat? : J: 'State if C: is a Cat?'
C(条件句): For all people in Beijing
现在请你针对下面的问题, 把问题中的内容按照上述原子定义提取出来
问题如下: {question}

请按照下面的json格式进行回答:

```json
[{{"Type":"类型(QIC,JC,C)",
"Extract":{{//不存在的填null
    "Q":"询问词",
    "J":"判断词",
    "I":['查询内容a', '查询内容b'],//只有查询内容用and或alongside连接时,才分成多个实体填入List
    "C":["条件句a","属性b"]
}}}},
{{}}]
```"""

NEW_PROMPT = """{fewshot}

/* Database schema */
{db_info}
{key_col_des}

# Based on the database schema and the examples above, pay attention to the following:
1. For parts involving division that contain integer types, CAST them to REAL.
2. "#Values in Database" display part values from the database. Please ignore the unnecessary values.
3. Please refer to the examples above and answer in the following format without any other content:
```
#reason: Analyze how to generate SQL based on the question.(format: the question want to ..., so the SQL SELECT ... and ...)
#columns: All columns ultimately used in SQL(format: table.column_1, table.column_2)
#values: the filter in SQL (format: 'filter in question' refer to 'table.column <op> value'. e.g. 'name is not tom' refer to name <> 'tom', 'in 2007' refer to "strftime('%Y', Date) = '2007'")
#SELECT: SELECT content (format like: 'query in question' refer to table.column. The order of columns in the SELECT clause must be the same as the order in the question.)
#SQL-like: SQL-like statements ignoring Join conditions
#SQL: SQL
```

/* Answer the following: {question} */
{q_order}"""

TMP_PROMPT = """You are an SQL expert, and now I would like you to write SQL based on the question.
{fewshot}

/* Database schema */
{db_info}
{key_col_des}

/* Based on the database schema and the question, pay attention to the following */
1. For parts involving division that contain integer types, CAST them to REAL.
2. #values in db display part values from the database. Please ignore the unnecessary values.

Please rewrite the question to SQL-like query in the format: "Show #SELECT (table.column), WHERE condition are xxx (refer to #values), Group by/Order By (refer to columns). Here are 3 example: 

#SQL-like: Show top 5 cards.id, where condition is cards.spend>100, order by cards.spend. 
#SQL: SELECT id FROM cards WHERE spend > 100 ORDER BY spend LIMIT 5

#SQL-like: Show Count(PaperAuthor.Name), Where condition is Paper.Year = 0
#SQL: SELECT COUNT(T2.Name) FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Year = 0

#SQL-like: Show Author.Name, Where condition is Author.Affiliation = 'University of Oxford', Group by Author.Name Order By Author.spent
#SQL: SELECT Name FROM Author WHERE Affiliation = 'University of Oxford' Group By Name ORDER BY spent ASC

/* Answer the following: {question} */
Please answer the question in the following format without any other content:
```
#reason: Analyze how to generate SQL based on the question.(format: the question want to ..., so the SQL SELECT ... and ...)
#columns: All columns ultimately used in SQL(format: table.column_1, table.column_2)
#values: the filter in SQL (format: 'filter in question' refer to table.column: value. e.g. 'name is not tom' refer to name <> "tom", 2007 refer to strftime('%Y', Date) = '2007')
#SELECT: SELECT content (display in the order asked by the questions, do not display content not specified by the questions).
#SQL-like: SQL-like statements ignoring Join conditions
#SQL: SQL
```"""

SOFT_PROMPT = """Your task is to perform a simple evaluation of the SQL.

The database system is SQLite. The SQL you need to evaluation is:
#question: {question}
#SQL: {SQL}

Answer in the following format: 
{{
"Judgment": true/false,
"SQL":If SQL is wrong, please correct SQL directly. else answer ""
}}"""

CORRECT_PROMPT = """You are an expert in SQL. Here are some examples of fix SQL
{fewshot}

/* Database schema is as follows: */
{db_info}
{key_col_des}

/* Now Plesease fix the following error SQL */
#question: {q}
#Error SQL: {result_info}
{advice}

Please answer according to the format below and do not output any other content.:
```
#reason: Analysis of How to fix the error
#SQL: right SQL
```"""

VOTE_PROMPT = """现在有问题如下:
#question: {question}
对应这个问题有如下几个SQL,请你从中选择最接近问题要求的SQL:
{sql}

请在上面的几个SQL中选择最符合题目要求的SQL, 不要回复其他内容:
#SQL:"""


def find_foreign_keys_MYSQL_like(DATASET_JSON, db_name):
    if DATASET_JSON is None:
        return "", set()
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names', 'table_names'], axis=1)
    f_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        foreign_keys = row['foreign_keys']
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([
                row['db_id'], tables[first_index], tables[second_index],
                first_column, second_column
            ])
    spider_foreign = pd.DataFrame(f_keys,
                                  columns=[
                                      'Database name', 'First Table Name',
                                      'Second Table Name',
                                      'First Table Foreign Key',
                                      'Second Table Foreign Key'
                                  ])

    df = spider_foreign[spider_foreign['Database name'] == db_name]
    output = []
    col_set = set()
    for index, row in df.iterrows():
        output.append(row['First Table Name'] + '.' +
                      row['First Table Foreign Key'] + " = " +
                      row['Second Table Name'] + '.' +
                      row['Second Table Foreign Key'])
        col_set.add(row['First Table Name'] + '.' +
                    row['First Table Foreign Key'])
        col_set.add(row['Second Table Name'] + '.' +
                    row['Second Table Foreign Key'])
    output = ", ".join(output)
    return output, col_set


def quote_field(field_name):
    if re.search(r'\W', field_name):
        return f"`{field_name}`"
    else:
        return field_name


class DB_AGENT:
    def __init__(self, chat_model) -> None:
        self.chat_model = chat_model

    def get_complete_table_info(self, db_type: str, db_path: str, table_name: str, table_df: pd.DataFrame,
                                credential: Optional[str] = None, db_id: Optional[str] = None):
        """Get complete table information using get_sql_exec_result"""
        try:
            # Build column info query per database type
            # Parse qualified names
            parts = table_name.split('.')
            table_only = parts[-1]
            schema_only = parts[-2] if len(parts) >= 2 else None

            if db_type == "sqlite":
                pragma_query = f"PRAGMA table_info(`{table_name}`)"
            elif db_type == "snowflake":
                # Shape output to match SQLite PRAGMA columns order: (cid,name,type,notnull,dflt_value,pk)
                schema_filter = f" AND UPPER(TABLE_SCHEMA) = '{schema_only.upper()}'" if schema_only else ""
                pragma_query = (
                    "SELECT 0 AS cid, COLUMN_NAME AS name, DATA_TYPE AS type, "
                    "CASE WHEN IS_NULLABLE='YES' THEN 0 ELSE 1 END AS notnull, "
                    "COLUMN_DEFAULT AS dflt_value, 0 AS pk "
                    "FROM INFORMATION_SCHEMA.COLUMNS "
                    f"WHERE UPPER(TABLE_NAME) = '{table_only.upper()}'" + schema_filter + " "
                    "ORDER BY ORDINAL_POSITION"
                )
            elif db_type == "big_query":
                # Use db_id as fully qualified dataset (e.g., project.dataset)
                pragma_query = (
                    "SELECT 0 AS cid, column_name AS name, data_type AS type, "
                    "CASE WHEN is_nullable = 'YES' THEN 0 ELSE 1 END AS notnull, "
                    "NULL AS dflt_value, 0 AS pk "
                    f"FROM `{db_id}.INFORMATION_SCHEMA.COLUMNS` "
                    f"WHERE table_name = '{table_only}' "
                    "ORDER BY ordinal_position"
                )
            else:
                # Fallback to SQLite-style for unknown types
                pragma_query = f"PRAGMA table_info(`{table_name}`)"

            # Prepare arguments for get_sql_exec_result
            exec_args = {"sql_query": pragma_query, "db_path": db_path, "db_id": db_id, "credential_path": credential}

            columns_info_df, err = get_sql_exec_result(db_type, **exec_args)
            if err:
                raise ValueError(f"Error fetching table info: {err}")

            columns_info = [tuple(row) for row in columns_info_df.values]  # Convert DF to list of tuples

            # Read table data
            if db_type == "sqlite":
                data_query = f"SELECT * FROM `{table_name}`"
            elif db_type == "snowflake":
                qual = f"{schema_only}.{table_only}" if schema_only else table_only
                data_query = f"SELECT * FROM {qual}"
            elif db_type == "big_query":
                data_query = f"SELECT * FROM `{db_id}.{table_only}`"
            else:
                data_query = f"SELECT * FROM `{table_name}`"
            exec_args["sql_query"] = data_query

            df, err = get_sql_exec_result(db_type, **exec_args)
            if err:
                raise ValueError(f"Error reading table data: {err}")

            contains_null = {column: df[column].isnull().any() for column in df.columns}
            contains_duplicates = {column: df[column].duplicated().any() for column in df.columns}

            dic = {}
            for _, row in table_df.iterrows():
                try:
                    col_description, val_description = "", ""
                    col = row.iloc[0].strip()
                    if pd.notna(row.iloc[2]):
                        col_description = re.sub(r'\s+', ' ', str(row.iloc[2]))
                    if col_description.strip() == col or col_description.strip() == "":
                        col_description = ''
                    if pd.notna(row.iloc[4]):
                        val_description = re.sub(r'\s+', ' ', str(row.iloc[4]))
                    if val_description.strip() == "" or val_description.strip() == col or val_description == col_description:
                        val_description = ""
                    col_description = col_description[:200]
                    val_description = val_description[:200]
                    dic[col] = col_description, val_description
                except Exception as e:
                    dic[col] = "", ""

            # For the LIMIT 1 query
            if db_type == "sqlite":
                limit_query = f"SELECT * FROM `{table_name}` LIMIT 1"
            elif db_type == "snowflake":
                qual = f"{schema_only}.{table_only}" if schema_only else table_only
                limit_query = f"SELECT * FROM {qual} LIMIT 1"
            elif db_type == "big_query":
                limit_query = f"SELECT * FROM `{db_id}.{table_only}` LIMIT 1"
            else:
                limit_query = f"SELECT * FROM `{table_name}` LIMIT 1"
            exec_args["sql_query"] = limit_query

            limit_df, err = get_sql_exec_result(db_type, **exec_args)
            if err:
                raise ValueError(f"Error fetching sample row: {err}")
            row = list(limit_df.values[0]) if not limit_df.empty else []  # Adjust to DF

            # Process values for each column
            for i, col in enumerate(df.columns):
                try:
                    df_tmp = df[col].dropna().drop_duplicates()
                    if len(df_tmp) >= 3:
                        vals = df_tmp.sample(3).values
                    else:
                        vals = df_tmp.values
                    val_p = []
                    for val in vals:
                        try:
                            val_p.append(int(val))
                        except:
                            val_p.append(val)
                    if len(vals) == 0:
                        raise ValueError
                    if i < len(row):
                        row[i] = val_p
                except:
                    pass

            # Display table name without schema to keep downstream expectations (table.column)
            display_table = table_only
            schema_str = f"## Table {display_table}:\n"
            columns = {}
            for column, val in zip(columns_info, row):
                schema_str_single = ""
                column_name, column_type, not_null, default_value, pk = column[1:6]
                tmp_col = column_name.strip()
                column_name = quote_field(column_name)

                col_des, val_des = dic.get(tmp_col, ["", ""])
                if col_des != "":
                    schema_str_single += f" The column is {col_des}. "
                if val_des != "":
                    schema_str_single += f" The values' format are {val_des}. "

                schema_str_single += f"The type is {column_type}, "
                if contains_null[tmp_col]:
                    schema_str_single += f"Which inlude Null"
                else:
                    schema_str_single += f"Which does not inlude Null"

                if contains_duplicates[tmp_col]:
                    schema_str_single += " and is Non-Unique. "
                else:
                    schema_str_single += " and is Unique. "

                include_null = f"{'Include Null' if contains_null[tmp_col] else 'Non-Null'}"
                unique = f"{'Non-Unique' if contains_duplicates[tmp_col] else 'Unique'}"
                if len(str(val)) > 360:
                    val = "<Long text>"
                    schema_str_single += f"Values format: <Long text>"
                elif type(val) is not list or len(val) < 3:
                    schema_str_single += f"Value of this column must in: {val}"
                else:
                    schema_str_single += f"Values format like: {val}"
                schema_str += f"{column_name}: {schema_str_single}\n"
                columns[f"{display_table}.{column_name}"] = (schema_str_single, col_des, val_des, column_type,
                                                          include_null,
                                                          unique, str(val))
            return schema_str, columns
        except Exception as e:
            logger.error(f"Error in get_complete_table_info: {e}")
            raise

    def get_db_des(self, db_type: str, db_path: str, db_dir: Optional[str], model,
                   credential: Optional[str] = None, db_id: Optional[str] = None):
        """Get database description using get_sql_exec_result"""
        try:
            # Prepare query based on database type
            if db_type == "sqlite":
                tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
            elif db_type == "snowflake":
                # If db_id encodes database.schema, filter SHOW TABLES accordingly
                if db_id and '.' in db_id:
                    db_name, schema_name = db_id.split('.', 1)
                    tables_query = f"SHOW TABLES IN {db_name}.{schema_name}"
                else:
                    tables_query = "SHOW TABLES"
            elif db_type == "big_query":
                tables_query = f"SELECT table_name FROM `{db_id}.INFORMATION_SCHEMA.TABLES`"
            else:
                tables_query = "SELECT name FROM sqlite_master WHERE type='table';"

            # Prepare arguments for get_sql_exec_result
            exec_args = {"sql_query": tables_query, "db_path": db_path, "db_id": db_id, "credential_path": credential}

            tables_df, err = get_sql_exec_result(db_type, **exec_args)
            if err:
                raise ValueError(f"Error fetching tables: {err}")

            # Normalize table list across engines
            if db_type == "sqlite":
                tables = [(row[0],) for row in tables_df.values]
            elif db_type == "snowflake":
                # SHOW TABLES returns columns like: created_on, name, ... Ensure we pick the name column
                # Try a few common positions/names to be robust
                lower_cols = [c.lower() for c in tables_df.columns]
                if 'name' in lower_cols:
                    name_idx = lower_cols.index('name')
                    # Prefer schema-qualified name if schema column exists
                    if 'schema_name' in lower_cols:
                        schema_idx = lower_cols.index('schema_name')
                        tables = [(f"{row[schema_idx]}.{row[name_idx]}",) for row in tables_df.values]
                    else:
                        tables = [(row[name_idx],) for row in tables_df.values]
                elif 'table_name' in lower_cols:
                    name_idx = lower_cols.index('table_name')
                    if 'schema_name' in lower_cols:
                        schema_idx = lower_cols.index('schema_name')
                        tables = [(f"{row[schema_idx]}.{row[name_idx]}",) for row in tables_df.values]
                    else:
                        tables = [(row[name_idx],) for row in tables_df.values]
                else:
                    # Fallback to first string-like column
                    tables = []
                    for row in tables_df.values:
                        picked = None
                        for i, v in enumerate(row):
                            if isinstance(v, str) and v:
                                picked = v
                                break
                        if picked is not None:
                            tables.append((picked,))
            elif db_type == "big_query":
                tables = [(row[0],) for row in tables_df.values]
            else:
                tables = [(row[0],) for row in tables_df.values]

            db_info = []
            db_col = dict()
            table_dir = os.path.join(db_dir, 'database_description') if db_dir else None
            if table_dir and os.path.exists(table_dir):
                file_list = os.listdir(table_dir)
                files_emb = model.encode(file_list, show_progress_bar=False)
            else:
                file_list = []
                files_emb = []

            for table in tables:
                if table[0] == 'sqlite_sequence':
                    continue
                if file_list:
                    files_sim = (files_emb @ model.encode(table[0] + '.csv', show_progress_bar=False).T)
                    if max(files_sim) > 0.9:
                        file = os.path.join(table_dir, file_list[files_sim.argmax()])
                    else:
                        file = os.path.join(table_dir, table[0] + '.csv')
                    try:
                        with open(file, 'rb') as f:
                            result = chardet.detect(f.read())
                        table_df = pd.read_csv(file, encoding=result['encoding'])
                    except Exception as e:
                        table_df = pd.DataFrame()
                else:
                    table_df = pd.DataFrame()

                table_info, columns = self.get_complete_table_info(db_type, db_path, table[0], table_df,
                                                                   credential, db_id)
                db_info.append(table_info)
                db_col.update(columns)

            db_info = "\n".join(db_info)
            return db_info, db_col
        except Exception as e:
            logger.error(f"Error in get_db_des: {e}")
            raise

    def db_conclusion(self, db_info):
        prompt = f"""/* Here is a examples about describe database */
#Forigen keys: 
Airlines.ORIGIN = Airports.Code, Airlines.DEST = Airports.Code, Airlines.OP_CARRIER_AIRLINE_ID = Air Carriers.Code
#Database Description: The database encompasses information related to flights, including airlines, airports, and flight operations.
#Tables Descriptions:
Air Carriers: Codes and descriptive information about airlines
Airports: IATA codes and descriptions of airports
Airlines: Detailed information about flights 

/* Here is a examples about describe database */
#Forigen keys:
data.ID = price.ID, production.ID = price.ID, production.ID = data.ID, production.country = country.origin
#Database Description: The database contains information related to cars, including country, price, specifications, Production
#Tables Descriptions:
Country: Names of the countries where the cars originate from.
Price: Price of the car in USD.
Data: Information about the car's specifications
Production: Information about car's production.

/* Describe the following database */
{db_info}
Please conclude the database in the following format:
#Database Description:
#Tables Descriptions:
"""
        return prompt

    def get_allinfo(self, db_json_dir: Optional[str], db: str, db_path: str, db_dir: Optional[str],
                    tables_info_dir: Optional[str], model, db_type: str = 'sqlite',
                    credential: Optional[str] = None, db_id: Optional[str] = None):
        """Get all database information with proper database connection handling"""
        try:
            db_info, db_col = self.get_db_des(db_type, db_path, db_dir, model, credential, db_id)
            foreign_keys, foreign_set = find_foreign_keys_MYSQL_like(tables_info_dir, db) if tables_info_dir else ("",
                                                                                                                   set())

            # Adapt database system name based on db_type
            db_system_name = {
                'sqlite': 'SQLite',
                'snowflake': 'Snowflake',
                'big_query': 'BigQuery'
            }.get(db_type, 'SQLite')

            all_info = f"Database Management System: {db_system_name}\n#Database name: {db}\n{db_info}\n#Foreign keys:\n{foreign_keys}\n"
            prompt = self.db_conclusion(all_info)
            db_all = self.chat_model.get_ans(prompt)
            all_info = f"{all_info}\n{db_all}\n"
            return all_info, db_col, foreign_set
        except Exception as e:
            logger.error(f"Error in get_allinfo: {e}")
            raise


def parse_des(pre_col_values, nouns, debug):
    pre_col_values = pre_col_values.split("/*")[0].strip()
    if debug:
        print(pre_col_values)
    col, values = pre_col_values.split('#values:')
    _, col = col.split("#columns:")
    col = strip_char(col)
    values = strip_char(values)

    if values == '':
        values = []
    else:
        values = re.findall(r"([\"'])(.*?)\1", values)
    nouns_all = re.findall(r"([\"'])(.*?)\1", nouns)
    values_noun = set(values).union(set(nouns_all))
    values_noun = [x[1] for x in values_noun]
    return values_noun, col


def strip_char(s):
    return s.strip('\n {}[]')


class ColumnRetriever:
    def __init__(self, bert_model, tables_info_dir):
        self.bert_model = bert_model
        self.tables_info_dir = tables_info_dir

    def get_col_retrieve(self, question, db, db_keys_col):
        if isinstance(self.tables_info_dir, (str, PathLike)) and Path(self.tables_info_dir).exists():
            schema = load_dataset(self.tables_info_dir)
        else:
            schema = {}
        db_schema = schema.get(db, {})
        schema_emb = self.bert_model.encode(db_schema, show_progress_bar=False)
        question_emb = self.bert_model.encode(question, show_progress_bar=False)
        schema_sim = schema_emb @ question_emb.T
        cols = set()
        for x in schema_sim.argsort()[::-1][:15]:
            if schema_sim[x] > 0.5:
                cols.add(db_schema[x])
        cols = set(cols).intersection(db_keys_col)
        return cols


class ColumnUpdater:
    def __init__(self, db_col):
        self.db_col = db_col

    def col_pre_update(self, origin_col, col_retrieve, foreign_set):
        cols = set(origin_col.split(', ')) | set(col_retrieve)
        cols = cols.intersection(self.db_col.keys())
        for x in foreign_set:
            if x in cols:
                cols.remove(x)
        return list(cols)

    def col_suffix(self, cols_select):
        column = ""
        for x in cols_select:
            des = self.db_col[x][0]
            column += f"{x}: {des}\n"
        return column


class DES_new:
    def __init__(self, bert_model, DB_emb, col_values):
        self.bert_model = bert_model
        self.DB_emb = DB_emb
        self.col_values = col_values

    def get_key_col_des(self, cols, values, debug=False, topk=10, shold=0.65):
        cols_select = []
        L_values = []
        if debug:
            print(cols)
        if cols:
            col_emb = self.bert_model.encode(cols, show_progress_bar=False)
            col_sim = col_emb @ self.DB_emb.T
        else:
            col_sim = np.empty((0, self.DB_emb.shape[1]))
        if values:
            val_emb = self.bert_model.encode(values, show_progress_bar=False)
            val_sim = val_emb @ self.DB_emb.T
        else:
            val_sim = np.empty((0, self.DB_emb.shape[1]))
        for i in range(col_sim.shape[0]):
            if col_sim.shape[1] == 0:
                continue
            indices = np.argsort(col_sim[i])[::-1][:topk]
            for x in indices:
                if col_sim[i, x] > shold:
                    cols_select.append(self.col_values[x])
        cols_select = list(set(cols_select))
        for i in range(val_sim.shape[0]):
            if val_sim.shape[1] == 0:
                continue
            indices = np.argsort(val_sim[i])[::-1][:topk * 3]
            for x in indices:
                if val_sim[i, x] > shold:
                    L_values.append((self.col_values[x], values[i]))
        L_values = list(set(L_values))
        return cols_select, L_values


def query_order(question, chat_model, select_prompt, temperature):
    select_prompt = select_prompt.format(question=question)
    ans = chat_model.get_ans(select_prompt, temperature=temperature)
    ans = re.sub("```json|```", "", ans)
    select_json = json.loads(ans)
    res, judge = json_ext(select_json)
    return res


def json_ext(jsonf):
    ans = []
    judge = False
    for x in jsonf:
        if x["Type"] == "QIC":
            Q = x["Extract"]["Q"].lower()
            if Q in ["how many", "how much", "which", "how often"]:
                for item in x["Extract"]["I"]:
                    ans.append(x["Extract"]["Q"] + " " + item)
            elif Q in ["when", "who", "where"]:
                ans.append(x["Extract"]["Q"])
            else:
                ans.extend(x["Extract"]["I"])
        elif x["Type"] == "JC":
            ans.append(x["Extract"]["J"])
            judge = True
    return ans, judge


# Ported from check_and_correct.py
def sql_raw_parse(sql, return_question):
    sql = sql.split('/*')[0].strip().replace('```sql', '').replace('```', '')
    sql = re.sub("```.*?", '', sql)
    rwq = None
    if return_question:
        rwq, sql = sql.split('#SQL:')
    else:
        sql = sql.split('#SQL:')[-1]
    if sql.startswith("\"") or sql.startswith("\'"):
        sql = sql[1:-1]
    sql = re.sub('\s+', ' ', sql).strip()
    return sql, rwq


def retable(sql):
    table_as = re.findall(' ([^ ]*) +AS +([^ ]*)', sql)
    for x in table_as:
        sql = sql.replace(f"{x[1]}.", f"{x[0]}.")
    return sql


def max_fun_check(sql_retable):
    fun_amb = re.findall("= *\( *SELECT *(MAX|MIN)\((.*?)\) +FROM +(\w+)", sql_retable)
    order_amb = set(re.findall("= (\(SELECT .* LIMIT \d\))", sql_retable))
    select_amb = set(re.findall("^SELECT[^\(\)]*? ((MIN|MAX)\([^\)]*?\)).*?LIMIT 1", sql_retable))
    return fun_amb, order_amb, select_amb


def foreign_pick(sql):
    matchs = re.findall("ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+) ", sql)
    ma_all = [x for y in matchs for x in y]
    return set(ma_all)


def column_pick(sql, db_col, foreign_set):
    matchs = foreign_pick(sql)
    cols = set()
    col_table = {}
    ans = set()
    sql_select = set(re.findall("SELECT (.*?) FROM ", sql))
    for x in db_col:
        if sql.find(x) != -1:
            cols.add(x)
        table, col = x.split('.')
        col_table.setdefault(col, [])
        col_table[col].append(table)
    for col in cols:
        table, col_name = col.split('.')
        flag = True
        for x in sql_select:
            if x.find(col) != -1:
                flag = False
                break
        if flag and (col in foreign_set or col in matchs):
            continue
        if col_table.get(col_name):
            Ambiguity = []
            for t in col_table[col_name]:
                tbc = f"{t}.{col_name}"
                if tbc != col:
                    Ambiguity.append(tbc)
            if len(Ambiguity):
                amb_des = col + ": " + ", ".join(Ambiguity)
                ans.add(amb_des)
    return sorted(list(ans))


def values_pick(vals, sql):
    val_dic = {}
    ans = set()
    try:
        for val in vals:
            val_dic.setdefault(val[1], [])
            val_dic[val[1]].append(val[0])
        for val in val_dic:
            in_sql, not_sql = [], []
            if sql.find(val):
                for x in val_dic[val]:
                    if sql.find(x) != -1:
                        in_sql.append(f"{x} = '{val}'")
                    else:
                        not_sql.append(f"{x} = '{val}'")
            if len(in_sql) and len(not_sql):
                ans.add(f"{', '.join(in_sql)}: {', '.join(not_sql)}")
        return sorted(list(ans))
    except:
        return []


def func_find(sql):
    fun_amb = re.findall("\( *SELECT *(MAX|MIN)\((.*?)\) +FROM +(\w+)", sql)
    fun_str = []
    for fun in fun_amb:
        fuc = fun[0]
        col = fun[1]
        table = fun[2]
        if fuc == "MAX":
            order = "DESC"
        else:
            order = "ASC"
        str_fun = f"(SELECT {fuc}({col}) FROM {table}): ORDER BY {table}.{col} {order} LIMIT 1"
        fun_str.append(str_fun)
    return "\n".join(fun_str)


t1_tabe_value = re.compile("(\w+\\.[\w]+) =\\s*'([^']+(?:''[^']*)*)'")
t2_tab_val = re.compile("(\\w+\\.`[^`]*?`) =\\s*'([^']+(?:''[^']*)*)'")


def join_exec(db_type: str, db_path: str, bx: str, al: str, question: str, SQL: str, chat_model,
              credential: Optional[str] = None, db_id: Optional[str] = None):
    """Execute join correction with proper database connection handling"""
    flag = False
    try:
        # Prepare arguments for get_sql_exec_result
        def exec_sql(sql_query: str):
            exec_args = {"sql_query": sql_query, "db_path": db_path, "db_id": db_id, "credential_path": credential}

            return get_sql_exec_result(db_type, **exec_args)

        if bx.startswith("IN"):
            b = bx[2:].strip(" ()").split(',')
            for x in b:
                sql_t = SQL.replace(bx, f"= {x}")
                df, err = exec_sql(sql_t)
                if err:
                    continue
                if len(df):
                    SQL = sql_t
                    flag = True
                    break
        elif al.find("OR") != -1:
            a = al.split("OR")
            for x in a:
                sql_t = SQL.replace(al, x.strip())
                df, err = exec_sql(sql_t)
                if err:
                    continue
                if len(df):
                    SQL = sql_t
                    flag = True
                    break
    except Exception as e:
        logger.error(f"Error in join_exec: {e}")

    return SQL, flag


def gpt_join_corect(SQL, question, chat_model):
    prompt = f"""下面的question对应的SQL错误的使用了JOIN函数,使用了JOIN table AS T ON Ta.column1 = Tb.column2 OR Ta.column1 = Tb.column3或JOIN table AS T ON Ta.column1 IN的JOIN方式,请你只保留 OR之中优先级最高的一组 Ta.column = Tb.column即可.

question:{question}
SQL: {SQL}

请直接给出新的SQL, 不要回复任何其他内容:
#SQL:"""
    SQL = chat_model.get_ans(prompt, 0.0).split("SQL:")[-1]
    return SQL


def select_check(SQL, db_col, chat_model, question):
    select = re.findall("^SELECT.*?\\|\\| ' ' \\|\\| .*?FROM", SQL)
    if select:
        SQL = SQL.replace("|| ' ' ||", ', ')

    select_amb = re.findall("^SELECT.*? (\\w+\\.\\*).*?FROM", SQL)
    if select_amb:
        prompt = f"""数据库存在以下字段:
{db_col}
现有问题为 {question}
SQL:{SQL}
我们规定视这种不明确的查询为对应的id
现在请你把上面SQL的{select_amb[0]}改为对应的id,请你直接给出SQL, 不要回复任何其他内容:
#SQL:"""
        SQL = chat_model.get_ans(prompt, 0.0).split("SQL:")[-1]
    return SQL


class soft_check:

    def __init__(self, bert_model, chat_model, soft_prompt, correct_dic, correct_prompt, vote_prompt) -> None:
        self.bert_model = bert_model
        self.chat_model = chat_model
        self.soft_prompt = soft_prompt
        self.correct_dic = correct_dic
        self.correct_prompt = correct_prompt
        self.vote_prompt = vote_prompt

    def vote_chose(self, SQLs, question):
        all_sql = '\n\n'.join(SQLs)
        prompt = self.vote_prompt.format(question=question, sql=all_sql)
        SQL_vote_response = self.chat_model.get_ans(prompt, 0.0)
        SQL_vote, _ = sql_raw_parse(SQL_vote_response, False)
        return SQL_vote

    def soft_correct(self, SQL, question, new_prompt, hint=""):
        soft_p = self.soft_prompt.format(SQL=SQL, question=question, hint=hint)
        soft_SQL = self.chat_model.get_ans(soft_p, 0.0)
        soft_SQL = re.sub("```\\w*", "", soft_SQL)
        soft_json = json.loads(soft_SQL)
        if (soft_json["Judgment"] == False or soft_json["Judgment"] == 'False') and soft_json["SQL"] != "":
            SQL = soft_json["SQL"]
            SQL = re.sub('\\s+', ' ', SQL).strip()
        elif (soft_json["Judgment"] == False or soft_json["Judgment"] == 'False'):
            SQL_response = self.chat_model.get_ans(new_prompt, 1.0)
            SQL, _ = sql_raw_parse(SQL_response, False)
        return SQL, soft_json["Judgment"]

    def double_check(self, new_prompt, values: list, values_final, SQL: str, question: str, new_db_info: str,
                     db_col: list, db: str, hint="") -> str:
        SQL = re.sub("(COUNT)(\\([^\\(\\)]*? THEN 1 ELSE 0.*?\\))", r"SUM\2", SQL)
        sql_retable = retable(SQL)
        SQL = self.values_check(sql_retable, values, values_final, SQL, question, new_db_info, db_col, hint)
        SQL = self.JOIN_error(SQL, question, db)
        SQL = self.func_check(sql_retable, SQL, question)
        SQL = self.func_check2(question, SQL)
        SQL = self.time_check(SQL)
        SQL = self.is_not_null(SQL)
        SQL = select_check(SQL, db_col, self.chat_model, question)
        return SQL, True

    def double_check_style_align(self, SQL: str, question: str, db_col: list, sql_retable: str) -> str:
        SQL = self.func_check(sql_retable, SQL, question)
        SQL = self.is_not_null(SQL)
        SQL = select_check(SQL, db_col, self.chat_model, question)
        return SQL, True

    def double_check_function_align(self, SQL: str, question: str, db: str, credential=None, db_id=None) -> str:
        SQL = self.JOIN_error(SQL, question, db, credential, db_id)
        SQL = self.func_check2(question, SQL)
        SQL = self.time_check(SQL)
        return SQL, True

    def double_check_agent_align(self, sql_retable: str, values: list, values_final, SQL: str, question: str,
                                 new_db_info: str, db_col: list, hint="") -> str:
        SQL = self.values_check(sql_retable, values, values_final, SQL, question, new_db_info, db_col, hint)
        return SQL, True

    def JOIN_error(self, SQL, question, db, credential=None, db_id=None):
        join_mutil = re.findall(
            "JOIN\\s+\\w+(\\s+AS\\s+\\w+){0,1}\\s+ON(\\s+\\w+\\.\\w+\\s*(=\\s*\\w+\\.\\w+(?:\\s+OR\\s+\\w+\\.\\w+\\s*=\\s*\\w+\\.\\w+)+|IN\\s+\\(.*?\\)))",
            SQL)
        flag = False
        if join_mutil:
            _, al, bx = join_mutil[0]
            try:
                # Determine db_type from db path or use default
                db_type = 'sqlite'  # Default for backward compatibility
                SQL, flag = join_exec(db_type, db, bx, al, question, SQL, self.chat_model, credential, db_id)
            except Exception as e:
                logger.error(f"Error in JOIN_error: {e}")
        if not flag and join_mutil:
            SQL = gpt_join_corect(SQL, question, self.chat_model)
        return SQL

    def is_not_null(self, SQL):
        SQL = SQL.strip()
        inn = re.findall("ORDER BY .*?(?<!DESC )LIMIT +\\d+;{0,1}", SQL)
        if not inn:
            return SQL
        for x in inn:
            if re.findall("SUM\\(|COUNT\\(", x):
                return SQL
        prompt = f"""请你为下面SQL ORDER BY的条件加上WHERE IS NOT NULL限制:
SQL:{SQL}

请直接给出新的SQL, 不要回复任何其他内容:
#SQL:"""
        SQL = self.chat_model.get_ans(prompt, 0.0).split("SQL:")[-1]
        return SQL

    def time_check(self, sql):
        time_error_fix = re.sub("(strftime *\\([^(]*?\\) *[>=<]+ *)(\\d{{4,}})", r"\1'\2'", sql)
        return time_error_fix

    def func_check2(self, question, SQL):
        res = re.search("ORDER BY ((MIN|MAX)\\((.*?)\\)).*? LIMIT \\d+", SQL)
        if res:
            prompt = f"""对于下面的qustion和SQL:
#question: {question}
#SQL: {SQL}

ERROR: {res.group()} 是一种不正确的用法, 请对SQL进行修正, 注意如果SQL中存在GROUP BY, 请判断{res.groups()[0]}的内容是否需要使用 SUM({res.groups()[2]})

请直接给出新的SQL, 不要回复任何其他内容:"""
            SQL_response = self.chat_model.get_ans(prompt, 0.1)
            SQL, _ = sql_raw_parse(SQL_response, False)
        return SQL

    def func_check(self, sql_retable, sql, question):
        fun_amb, order_amb, select_amb = max_fun_check(sql_retable)
        if not fun_amb and not order_amb and not select_amb:
            return sql
        fun_str = []
        origin_f = []
        for fun in fun_amb:
            fuc = fun[0]
            col = fun[1]
            table = fun[2]
            if fuc == "MAX":
                order = "DESC"
            else:
                order = "ASC"
            str_fun = f"WHERE {col} = (SELECT {fuc}({col}) FROM {table}): 请用 ORDER BY {table}.{col} {order} LIMIT 1 代替嵌套SQL"
            origin_f.append(f"WHERE {col} = (SELECT {fuc}({col}) FROM {table})")
            fun_str.append(str_fun)
        for fun in order_amb:
            origin_f.append(fun)
            fun_str.append(f"{fun}: 使用JOIN 形式代替嵌套")
        for fun in select_amb:
            origin_f.append(fun[0])
            fun_str.append(f"{fun[0]}: {fun[1]} function 函数 冗余,请更改")
        func_amb = "\n".join(fun_str)
        prompt = f"""对于下面的问题和SQL, 请根据ERROR和#change ambuity修改:
#question: {question}
#SQL: {sql}
ERROR:{",".join(origin_f)} 不符合要求, 请使用 JOIN ORDER BY LIMIT 形式代替
#change ambuity: {func_amb}

请直接给出新的SQL, 不要回复任何其他内容:"""
        sql_response = self.chat_model.get_ans(prompt, 0.0)
        sql, _ = sql_raw_parse(sql_response, False)
        return sql

    def values_check(self, sql_retable, values, values_final, sql, question, new_db_info, db_col, hint=""):
        dic_v = {}
        dic_c = {}
        l_v = list(set([x[1] for x in values]))
        tables = "( " + " | ".join(set([x.split(".")[0] for x in db_col])) + " )"
        for x in values:
            dic_v.setdefault(x[1], [])
            dic_v[x[1]].append(x[0])
            dic_c.setdefault(x[0], [])
            dic_c[x[0]].append(x[1])
        value_sql = re.findall(t1_tabe_value, sql_retable)
        value_sql.extend(re.findall(t2_tab_val, sql_retable))
        tabs = set(re.findall(tables, sql))
        if len(tabs) == 1:
            val_single = re.findall("[ \\(]([\\w]+) =\\s*'([^']+(?:''[^']*)*)'", sql)
            val_single.extend(re.findall("[ \\(]([\\w]+) =\\s*'([^']+(?:''[^']*)*)'", sql))
            val_single = set(val_single)
            tab = tabs.pop()[1:-1]
            for x in val_single:
                value_sql.append((f"{tab}.{x[0]}", x[1]))
        badval_l = []
        change_val = []
        value_sql = set(value_sql)
        for tab_val in value_sql:
            tab, val = tab_val
            if len(re.findall("\\d", val)) / len(val) > 0.6:
                continue
            tmp_col = dic_v.get(val)
            if not tmp_col and len(l_v):
                val_close = self.bert_model.encode(val, show_progress_bar=False) @ self.bert_model.encode(l_v,
                                                                                                          show_progress_bar=False).T
                if val_close.max() > 0.95:
                    val_new = l_v[val_close.argmax()]
                    sql = sql.replace(f"'{val}'", f"'{val_new}'")
                    val = val_new
            tmp_col = dic_v.get(val)
            tmp_val = dic_c.get(tab, {})
            if tmp_col and tab not in tmp_col:
                lt = [f"{x} ='{val}'" for x in tmp_col]
                lt.extend([f"{x} ='{val}'" for x in tmp_val])
                rep = ", ".join(lt)
                badval_l.append(f"{tab} = '{val}'")
                change_val.append(f"{tab} = '{val}': {rep}")
        if badval_l:
            v_l = "\n".join(change_val)
            prompt = f"""Database Schema:
{new_db_info}

#question: {question}
#SQL: {sql}
ERROR: 数据库中不存在: {', '.join(badval_l)}
请用以下条件重写SQL:\n{v_l}

请直接给出新的SQL,不要回复任何其他内容:
#SQL:"""
            sql_response = self.chat_model.get_ans(prompt, 0.0)
            sql, _ = sql_raw_parse(sql_response, False)
        return sql

    def correct_sql(self, db_type, db_path, sql, query, db_info, hint, key_col_des, new_prompt, db_col={},
                    foreign_set={}, L_values=[], credential=None, db_id=None):
        count = 0
        raw = sql
        none_case = False

        # Prepare exec_sql function with proper database connection handling
        def exec_sql(sql_query):
            exec_args = {"sql_query": sql_query, "db_path": db_path, "db_id": db_id, "credential_path": credential}

            return get_sql_exec_result(db_type, **exec_args)

        while count <= 3:
            df, err = exec_sql(sql)
            try:
                if err:
                    raise ValueError(err)
                if len(df) == 0:
                    raise ValueError("Error: Result: None")
                else:
                    break
            except Exception as e:
                if count >= 3:
                    wsql = sql
                    sql_response = self.chat_model.get_ans(new_prompt, 0.2)
                    sql, _ = sql_raw_parse(sql_response, False)
                    none_case = True
                    break
                count += 1
                tag = str(e)
                e_s = str(e).split("':")[-1] if "':" in str(e) else str(e)
                result_info = f"{sql}\nError: {e_s}"
            if sql.find("SELECT") == -1:
                sql_response = self.chat_model.get_ans(new_prompt, 0.3)
                sql, _ = sql_raw_parse(sql_response, False)
            else:
                fewshot = self.correct_dic.get("default", "")
                advice = ""
                for x in self.correct_dic:
                    if tag.find(x) != -1:
                        fewshot = self.correct_dic[x]
                        if e_s == "Result: None":
                            sql_re = retable(sql)
                            adv = column_pick(sql_re, db_col, foreign_set)
                            adv = '\n'.join(adv)
                            val_advs = values_pick(L_values, sql_re)
                            val_advs = '\n'.join(val_advs)
                            func_call = func_find(sql)
                            if len(adv) or len(val_advs) or len(func_call):
                                advice = "#Change Ambiguity: " + "(replace or add)\n"
                                l = [x for x in [adv, val_advs, func_call] if len(x)]
                                advice += "\n\n".join(l)
                        elif x == "no such column":
                            advice += "Please check if this column exists in other tables"
                        break
                cor_prompt = self.correct_prompt.format(fewshot=fewshot, db_info=db_info, key_col_des=key_col_des,
                                                        q=query, hint=hint, result_info=result_info, advice=advice)
                sql_response = self.chat_model.get_ans(cor_prompt, 0.2 + count / 5)
                sql, _ = sql_raw_parse(sql_response, False)
            raw = sql
        return sql, none_case


def get_sql_ans(SQL: str, db_type: str, db_path: str, credential: Optional[str] = None, db_id: Optional[str] = None):
    """Execute SQL and get answer with proper database connection handling"""
    try:
        s = time.time()

        # Prepare arguments for get_sql_exec_result
        exec_args = {"sql_query": SQL, "db_path": db_path, "db_id": db_id, "credential_path": credential}

        df, err = get_sql_exec_result(db_type, **exec_args)
        if err:
            raise ValueError(err)

        ans = set(tuple(x) for x in df.values)
        time_cost = time.time() - s
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        ans, time_cost = set(), 100000
    return ans, time_cost


def process_sql(dcheck, sql, l_values, values, question, new_db_info, db_col_keys, hint, key_col_des, tmp_prompt,
                db_col, foreign_set, align_methods, db_type, db_path, credential=None, db_id=None):
    """Process SQL with proper database connection handling"""
    node_names = align_methods.split('+')
    align_functions = {
        "agent_align": dcheck.double_check_agent_align,
        "style_align": dcheck.double_check_style_align,
        "function_align": dcheck.double_check_function_align
    }
    sql = re.sub("(COUNT)(\\([^(]*? THEN 1 ELSE 0.*?\\))", r"SUM\2", sql)
    sql_retable = retable(sql)
    judgment = None
    sql_history = {}
    sql_correct = sql

    for node_name in node_names:
        if node_name in align_functions:
            if node_name == "agent_align":
                sql, judgment = align_functions[node_name](sql_retable, l_values, values, sql, question, new_db_info,
                                                           db_col_keys, hint)
            elif node_name == "style_align":
                sql, judgment = align_functions[node_name](sql, question, db_col_keys, sql_retable)
            elif node_name == "function_align":
                sql, judgment = align_functions[node_name](sql, question, db_path, credential, db_id)
            sql_history[node_name] = sql

    align_sql = sql
    can_ex = True
    nocse = True
    ans = set()
    time_cost = 10000000

    try:
        sql, nocse = dcheck.correct_sql(db_type, db_path, sql, question, new_db_info, hint, key_col_des, tmp_prompt,
                                        db_col, foreign_set, l_values, credential, db_id)
    except Exception as e:
        logger.error(f"Error in correct_sql: {e}")
        can_ex = False

    if can_ex:
        ans, time_cost = get_sql_ans(sql, db_type, db_path, credential, db_id)

    return sql_history, sql, ans, nocse, time_cost, align_sql, ans, sql_correct, ans


def muti_process_sql(dcheck, sqls, l_values, values, question, new_db_info, hint, key_col_des, tmp_prompt, db_col,
                     foreign_set, align_methods, db_type, db_path, n, credential=None, db_id=None):
    """Multi-process SQL processing with proper database connection handling"""
    vote = []
    none_case = False

    with ThreadPoolExecutor(max_workers=n) as executor:
        future_to_sql = {
            executor.submit(process_sql, dcheck, sql, l_values, values, question, new_db_info, db_col.keys(), hint,
                            key_col_des, tmp_prompt, db_col, foreign_set, align_methods, db_type, db_path,
                            credential, db_id): (sqls[sql], sql) for sql in sqls}

        for future in as_completed(future_to_sql):
            count, tmp_sql = future_to_sql[future]
            try:
                sql_history, sql, ans, none_c, time_cost, align_sql, align_ans, sql_correct, correct_ans = future.result()
                vote.append({
                    "sql_history": sql_history,
                    "sql": sql,
                    "answer": list(ans),  # convert set to list for JSON
                    "count": count,
                    "time_cost": time_cost,
                    "align_sql": align_sql,
                    "align_ans": list(align_ans),
                    "correct_sql": sql_correct,
                    "correct_ans": list(correct_ans)
                })
                none_case = none_case or none_c
            except Exception as e:
                logger.error(f"Error processing SQL candidate: {e}")
                vote.append({
                    "sql_history": tmp_sql,
                    "sql": tmp_sql,
                    "answer": [],
                    "count": 1,
                    "time_cost": 10000000,
                    "align_sql": tmp_sql,
                    "correct_sql": tmp_sql,
                    "align_ans": [],
                    "correct_ans": []
                })
                none_case = True
    return vote, none_case


@BaseGenerator.register_actor
class OpenSearchSQLGenerator(BaseGenerator):
    NAME = "OpenSearchSQLGenerator"

    EXTRACT_PROMPT = """/* Some extract examples are provided based on similar problems: */
/* Answer the following: Please give the name of the course in which most numbers of the students got an A. Also, list the full name of the students who got an A in this course. most number of students got an A refers MAX(COUNT(student_id WHERE grade = 'A')); full name = f_name, l_name; got an A refers to grade = 'A'; */
#reason: The question requires display in order: "name of the course", "full name"."A" is filtering condition.
#columns: course.name, student.f_name, student.l_name, registration.grade, registration.student_id
#values: "A"

/* Answer the following:How much more votes for episode 1 than for episode 5? more votes refers to SUBTRACT(SUM(votes when episode = 1), SUM(votes when episode = 5)) */
#reason: The question requires display in order: "How much more vote". The definition of "more vote" is SUBTRACT(SUM(votes when episode = 1), SUM(votes when episode = 5)). 1, 5 are filtering conditions.
#columns: Episode.episode, Vote.votes
#values: "1", "5"

/* Answer the following: What is the average score of the movie "The Fall of Berlin" in 2019? Average score refers to Avg(rating_score); */
#reason: The question requires display in order: "average score". Average score is Avg(rating_score), "The Fall of Berlin",2019 are filtering conditions.
#columns: ratings.rating_score, ratings.rating_id, ratings.rating_timestamp_utc, movies.movie_title
#values: "The Fall of Berlin", "2019"

/* Answer the following: How many distinct orders were there in 2003 when the quantity ordered was less than 30? "year(orderDate) = '2003'; quantityOrdered < 30;" */
#reason:  The question requires display in order: "How many distinct orders"." in 2003", "less than 30" are filtering conditions.
#columns: orderdetails.orderNumber, orderdetails.quantityOrdered, orders.orderDate
#values: "30", "2003"

{fewshot}

/* Database schema */
{db_info}

Attention:
1. if the question have when\\where\\which, pay attention to pick table.column related to time, location and name in #columns
2. Please answer the question in the following format without any other content:
```
#reason: Analysis of which columns and values might be relevant to the question. Note that when dealing with questions about time, who, which, what, etc., you should keep column related to time, names, and locations in the #column.(format: the question query xxx, the related column include table.column, the values include values)
#columns: The top 10 columns relevant to the question( format: table.column_1, table.column_2 ...)
#values: Potential filter values that the question might query(format: "value1", "value2" ...)
```
/* Answer the following: {query} */
"""

    NOUN_PROMPT = """Please extract all nouns and phrases from the following sentence, separating the results directly with a comma( format: "noun_1", "noun_2","phrases" ):
{raw_question}"""

    SELECT_PROMPT = """现在我们定义一个问句的语法原子单元如下:
Q: 询问词: 如 calculate\\ Include\\ List\\ List out\\ List all\\ give\\ state\\ Name\\ In which\\ How many\\  which\\ what\\ who\\ when\\ provide\\ Tally\\ identify\\ Find\\ mention\\ write等
J: 判断词： 如 Do\\ Did\\ If\\ Is\\ Are等
I: 查询内容: 查询的主体内容, 如: name, ID, date, location, item, biggest city.
C: 条件句: 通过介词和连词引入的查询的要求或属性, 如大于、等于、排序、聚合等. 介词和连词有: of\\ have\\ with\\ that\\ by. 条件句的形式例子有: with condition\\ have condition\\ of attribute\\ that was condition


一个问题通过这些原子串联起来。常见的串联方式有
QIC(询问句): List the student with score more than 80: Q: 'List' I: 'the student' C: 'with score more than 80'
JC(判断句): State if Tom is a Cat? : J: 'State if C: is a Cat?'
C(条件句): For all people in Beijing
现在请你针对下面的问题, 把问题中的内容按照上述原子定义提取出来
问题如下: {question}

请按照下面的json格式进行回答:

```json
[{{"Type":"类型(QIC,JC,C)",
"Extract":{{//不存在的填null
    "Q":"询问词",
    "J":"判断词",
    "I":['查询内容a', '查询内容b'],//只有查询内容用and或alongside连接时,才分成多个实体填入List
    "C":["条件句a","属性b"]
}}}},
{{}}]
```"""

    NEW_PROMPT = """{fewshot}

/* Database schema */
{db_info}
{key_col_des}

# Based on the database schema and the examples above, pay attention to the following:
1. For parts involving division that contain integer types, CAST them to REAL.
2. "#Values in Database" display part values from the database. Please ignore the unnecessary values.
3. Please refer to the examples above and answer in the following format without any other content:
```
#reason: Analyze how to generate SQL based on the question.(format: the question want to ..., so the SQL SELECT ... and ...)
#columns: All columns ultimately used in SQL(format: table.column_1, table.column_2)
#values: the filter in SQL (format: 'filter in question' refer to 'table.column <op> value'. e.g. 'name is not tom' refer to name <> 'tom', 'in 2007' refer to "strftime('%Y', Date) = '2007'")
#SELECT: SELECT content (format like: 'query in question' refer to table.column. The order of columns in the SELECT clause must be the same as the order in the question.)
#SQL-like: SQL-like statements ignoring Join conditions
#SQL: SQL
```

/* Answer the following: {question} */
{q_order}"""

    TMP_PROMPT = """You are an SQL expert, and now I would like you to write SQL based on the question.
{fewshot}

/* Database schema */
{db_info}
{key_col_des}

/* Based on the database schema and the question, pay attention to the following */
1. For parts involving division that contain integer types, CAST them to REAL.
2. #values in db display part values from the database. Please ignore the unnecessary values.

Please rewrite the question to SQL-like query in the format: "Show #SELECT (table.column), WHERE condition are xxx (refer to #values), Group by/Order By (refer to columns). Here are 3 example: 

#SQL-like: Show top 5 cards.id, where condition is cards.spend>100, order by cards.spend. 
#SQL: SELECT id FROM cards WHERE spend > 100 ORDER BY spend LIMIT 5

#SQL-like: Show Count(PaperAuthor.Name), Where condition is Paper.Year = 0
#SQL: SELECT COUNT(T2.Name) FROM Paper AS T1 INNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId WHERE T1.Year = 0

#SQL-like: Show Author.Name, Where condition is Author.Affiliation = 'University of Oxford', Group by Author.Name Order By Author.spent
#SQL: SELECT Name FROM Author WHERE Affiliation = 'University of Oxford' Group By Name ORDER BY spent ASC

/* Answer the following: {question} */
Please answer the question in the following format without any other content:
```
#reason: Analyze how to generate SQL based on the question.(format: the question want to ..., so the SQL SELECT ... and ...)
#columns: All columns ultimately used in SQL(format: table.column_1, table.column_2)
#values: the filter in SQL (format: 'filter in question' refer to table.column: value. e.g. 'name is not tom' refer to name <> "tom", 2007 refer to strftime('%Y', Date) = '2007')
#SELECT: SELECT content (display in the order asked by the questions, do not display content not specified by the questions).
#SQL-like: SQL-like statements ignoring Join conditions
#SQL: SQL
```"""

    SOFT_PROMPT = """Your task is to perform a simple evaluation of the SQL.

The database system is SQLite. The SQL you need to evaluation is:
#question: {question}
#SQL: {SQL}

Answer in the following format: 
{{
"Judgment": true/false,
"SQL":If SQL is wrong, please correct SQL directly. else answer ""
}}"""

    CORRECT_PROMPT = """You are an expert in SQL. Here are some examples of fix SQL
{fewshot}

/* Database schema is as follows: */
{db_info}
{key_col_des}

/* Now Plesease fix the following error SQL */
#question: {q}
#Error SQL: {result_info}
{advice}

Please answer according to the format below and do not output any other content.:
```
#reason: Analysis of How to fix the error
#SQL: right SQL
```"""

    VOTE_PROMPT = """现在有问题如下:
#question: {question}
对应这个问题有如下几个SQL,请你从中选择最接近问题要求的SQL:
{sql}

请在上面的几个SQL中选择最符合题目要求的SQL, 不要回复其他内容:
#SQL:"""

    def __init__(self, dataset=None, llm=None, bert_model_name="all-mpnet-base-v2", n_candidates=21, temperature=0.7,
                 top_k=10, use_few_shot=True, use_feedback_debug=True, debug_turn_n=2, db_path=None, credential=None,
                 is_save=True, save_dir="../files/pred_sql", tables_json_path=None, tables_info_dir=None, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.llm = llm
        self.bert_model = SentenceTransformer(bert_model_name)
        self.n_candidates = n_candidates
        self.temperature = temperature
        self.top_k = top_k
        self.use_few_shot = use_few_shot
        self.use_feedback_debug = use_feedback_debug
        self.debug_turn_n = debug_turn_n

        # Follow the same pattern as LinkAlignGenerator for database parameters
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

        self.is_save = is_save
        self.save_dir = save_dir
        self.tables_json_path = tables_json_path
        self.tables_info_dir = tables_info_dir
        self.correct_dic = {"default": ""}  # Simplified

    def get_ans(self, prompt, temperature=0.0, n=1, single=True):
        """Get answer from LLM with proper error handling"""
        try:
            if single and n == 1:
                return self.llm.complete(prompt, temperature=temperature).text
            else:
                responses = [self.llm.complete(prompt, temperature=temperature).text for _ in range(n)]
                return responses
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            if single and n == 1:
                return ""
            else:
                return [""] * n

    def get_sql(self, prompt, temperature, return_question=False, n=1, single=False):
        """Generate SQL queries from prompt"""
        try:
            responses = self.get_ans(prompt, temperature, n, single)
            if not isinstance(responses, list):
                responses = [responses]

            sqls = []
            for resp in responses:
                sql, rwq = sql_raw_parse(resp, return_question)
                sqls.append(sql)
            return sqls, None
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return ["SELECT 1"], None  # Return fallback SQL

    def act(self, item, schema=None, schema_links=None, data_logger=None, **kwargs):
        """Generates SQL for a single data sample following Generator interface."""
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        logger.info(f"OpenSearchSQLGenerator starting to process sample {item}")

        try:
            # Get data sample
            row = self.dataset[item]
            question = row['question']
            db_id = row['db_id']
            evidence = row.get('evidence', 'None')
            if evidence == '':
                evidence = 'None'
            db_type = row.get('db_type', 'sqlite')

            # Follow the same pattern as LinkAlignGenerator for database path handling
            if db_type == 'sqlite':
                db_path = str(Path(self.db_path) / (db_id + ".sqlite")) if self.db_path else None
            else:
                db_path = self.db_path

            logger.info(f"Starting SQL generation for question: {question} in database: {db_id} (type: {db_type})")

            # Validate required parameters
            if not db_path and db_type == 'sqlite':
                raise ValueError(f"Database path is required for {db_type} database")
            if db_type in ['snowflake', 'big_query'] and not self.credential:
                raise ValueError(f"Credential is required for {db_type} database")

            fewshot = ''
            if self.use_few_shot:
                reasoning_example_path = row.get("reasoning_examples", None)
                if reasoning_example_path:
                    fewshot = load_dataset(reasoning_example_path)

            # Step 1: generate_db_schema
            db_agent = DB_AGENT(self)
            all_info, db_col_dic, foreign_set = db_agent.get_allinfo(
                self.tables_json_path, db_id, db_path, self.db_path, self.tables_info_dir,
                self.bert_model, db_type, self.credential, db_id
            )

            logger.info(f"Generated database schema for {db_id}")

            # Step 2: extract_col_value
            ext_prompt = self.EXTRACT_PROMPT.format(fewshot=fewshot, db_info=all_info, query=question)
            key_col_des_raw = self.get_ans(ext_prompt, temperature=0.0)

            logger.info("Extracted column and value information")

            # Step 3: extract_query_noun
            noun_prompt = self.NOUN_PROMPT.format(raw_question=question)
            noun_ext = self.get_ans(noun_prompt, temperature=0.0)
            values_noun, col = parse_des(key_col_des_raw, noun_ext, debug=False)

            logger.info(f"Extracted nouns: {values_noun}")

            # Step 4: column_retrieve_and_other_info
            try:
                col_retrieve = ColumnRetriever(self.bert_model, self.tables_info_dir).get_col_retrieve(
                    question, db_id, db_col_dic.keys()) if self.tables_info_dir else set()
                cols = ColumnUpdater(db_col_dic).col_pre_update(col, col_retrieve, foreign_set)
                des = DES_new(self.bert_model, self.bert_model.encode(list(db_col_dic.keys()), show_progress_bar=False),
                              list(db_col_dic.keys()))
                cols_select, L_values = des.get_key_col_des(cols, values_noun, debug=False, topk=self.top_k, shold=0.65)
                column = ColumnUpdater(db_col_dic).col_suffix(cols_select)
                foreign_keys, _ = find_foreign_keys_MYSQL_like(self.tables_json_path,
                                                               db_id) if self.tables_json_path else (
                    "", set())
                q_order = query_order(question, self, self.SELECT_PROMPT, temperature=0.3)
                q_order = " ".join(q_order) if q_order else ""
            except Exception as e:
                logger.error(f"Error in column retrieval: {e}")
                cols_select, L_values = [], []
                column, foreign_keys, q_order = "", "", ""

            logger.info(f"Retrieved and updated columns: {cols_select}")

            # Step 5: candidate_generate
            values_str = [f"{x[0]}: '{x[1]}'" for x in L_values]
            key_col_des = "#Values in Database:\n" + '\n'.join(values_str)

            # Adapt database system name based on db_type
            db_system_name = {
                'sqlite': 'SQLite',
                'snowflake': 'Snowflake',
                'big_query': 'BigQuery'
            }.get(db_type, 'SQLite')

            new_db_info = f"Database Management System: {db_system_name}\n#Database name: {db_id} \n{column}\n\n#Foreign keys:\n{foreign_keys}\n"
            new_prompt = self.NEW_PROMPT.format(fewshot=fewshot, db_info=new_db_info, key_col_des=key_col_des,
                                                question=question, q_order=q_order)
            SQLs, _ = self.get_sql(new_prompt, self.temperature, return_question=True, n=self.n_candidates,
                                   single=False)

            logger.info(f"Generated {len(SQLs)} candidate SQL queries")
            if data_logger:
                data_logger.info(f"{self.NAME}.candidates | count={len(SQLs)}")

            # Step 6: align_correct
            dcheck = soft_check(self.bert_model, self, self.SOFT_PROMPT, self.correct_dic, self.CORRECT_PROMPT,
                                self.VOTE_PROMPT)
            SQLs_dic = {}
            for sql in SQLs:
                sql, _ = sql_raw_parse(sql, False)
                SQLs_dic.setdefault(sql, 0)
                SQLs_dic[sql] += 1

            tmp_prompt = self.TMP_PROMPT.format(fewshot=fewshot, db_info=new_db_info, key_col_des=key_col_des,
                                                question=question)

            # Pass database connection parameters properly
            vote, none_case = muti_process_sql(
                dcheck, SQLs_dic, L_values, values_noun, question, new_db_info, evidence,
                key_col_des, tmp_prompt, db_col_dic, foreign_set,
                "style_align+function_align+agent_align", db_type, db_path,
                self.n_candidates, self.credential, db_id
            )

            # Step 7: vote
            vote_sqls = [v["sql"] for v in vote]
            pred_sql = dcheck.vote_chose(vote_sqls, question)

            logger.info(f"Selected SQL after voting: {pred_sql}")

            # Optional debugging with proper parameter handling
            if self.use_feedback_debug:
                try:
                    debug_args = {
                        "question": question,
                        "schema": all_info,
                        "sql_query": pred_sql,
                        "llm": self.llm,
                        "db_id": db_id,
                        "db_path": db_path,
                        "db_type": db_type,
                        "credential": self.credential,
                        "debug_turn_n": self.debug_turn_n
                    }
                    _, pred_sql = sql_debug_by_feedback(**debug_args)
                    logger.info(f"SQL after feedback debug: {pred_sql}")
                except Exception as e:
                    logger.error(f"Error in feedback debug: {e}")
                    # Continue with the original pred_sql

            # Ensure pred_sql is not empty or None
            if not pred_sql or pred_sql.strip() == "":
                # Try to get a valid SQL from the vote results
                for v in vote:
                    if v.get("sql") and v["sql"].strip():
                        pred_sql = v["sql"]
                        break

                # If still empty, use fallback
                if not pred_sql or pred_sql.strip() == "":
                    pred_sql = "SELECT 1"  # Fallback SQL

            # Clean up the SQL
            pred_sql = pred_sql.strip()
            if not pred_sql.startswith("SELECT"):
                pred_sql = "SELECT 1"

            # Ensure we have valid SQL to save
            if not pred_sql or not pred_sql.strip():
                pred_sql = "SELECT 1"
                logger.warning(f"Using fallback SQL for item {item}")

            pred_sql = self.save_output(pred_sql, item, row.get("instance_id"))

            logger.info(f"OpenSearchSQLGenerator completed processing sample {item}")
            logger.info(f"Final predicted SQL: {pred_sql}")
            if data_logger:
                data_logger.info(f"{self.NAME}.final_sql | sql={pred_sql}")
                data_logger.info(f"{self.NAME}.act end | item={item}")

            return pred_sql
        except Exception as e:
            logger.error(f"Error generating SQL for sample {item}: {str(e)}")

            # Provide a fallback SQL in case of errors
            fallback_sql = "SELECT 1"
            instance_id = row.get("instance_id", item) if 'row' in locals() else item
            fallback_sql = self.save_output(fallback_sql, item, instance_id)

            return fallback_sql  # Return fallback instead of raising exception
