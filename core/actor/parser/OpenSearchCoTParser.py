import pandas as pd
import re
import sqlite3
import os
import chardet
import json
from sentence_transformers import SentenceTransformer
from core.actor.parser.BaseParse import BaseParser
from core.utils import load_dataset, save_dataset
from pathlib import Path

import numpy as np


class OpenSearchCoTParser(BaseParser):
    NAME = "OpenSearchCoTParser"

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

    def __init__(self, dataset=None, llm=None, bert_model_name="all-mpnet-base-v2", top_k=10, tables_json_path=None,
                 tables_info_dir=None, db_path=None, is_save=True, save_dir="../files/schema_links",
                 output_format="list", use_few_shot=False, **kwargs):
        self.dataset = dataset
        self.llm = llm
        self.bert_model = SentenceTransformer(bert_model_name)
        self.top_k = top_k
        self.tables_json_path = tables_json_path
        self.tables_info_dir = tables_info_dir
        self.db_path = db_path
        self.is_save = is_save
        self.save_dir = save_dir
        self.output_format = output_format
        self.use_few_shot = use_few_shot

    def get_ans(self, prompt, temperature=0.0):
        return self.llm.complete(prompt, temperature=temperature).text

    def find_foreign_keys_MYSQL_like(self, DATASET_JSON, db_name):
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

    def quote_field(self, field_name):
        if re.search(r'\W', field_name):
            return f"`{field_name}`"
        else:
            return field_name

    class DB_AGENT:
        def __init__(self, parser) -> None:
            self.parser = parser

        def get_complete_table_info(self, conn, table_name, table_df):
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns_info = cursor.fetchall()
            df = pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)
            contains_null = {
                column: df[column].isnull().any()
                for column in df.columns
            }
            contains_duplicates = {
                column: df[column].duplicated().any()
                for column in df.columns
            }
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
            row = list(cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 1").fetchall()[0])
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
                    row[i] = val_p
                except:
                    pass
            schema_str = f"## Table {table_name}:\n"
            columns = {}
            for column, val in zip(columns_info, row):
                schema_str_single = ""
                column_name, column_type, not_null, default_value, pk = column[1:6]
                tmp_col = column_name.strip()
                column_name = self.parser.quote_field(column_name)
                col_des, val_des = dic.get(tmp_col, ["", ""])
                if col_des != "":
                    schema_str_single += f" The column is {col_des}. "
                if val_des != "":
                    schema_str_single += f" The values' format are {val_des}. "

                schema_str_single += f"The type is {column_type}, "
                if contains_null[tmp_col]:
                    schema_str_single += f"Which include Null"
                else:
                    schema_str_single += f"Which does not include Null"

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
                columns[f"{table_name}.{column_name}"] = (schema_str_single, col_des, val_des, column_type,
                                                          include_null, unique, str(val))
            return schema_str, columns

        def get_db_des(self, sqllite_dir, db_dir, model):
            conn = sqlite3.connect(sqllite_dir)
            table_dir = os.path.join(db_dir, 'database_description') if db_dir else None
            sql = "SELECT name FROM sqlite_master WHERE type='table';"
            cursor = conn.cursor()
            tables = cursor.execute(sql).fetchall()
            db_info = []
            db_col = dict()
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
                table_info, columns = self.get_complete_table_info(conn, table[0], table_df)
                db_info.append(table_info)
                db_col.update(columns)
            db_info = "\n".join(db_info)
            cursor.close()
            conn.close()
            return db_info, db_col

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

        def get_allinfo(self, db_json_dir, db, sqllite_dir, db_dir, tables_info_dir, model):
            db_info, db_col = self.get_db_des(sqllite_dir, db_dir, model)
            foreign_keys, foreign_set = self.parser.find_foreign_keys_MYSQL_like(tables_info_dir,
                                                                                 db) if tables_info_dir else ("", set())
            all_info = f"Database Management System: SQLite\n#Database name: {db}\n{db_info}\n#Forigen keys:\n{foreign_keys}\n"
            prompt = self.db_conclusion(all_info)
            db_all = self.parser.get_ans(prompt)
            all_info = f"{all_info}\n{db_all}\n"
            return all_info, db_col, foreign_set

    def parse_des(self, pre_col_values, nouns, debug):
        pre_col_values = pre_col_values.split("/*")[0].strip()
        if debug:
            print(pre_col_values)
        col, values = pre_col_values.split('#values:')
        _, col = col.split("#columns:")
        col = self.strip_char(col)
        values = self.strip_char(values)

        if values == '':
            values = []
        else:
            values = re.findall(r"([\"'])(.*?)\1", values)
        nouns_all = re.findall(r"([\"'])(.*?)\1", nouns)
        values_noun = set(values).union(set(nouns_all))
        values_noun = [x[1] for x in values_noun]
        return values_noun, col

    def strip_char(self, s):
        return s.strip('\n {}[]')

    class ColumnRetriever:
        def __init__(self, bert_model, tables_info_dir):
            self.bert_model = bert_model
            self.tables_info_dir = tables_info_dir

        def get_col_retrieve(self, question, db, db_keys_col):
            db_schema = json.load(open(self.tables_info_dir, 'r'))
            db_schema = db_schema[db]
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

    def json_ext(self, jsonf):
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

    def query_order(self, question, temperature):
        select_prompt = self.SELECT_PROMPT.format(question=question)
        ans = self.get_ans(select_prompt, temperature=temperature)
        ans = re.sub("```json|```", "", ans)
        select_json = json.loads(ans)
        res, judge = self.json_ext(select_json)
        return res, judge

    def act(self, item, schema=None, **kwargs):
        row = self.dataset[item]
        question = row["question"]
        db_id = row["db_id"]
        db_sqlite_path = str(Path(self.db_path) / (db_id + ".sqlite"))

        fewshot = ''
        if self.use_few_shot:
            reasoning_example_path = row.get("reasoning_examples", None)
            if reasoning_example_path:
                fewshot = load_dataset(reasoning_example_path)

        db_agent = self.DB_AGENT(self)
        all_info, db_col, foreign_set = db_agent.get_allinfo(self.tables_json_path, db_id, db_sqlite_path, self.db_path,
                                                             self.tables_info_dir, self.bert_model)

        ext_prompt = self.EXTRACT_PROMPT.format(fewshot=fewshot, db_info=all_info, query=question)
        key_col_des_raw = self.get_ans(ext_prompt, temperature=0.0)

        noun_prompt = self.NOUN_PROMPT.format(raw_question=question)
        noun_ext = self.get_ans(noun_prompt, temperature=0.0)
        values_noun, col = self.parse_des(key_col_des_raw, noun_ext, debug=False)

        col_retrieve = self.ColumnRetriever(self.bert_model, self.tables_info_dir).get_col_retrieve(question, db_id,
                                                                                                    db_col.keys()) if self.tables_info_dir else set()

        col_updater = self.ColumnUpdater(db_col)
        cols = col_updater.col_pre_update(col, col_retrieve, foreign_set)

        des = self.DES_new(self.bert_model, self.bert_model.encode(list(db_col.keys()), show_progress_bar=False),
                           list(db_col.keys()))
        cols_select, L_values = des.get_key_col_des(cols, values_noun, debug=False, topk=self.top_k, shold=0.65)

        schema_links = list(set(cols_select + [x[1] for x in L_values]))

        output = str(schema_links) if self.output_format == "str" else schema_links

        if self.is_save:
            instance_id = row.get("instance_id", item)
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if hasattr(self.dataset,
                                                                               'dataset_index') and self.dataset.dataset_index else save_path
            file_ext = ".txt" if self.output_format == "str" else ".json"
            save_path = save_path / f"{self.NAME}_{instance_id}{file_ext}"

            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_dataset(output, new_data_source=save_path)
            self.dataset.setitem(item, "schema_links", str(save_path))

        return output
