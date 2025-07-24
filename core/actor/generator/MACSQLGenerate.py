import sys
import os
import json
import time
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional, Any
from loguru import logger
import re

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, load_dataset, save_dataset
from core.utils import parse_schema_from_df, single_central_process
from llama_index.core.llms.llm import LLM

# Remove the sys.path.append and direct imports from baselines

# Embedded constants from const.py
MAX_ROUND = 3
SELECTOR_NAME = 'Selector'
DECOMPOSER_NAME = 'Decomposer'
REFINER_NAME = 'Refiner'
SYSTEM_NAME = 'System'

# Simplified utility functions from utils.py
def parse_json(text: str) -> dict:
    start = text.find("```json")
    end = text.find("```", start + 7)
    if start != -1 and end != -1:
        json_string = text[start + 7: end]
        try:
            return json.loads(json_string)
        except:
            return {}
    return {}

def parse_sql_from_string(input_string):
    sql_pattern = r'```sql(.*?)```'
    all_sqls = []
    for match in re.finditer(sql_pattern, input_string, re.DOTALL):
        all_sqls.append(match.group(1).strip())
    if all_sqls:
        return all_sqls[-1]
    else:
        return "error: No SQL found in the input string"

# Embedded prompt templates from const.py (simplified for brevity)
selector_template = '''\nAs an experienced and professional database administrator, your task is to analyze a user question and a database schema to provide relevant information. The database schema consists of table descriptions, each containing multiple column descriptions. Your goal is to identify the relevant tables and columns based on the user question and evidence provided.\n\n[Instruction]:\n1. Discard any table schema that is not related to the user question and evidence.\n2. Sort the columns in each relevant table in descending order of relevance and keep the top 6 columns.\n3. Ensure that at least 3 tables are included in the final output JSON.\n4. The output should be in JSON format.\n\nRequirements:\n1. If a table has less than or equal to 10 columns, mark it as \"keep_all\".\n2. If a table is completely irrelevant to the user question and evidence, mark it as \"drop_all\".\n3. Prioritize the columns in each relevant table based on their relevance.\n\nHere is a typical example:\n\n==========\n【DB_ID】 banking_system\n【Schema】\n# Table: account\n[\n  (account_id, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),\n  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),\n  (frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),\n  (date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)\n]\n# Table: client\n[\n  (client_id, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),\n  (gender, gender. Value examples: ['M', 'F']. And F：female . M：male ),\n  (birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),\n  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].)\n]\n# Table: loan\n[\n  (loan_id, the id number identifying the loan data. Value examples: [4959, 4960, 4961].),\n  (account_id, the id number identifying the account. Value examples: [10, 80, 55, 43].),\n  (date, the date when the loan is approved. Value examples: ['1998-07-12', '1998-04-19'].),\n  (amount, the id number identifying the loan data. Value examples: [1567, 7877, 9988].),\n  (duration, the id number identifying the loan data. Value examples: [60, 48, 24, 12, 36].),\n  (payments, the id number identifying the loan data. Value examples: [3456, 8972, 9845].),\n  (status, the id number identifying the loan data. Value examples: ['C', 'A', 'D', 'B'].)\n]\n# Table: district\n[\n  (district_id, location of branch. Value examples: [77, 76].),\n  (A2, area in square kilometers. Value examples: [50.5, 48.9].),\n  (A4, number of inhabitants. Value examples: [95907, 95616].),\n  (A5, number of households. Value examples: [35678, 34892].),\n  (A6, literacy rate. Value examples: [95.6, 92.3, 89.7].),\n  (A7, number of entrepreneurs. Value examples: [1234, 1456].),\n  (A8, number of cities. Value examples: [5, 4].),\n  (A9, number of schools. Value examples: [15, 12, 10].),\n  (A10, number of hospitals. Value examples: [8, 6, 4].),\n  (A11, average salary. Value examples: [12541, 11277].),\n  (A12, poverty rate. Value examples: [12.4, 9.8].),\n  (A13, unemployment rate. Value examples: [8.2, 7.9].),\n  (A15, number of crimes. Value examples: [256, 189].)\n]\n【Foreign keys】\nclient.`district_id` = district.`district_id`\n【Question】\nWhat is the gender of the youngest client who opened account in the lowest average salary branch?\n【Evidence】\nLater birthdate refers to younger age; A11 refers to average salary\n【Answer】\n```json\n{{\n  \"account\": \"keep_all\",\n  \"client\": \"keep_all\",\n  \"loan\": \"drop_all\",\n  \"district\": [\"district_id\", \"A11\", \"A2\", \"A4\", \"A6\", \"A7\"]\n}}\n```\nQuestion Solved.\n\n==========\n\nHere is a new example, please start answering:\n\n【DB_ID】 {db_id}\n【Schema】\n{desc_str}\n【Foreign keys】\n{fk_str}\n【Question】\n{query}\n【Evidence】\n{evidence}\n【Answer】\n'''

decompose_template_bird = '''Given a 【Database schema】 description, a knowledge 【Evidence】 and the 【Question】, you need to use valid SQLite and understand the database and knowledge, and then decompose the question into subquestions for text-to-SQL generation.
When generating SQL, we should always consider constraints:
【Constraints】
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values

==========

【Database schema】
# Table: frpm
[
  (CDSCode, CDSCode. Value examples: ['01100170109835', '01100170112607'].),
  (Charter School (Y/N), Charter School (Y/N). Value examples: [1, 0, None]. And 0: N;. 1: Y),
  (Enrollment (Ages 5-17), Enrollment (Ages 5-17). Value examples: [5271.0, 4734.0].),
  (Free Meal Count (Ages 5-17), Free Meal Count (Ages 5-17). Value examples: [3864.0, 2637.0]. And eligible free rate = Free Meal Count / Enrollment)
]
# Table: satscores
[
  (cds, California Department Schools. Value examples: ['10101080000000', '10101080109991'].),
  (sname, school name. Value examples: ['None', 'Middle College High', 'John F. Kennedy High', 'Independence High', 'Foothill High'].),
  (NumTstTakr, Number of Test Takers in this school. Value examples: [24305, 4942, 1, 0, 280]. And number of test takers in each school),
  (AvgScrMath, average scores in Math. Value examples: [699, 698, 289, None, 492]. And average scores in Math),
  (NumGE1500, Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500. Value examples: [5837, 2125, 0, None, 191]. And Number of Test Takers Whose Total SAT Scores Are Greater or Equal to 1500. . commonsense evidence:. . Excellence Rate = NumGE1500 / NumTstTakr)
]
【Foreign keys】
frpm.`CDSCode` = satscores.`cds`
【Question】
List school names of charter schools with an SAT excellence rate over the average.
【Evidence】
Charter schools refers to `Charter School (Y/N)` = 1 in the table frpm; Excellence rate = NumGE1500 / NumTstTakr


Decompose the question into sub questions, considering 【Constraints】, and generate the SQL after thinking step by step:
Sub question 1: Get the average value of SAT excellence rate of charter schools.
SQL
```sql
SELECT AVG(CAST(T2.`NumGE1500` AS REAL) / T2.`NumTstTakr`)
    FROM frpm AS T1
    INNER JOIN satscores AS T2
    ON T1.`CDSCode` = T2.`cds`
    WHERE T1.`Charter School (Y/N)` = 1
```

Sub question 2: List out school names of charter schools with an SAT excellence rate over the average.
SQL
```sql
SELECT T2.`sname`
  FROM frpm AS T1
  INNER JOIN satscores AS T2
  ON T1.`CDSCode` = T2.`cds`
  WHERE T2.`sname` IS NOT NULL
  AND T1.`Charter School (Y/N)` = 1
  AND CAST(T2.`NumGE1500` AS REAL) / T2.`NumTstTakr` > (
    SELECT AVG(CAST(T4.`NumGE1500` AS REAL) / T4.`NumTstTakr`)
    FROM frpm AS T3
    INNER JOIN satscores AS T4
    ON T3.`CDSCode` = T4.`cds`
    WHERE T3.`Charter School (Y/N)` = 1
  )
```

Question Solved.

==========

【Database schema】
# Table: account
[
  (account_id, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),
  (date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)
]
# Table: client
[
  (client_id, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),
  (gender, gender. Value examples: ['M', 'F']. And F：female . M：male ),
  (birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].)
]
# Table: district
[
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (A4, number of inhabitants . Value examples: ['95907', '95616', '94812'].),
  (A11, average salary. Value examples: [12541, 11277, 8114].)
]
【Foreign keys】
account.`district_id` = district.`district_id`
client.`district_id` = district.`district_id`
【Question】
What is the gender of the youngest client who opened account in the lowest average salary branch?
【Evidence】
Later birthdate refers to younger age; A11 refers to average salary

Decompose the question into sub questions, considering 【Constraints】, and generate the SQL after thinking step by step:
Sub question 1: What is the district_id of the branch with the lowest average salary?
SQL
```sql
SELECT `district_id`
  FROM district
  ORDER BY `A11` ASC
  LIMIT 1
```

Sub question 2: What is the youngest client who opened account in the lowest average salary branch?
SQL
```sql
SELECT T1.`client_id`
  FROM client AS T1
  INNER JOIN district AS T2
  ON T1.`district_id` = T2.`district_id`
  ORDER BY T2.`A11` ASC, T1.`birth_date` DESC 
  LIMIT 1
```

Sub question 3: What is the gender of the youngest client who opened account in the lowest average salary branch?
SQL
```sql
SELECT T1.`gender`
  FROM client AS T1
  INNER JOIN district AS T2
  ON T1.`district_id` = T2.`district_id`
  ORDER BY T2.`A11` ASC, T1.`birth_date` DESC 
  LIMIT 1 
```
Question Solved.

==========

【Database schema】
{desc_str}
【Foreign keys】
{fk_str}
【Question】
{query}
【Evidence】
{evidence}

Decompose the question into sub questions, considering 【Constraints】, and generate the SQL after thinking step by step:
'''

decompose_template_spider = '''Given a 【Database schema】 description, a knowledge 【Evidence】 and the 【Question】, you need to use valid SQLite and understand the database and knowledge, and then decompose the question into subquestions for text-to-SQL generation.
When generating SQL, we should always consider constraints:
【Constraints】
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values

==========

【Database schema】
# Table: account
[
  (account_id, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),
  (date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)
]
# Table: client
[
  (client_id, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),
  (gender, gender. Value examples: ['M', 'F']. And F：female . M：male ),
  (birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].)
]
# Table: district
[
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (A4, number of inhabitants . Value examples: ['95907', '95616', '94812'].),
  (A11, average salary. Value examples: [12541, 11277, 8114].)
]
【Foreign keys】
account.`district_id` = district.`district_id`
client.`district_id` = district.`district_id`
【Question】
What is the gender of the youngest client who opened account in the lowest average salary branch?
【Evidence】
Later birthdate refers to younger age; A11 refers to average salary

Decompose the question into sub questions, considering 【Constraints】, and generate the SQL after thinking step by step:
'''

refiner_template = '''Given a 【Database schema】 description, a knowledge 【Evidence】 and the 【Question】, you need to use valid SQLite and understand the database and knowledge, and then decompose the question into subquestions for text-to-SQL generation.
When generating SQL, we should always consider constraints:
【Constraints】
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values

==========

【Database schema】
# Table: account
[
  (account_id, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),
  (date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)
]
# Table: client
[
  (client_id, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),
  (gender, gender. Value examples: ['M', 'F']. And F：female . M：male ),
  (birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].)
]
# Table: district
[
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (A4, number of inhabitants . Value examples: ['95907', '95616', '94812'].),
  (A11, average salary. Value examples: [12541, 11277, 8114].)
]
【Foreign keys】
account.`district_id` = district.`district_id`
client.`district_id` = district.`district_id`
【Question】
What is the gender of the youngest client who opened account in the lowest average salary branch?
【Evidence】
Later birthdate refers to younger age; A11 refers to average salary

Decompose the question into sub questions, considering 【Constraints】, and generate the SQL after thinking step by step:
'''

# Embedded Selector class
class Selector:
    name = SELECTOR_NAME
    def __init__(self, llm: LLM, data_path: str, tables_json_path: str, model_name: str, dataset_name:str, lazy: bool = False, without_selector: bool = False):
        self.llm = llm
        self.data_path = data_path.strip('/').strip('\\')
        self.tables_json_path = tables_json_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.db2infos = {}
        self.db2dbjsons = {}
        self.init_db2jsons()
        if not lazy:
            self._load_all_db_info()
        self._message = {}
        self.without_selector = without_selector
    def init_db2jsons(self):
        if not os.path.exists(self.tables_json_path):
            raise FileNotFoundError(f"tables.json not found in {self.tables_json_path}")
        data = load_json_file(self.tables_json_path)  # Embed load_json_file if needed
        for item in data:
            db_id = item['db_id']
            table_names = item['table_names']
            item['table_count'] = len(table_names)
            column_count_lst = [0] * len(table_names)
            for tb_idx, col in item['column_names']:
                if tb_idx >= 0:
                    column_count_lst[tb_idx] += 1
            item['max_column_count'] = max(column_count_lst)
            item['total_column_count'] = sum(column_count_lst)
            item['avg_column_count'] = sum(column_count_lst) // len(table_names)
            self.db2dbjsons[db_id] = item

    def _get_column_attributes(self, cursor, table):
        cursor.execute(f"PRAGMA table_info(`{table}`)")
        columns = cursor.fetchall()
        column_names = []
        column_types = []
        for column in columns:
            column_names.append(column[1])
            column_types.append(column[2])
        return column_names, column_types

    def _get_unique_column_values_str(self, cursor, table, column_name):
        cursor.execute(f"SELECT DISTINCT `{column_name}` FROM `{table}`")
        values = cursor.fetchall()
        if not values:
            return "No values found."
        return ", ".join([str(v[0]) for v in values])

    def _get_value_examples_str(self, cursor, table, column_name):
        cursor.execute(f"SELECT `{column_name}` FROM `{table}` LIMIT 5")
        values = cursor.fetchall()
        if not values:
            return "No value examples found."
        return ", ".join([str(v[0]) for v in values])

    def _load_single_db_info(self, db_id: str):
        if db_id in self.db2infos:
            return self.db2infos[db_id]
        db_path = os.path.join(self.data_path, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [t[0] for t in cursor.fetchall()]

        # Get column names and types for each table
        table_info = {}
        for table_name in table_names:
            column_names, column_types = self._get_column_attributes(cursor, table_name)
            table_info[table_name] = {
                "column_names": column_names,
                "column_types": column_types,
                "foreign_keys": [],
                "primary_keys": []
            }

            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
            foreign_keys = cursor.fetchall()
            for fk in foreign_keys:
                if fk[2] == '1': # 1 means 'ON DELETE CASCADE'
                    table_info[table_name]["foreign_keys"].append({
                        "column": fk[3],
                        "referenced_table": fk[4],
                        "referenced_column": fk[5]
                    })

            # Get primary keys
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            primary_keys = [row[1] for row in cursor.fetchall() if row[5] == 1] # 1 means 'PRIMARY KEY'
            table_info[table_name]["primary_keys"] = primary_keys

        conn.close()
        self.db2infos[db_id] = table_info
        return table_info

    def _load_all_db_info(self):
        for db_id in self.db2dbjsons.keys():
            self._load_single_db_info(db_id)

    def _build_bird_table_schema_list_str(self, table_info: Dict) -> str:
        schema_str = ""
        for table_name, info in table_info.items():
            schema_str += f"# Table: {table_name}\n[\n"
            for col_name in info["column_names"]:
                schema_str += f"  ({col_name}, {col_name} description. Value examples: [{self._get_value_examples_str(None, table_name, col_name)}].),\n"
            schema_str += "]\n"
        return schema_str

    def _get_db_desc_str(self, db_id: str, extracted_schema: Dict, use_gold_schema: bool = False) -> Tuple[str, str, Dict]:
        if use_gold_schema:
            table_info = self._load_single_db_info(db_id)
            desc_str = self._build_bird_table_schema_list_str(table_info)
            fk_str = self._get_foreign_keys_str(table_info)
            chosen_db_schem_dict = {}
            for table_name, info in table_info.items():
                chosen_db_schem_dict[table_name] = info["column_names"]
            return desc_str, fk_str, chosen_db_schem_dict

        # If using extracted schema, try to infer schema from it
        if extracted_schema:
            desc_str = self._build_bird_table_schema_list_str(extracted_schema)
            fk_str = self._get_foreign_keys_str(extracted_schema)
            chosen_db_schem_dict = {}
            for table_name, info in extracted_schema.items():
                chosen_db_schem_dict[table_name] = info["column_names"]
            return desc_str, fk_str, chosen_db_schem_dict

        # Fallback to loading full schema if no schema provided
        table_info = self._load_single_db_info(db_id)
        desc_str = self._build_bird_table_schema_list_str(table_info)
        fk_str = self._get_foreign_keys_str(table_info)
        chosen_db_schem_dict = {}
        for table_name, info in table_info.items():
            chosen_db_schem_dict[table_name] = info["column_names"]
        return desc_str, fk_str, chosen_db_schem_dict

    def _get_foreign_keys_str(self, table_info: Dict) -> str:
        fk_str = ""
        for table_name, info in table_info.items():
            for fk in info["foreign_keys"]:
                fk_str += f"{table_name}.`{fk['column']}` = {fk['referenced_table']}.`{fk['referenced_column']}`\n"
        return fk_str

    def _is_need_prune(self, db_id: str, db_schema: str) -> bool:
        # This is a simplified version. In a real LLM, it would analyze the schema and query.
        # For now, we'll just return False if no schema is provided.
        if not db_schema:
            return True
        return False

    def _prune(self, db_id: str, query: str, db_schema: str, db_fk: str, evidence: str) -> Dict:
        prompt = selector_template.format(db_id=db_id, desc_str=db_schema, fk_str=db_fk, query=query, evidence=evidence)
        response = self.llm.complete(prompt)
        reply = response.text.strip()
        return parse_json(reply)

    def talk(self, message):
        if message['send_to'] != self.name: return
        self._message = message
        db_id, ext_sch, query, evidence = message.get('db_id'), message.get('extracted_schema', {}), message.get('query'), message.get('evidence')
        use_gold_schema = False
        if ext_sch:
            use_gold_schema = True
        db_schema, db_fk, chosen_db_schem_dict = self._get_db_desc_str(db_id=db_id, extracted_schema=ext_sch, use_gold_schema=use_gold_schema)
        need_prune = self._is_need_prune(db_id, db_schema)
        if self.without_selector:
            need_prune = False
        if ext_sch == {} and need_prune:
            try:
                raw_extracted_schema_dict = self._prune(db_id=db_id, query=query, db_schema=db_schema, db_fk=db_fk, evidence=evidence)
            except Exception as e:
                logger.error(f"Prune failed: {e}")
                raw_extracted_schema_dict = {}
            db_schema_str, db_fk, chosen_db_schem_dict = self._get_db_desc_str(db_id=db_id, extracted_schema=raw_extracted_schema_dict)
            message['extracted_schema'] = raw_extracted_schema_dict
            message['chosen_db_schem_dict'] = chosen_db_schem_dict
            message['desc_str'] = db_schema_str
            message['fk_str'] = db_fk
            message['pruned'] = True
            message['send_to'] = DECOMPOSER_NAME
        else:
            message['chosen_db_schem_dict'] = chosen_db_schem_dict
            message['desc_str'] = db_schema
            message['fk_str'] = db_fk
            message['pruned'] = False
            message['send_to'] = DECOMPOSER_NAME

# Similarly embed Decomposer, Refiner, and ChatManager classes
class Decomposer:
    name = DECOMPOSER_NAME
    def __init__(self, llm: LLM, dataset_name: str):
        self.llm = llm
        self.dataset_name = dataset_name
        self._message = {}
    def talk(self, message):
        if message['send_to'] != self.name: return
        self._message = message
        query, evidence, schema_info, fk_info = message.get('query'), message.get('evidence'), message.get('desc_str'), message.get('fk_str')
        if self.dataset_name == 'bird':
            prompt = decompose_template_bird.format(query=query, desc_str=schema_info, fk_str=fk_info, evidence=evidence)
        else:
            prompt = decompose_template_spider.format(query=query, desc_str=schema_info, fk_str=fk_info)
        response = self.llm.complete(prompt)  # Adapt to Squrve's LLM
        reply = response.text.strip()
        res = parse_sql_from_string(reply)
        message['final_sql'] = res
        message['qa_pairs'] = reply
        message['fixed'] = False
        message['send_to'] = REFINER_NAME

class Refiner:
    name = REFINER_NAME
    def __init__(self, llm: LLM, data_path: str, dataset_name: str):
        self.llm = llm
        self.data_path = data_path
        self.dataset_name = dataset_name
        self._message = {}
    # Paste _execute_sql, _is_need_refine, _refine, talk from agents.py
    # Add @func_set_timeout if needed, but since it's not in Squrve, use try-except
    def _execute_sql(self, sql: str, db_id: str) -> dict:
        # Full code
        try:
            db_file = os.path.join(self.data_path, f"{db_id}.sqlite")
            if not os.path.exists(db_file):
                logger.warning(f"数据库文件不存在: {db_file}")
                return {"success": False, "error": "Database file not found"}

            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # 执行 SQL
            cursor.execute(sql)
            result = cursor.fetchall()
            
            # 获取列名
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            conn.close()
            
            return {
                "success": True,
                "result": result,
                "column_names": column_names,
                "row_count": len(result)
            }
        except Exception as e:
            logger.error(f"SQL 执行失败: {e}")
            return {"success": False, "error": str(e)}
    def _is_need_refine(self, exec_result: dict):
        # Full code
        if exec_result.get("success") and exec_result.get("row_count") == 0:
            return True
        return False
    def _refine(self, query: str, evidence:str, schema_info: str, fk_info: str, error_info: dict) -> str:
        # Use self.llm.complete for prompt
        prompt = refiner_template.format(desc_str=schema_info, fk_str=fk_info, query=query, evidence=evidence)
        response = self.llm.complete(prompt)
        reply = response.text.strip()
        return reply
    def talk(self, message):
        # Full code, adapting LLM calls
        if message['send_to'] != self.name: return
        self._message = message
        query, evidence, schema_info, fk_info = message.get('query'), message.get('evidence'), message.get('desc_str'), message.get('fk_str')
        if not query:
            message['send_to'] = SYSTEM_NAME
            return

        # 1. Decompose the question into sub-questions
        sub_questions = []
        try:
            prompt = decompose_template_bird.format(query=query, desc_str=schema_info, fk_str=fk_info, evidence=evidence)
            response = self.llm.complete(prompt)
            reply = response.text.strip()
            sub_questions = [q.strip() for q in reply.split('\n') if q.strip().startswith('Sub question')]
            if not sub_questions:
                message['send_to'] = SYSTEM_NAME # Fallback to system if no sub-questions
                return
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            message['send_to'] = SYSTEM_NAME # Fallback to system on error
            return

        # 2. Generate SQL for each sub-question
        final_sql = ""
        for i, sub_q in enumerate(sub_questions):
            message['send_to'] = DECOMPOSER_NAME # Pass to decomposer
            sub_q_sql = ""
            try:
                prompt = decompose_template_bird.format(query=sub_q, desc_str=schema_info, fk_str=fk_info, evidence=evidence)
                response = self.llm.complete(prompt)
                reply = response.text.strip()
                sub_q_sql = parse_sql_from_string(reply)
                if not sub_q_sql or sub_q_sql.lower().strip() == "error: no sql found in the input string":
                    message['send_to'] = SYSTEM_NAME # Fallback to system if SQL generation fails
                    return
            except Exception as e:
                logger.error(f"SQL generation for sub-question {i+1} failed: {e}")
                message['send_to'] = SYSTEM_NAME # Fallback to system on error
                return

            final_sql += f"Sub question {i+1}: {sub_q}\nSQL\n```sql\n{sub_q_sql}\n```\n\n"

        message['final_sql'] = final_sql
        message['qa_pairs'] = final_sql # Store SQL as qa_pairs for now
        message['fixed'] = False
        message['send_to'] = SYSTEM_NAME # Return to system

class ChatManager:
    def __init__(self, llm: LLM, data_path: str, tables_json_path: str, log_path: str, model_name: str, dataset_name:str, lazy: bool=False, without_selector: bool=False):
        self.llm = llm
        self.data_path = data_path
        self.tables_json_path = tables_json_path
        self.log_path = log_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.ping_network()
        self.chat_group = [
            Selector(llm=llm, data_path=data_path, tables_json_path=tables_json_path, model_name=model_name, dataset_name=dataset_name, lazy=lazy, without_selector=without_selector),
            Decomposer(llm=llm, dataset_name=dataset_name),
            Refiner(llm=llm, data_path=data_path, dataset_name=dataset_name)
        ]
        # INIT_LOG__PATH_FUNC(log_path) - adapt or remove if not needed
    def ping_network(self):
        # Simplified or remove
        pass
    def _chat_single_round(self, message):
        for agent in self.chat_group:
            if message['send_to'] == agent.name:
                agent.talk(message)
    def start(self, user_message):
        start_time = time.time()
        if user_message['send_to'] == SYSTEM_NAME:
            user_message['send_to'] = SELECTOR_NAME
        for _ in range(MAX_ROUND):
            self._chat_single_round(user_message)
            if user_message['send_to'] == SYSTEM_NAME:
                break
        end_time = time.time()
        exec_time = end_time - start_time
        logger.info(f"Execute {exec_time} seconds")

# Now, in MACSQLGenerator, use these embedded classes instead of importing

class MACSQLGenerator(BaseGenerator):
    """
    MAC-SQL Generator: Multi-Agent Collaborative SQL Generation
    Implements the MAC-SQL method for end-to-end Text-to-SQL generation
    """

    NAME = "MACSQLGenerator"
    OUTPUT_NAME = "pred_sql"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            max_round: int = 3,
            model_name: str = "gpt-4",
            without_selector: bool = False,
            **kwargs
    ):
        super().__init__()
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir
        self.max_round = max_round
        self.model_name = model_name
        self.without_selector = without_selector
        
        # 安全地初始化 db_path 和 credential
        if hasattr(dataset, 'db_path') and dataset.db_path:
            self.db_path = dataset.db_path
        else:
            self.db_path = None
            
        if hasattr(dataset, 'credential') and dataset.credential:
            self.credential = dataset.credential
        else:
            self.credential = None

    def _init_mac_sql_environment(self):
        """初始化 MAC-SQL 运行环境"""
        try:
            # 设置 MAC-SQL 的环境变量
            mac_sql_path = os.path.join(os.path.dirname(__file__), '../../../baselines/MAC-SQL')
            if mac_sql_path not in sys.path:
                sys.path.insert(0, mac_sql_path)
            
            # 导入 MAC-SQL 核心组件
            from core.chat_manager import ChatManager
            from core.const import SYSTEM_NAME, SELECTOR_NAME, DECOMPOSER_NAME, REFINER_NAME
            from core.agents import Selector, Decomposer, Refiner
            
            return ChatManager, Selector, Decomposer, Refiner
        except ImportError as e:
            logger.error(f"无法导入 MAC-SQL 组件: {e}")
            raise ImportError(f"MAC-SQL 组件导入失败: {e}")

    def _create_tables_json(self, schema: Union[Dict, List, pd.DataFrame]) -> Dict:
        """将 Squrve 的 schema 格式转换为 MAC-SQL 的 tables.json 格式"""
        if isinstance(schema, pd.DataFrame):
            schema_dict = schema.to_dict('records')
        elif isinstance(schema, list):
            schema_dict = schema
        else:
            schema_dict = schema

        # 构建 MAC-SQL 格式的 tables.json
        tables_json = {
            "db_id": "temp_db",  # 临时数据库ID
            "table_names": [],
            "column_names": [],
            "column_types": [],
            "foreign_keys": [],
            "primary_keys": []
        }

        # 按表名分组处理
        table_groups = {}
        for item in schema_dict:
            table_name = item.get('table_name', '')
            column_name = item.get('column_name', '')
            column_type = item.get('column_types', '')
            
            if table_name not in table_groups:
                table_groups[table_name] = []
                tables_json["table_names"].append(table_name)
            
            table_groups[table_name].append({
                'column_name': column_name,
                'column_type': column_type
            })

        # 构建 column_names 和 column_types
        for table_idx, table_name in enumerate(tables_json["table_names"]):
            columns = table_groups[table_name]
            for col_idx, col in enumerate(columns):
                tables_json["column_names"].append([table_idx, col['column_name']])
                tables_json["column_types"].append(col['column_type'])

        return tables_json

    def _create_chat_manager(self, db_path: str, tables_json: Dict) -> Any:
        """创建 MAC-SQL 的 ChatManager"""
        try:
            # 创建临时 tables.json 文件
            temp_tables_path = os.path.join(self.save_dir, "temp_tables.json")
            os.makedirs(os.path.dirname(temp_tables_path), exist_ok=True)
            with open(temp_tables_path, 'w', encoding='utf-8') as f:
                json.dump([tables_json], f, ensure_ascii=False, indent=2)

            # 创建 ChatManager
            chat_manager = ChatManager(
                llm=self.llm,
                data_path=db_path,
                tables_json_path=temp_tables_path,
                log_path="",
                model_name=self.model_name,
                dataset_name="spider",  # 默认使用 spider 格式
                lazy=True,
                without_selector=self.without_selector
            )
            
            return chat_manager, temp_tables_path
        except Exception as e:
            logger.error(f"创建 ChatManager 失败: {e}")
            raise

    def _init_user_message(self, item: Dict, schema: str, db_id: str) -> Dict:
        """初始化用户消息"""
        question = item.get('question', '')
        evidence = item.get('evidence', '')
        ground_truth = item.get('query', '')
        
        # 评估难度
        difficulty = self._evaluate_difficulty(item)
        
        user_message = {
            "idx": item.get('instance_id', 0),
            "db_id": db_id,
            "query": question,
            "evidence": evidence,
            "extracted_schema": {},
            "ground_truth": ground_truth,
            "difficulty": difficulty,
            "send_to": "System"
        }
        
        return user_message

    def _evaluate_difficulty(self, item: Dict) -> str:
        """评估查询难度"""
        # 简单的难度评估逻辑
        question = item.get('question', '')
        query = item.get('query', '')
        
        if not query:
            return 'easy'
        
        # 基于 SQL 复杂度评估难度
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ['union', 'intersect', 'except', 'with', 'cte']):
            return 'hard'
        elif any(keyword in query_lower for keyword in ['join', 'group by', 'having', 'subquery']):
            return 'medium'
        else:
            return 'easy'

    def _execute_sql_with_feedback(self, sql: str, db_id: str, db_path: str) -> Dict:
        """执行 SQL 并获取反馈"""
        try:
            db_file = os.path.join(db_path, f"{db_id}.sqlite")
            if not os.path.exists(db_file):
                logger.warning(f"数据库文件不存在: {db_file}")
                return {"success": False, "error": "Database file not found"}

            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # 执行 SQL
            cursor.execute(sql)
            result = cursor.fetchall()
            
            # 获取列名
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            conn.close()
            
            return {
                "success": True,
                "result": result,
                "column_names": column_names,
                "row_count": len(result)
            }
        except Exception as e:
            logger.error(f"SQL 执行失败: {e}")
            return {"success": False, "error": str(e)}

    def _refine_sql_with_llm(self, question: str, schema: str, sql: str, error_info: Dict) -> str:
        """使用 LLM 优化 SQL"""
        try:
            prompt = f"""You are an expert SQL developer. The following SQL query has an error and needs to be fixed.

Question: {question}

Database Schema:
{schema}

Original SQL: {sql}

Error: {error_info.get('error', 'Unknown error')}

Please provide the corrected SQL query. Only return the SQL statement without any explanation:"""

            response = self.llm.complete(prompt)
            refined_sql = response.text.strip()
            
            # 清理 SQL 输出
            if refined_sql.startswith('```sql'):
                refined_sql = refined_sql[6:]
            if refined_sql.endswith('```'):
                refined_sql = refined_sql[:-3]
            
            return refined_sql.strip()
        except Exception as e:
            logger.error(f"SQL 优化失败: {e}")
            return sql

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            **kwargs
    ):
        """实现 MAC-SQL 的端到端 SQL 生成逻辑"""
        logger.info(f"MACSQLGenerator 开始处理样本 {item}")
        
        # 获取数据样本
        row = self.dataset[item]
        question = row['question']
        db_id = row['db_id']
        db_type = row.get('db_type', 'sqlite')
        
        logger.debug(f"处理问题: {question[:100]}... (数据库: {db_id}, 类型: {db_type})")

        # 加载和处理 schema
        if isinstance(schema, (str, Path)):
            schema = load_dataset(schema)

        if schema is None:
            # 从数据集获取 schema
            schema = self.dataset.get_db_schema(item)
            if schema is None:
                raise Exception("无法获取有效的数据库模式!")

        # 标准化 schema 格式
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            schema_str = parse_schema_from_df(schema)
        else:
            raise Exception("无法处理数据库模式格式!")

        # 转换为 MAC-SQL 格式
        tables_json = self._create_tables_json(schema)
        tables_json["db_id"] = db_id

        # 设置数据库路径
        if self.db_path:
            db_path = self.db_path
        else:
            db_path = os.path.join(os.path.dirname(self.save_dir), "databases")

        # 创建 ChatManager
        try:
            chat_manager, temp_tables_path = self._create_chat_manager(db_path, tables_json)
        except Exception as e:
            logger.error(f"创建 ChatManager 失败: {e}")
            # 如果 MAC-SQL 组件不可用，使用简单的 LLM 生成
            return self._fallback_sql_generation(question, schema_str, row)

        # 初始化用户消息
        user_message = self._init_user_message(row, schema_str, db_id)

        # 执行 MAC-SQL 流程
        try:
            logger.debug("开始 MAC-SQL 多智能体协作生成...")
            chat_manager.start(user_message)
            
            # 获取生成的 SQL
            pred_sql = user_message.get('pred', '')
            if not pred_sql:
                raise Exception("MAC-SQL 未生成有效的 SQL")

            logger.debug(f"MAC-SQL 生成完成: {pred_sql[:100]}...")

        except Exception as e:
            logger.error(f"MAC-SQL 执行失败: {e}")
            # 回退到简单生成
            pred_sql = self._fallback_sql_generation(question, schema_str, row)

        # SQL 后处理和验证
        pred_sql = self._post_process_sql(pred_sql, question, schema_str, db_id, db_path)

        # 保存结果
        if self.is_save:
            instance_id = row.get("instance_id")
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            save_path = save_path / f"{self.NAME}_{instance_id}.sql"

            save_dataset(pred_sql, new_data_source=save_path)
            self.dataset.setitem(item, "pred_sql", str(save_path))
            logger.debug(f"SQL 已保存到: {save_path}")

        # 清理临时文件
        try:
            if 'temp_tables_path' in locals():
                os.remove(temp_tables_path)
        except:
            pass

        logger.info(f"MACSQLGenerator 样本 {item} 处理完成")
        return pred_sql

    def _fallback_sql_generation(self, question: str, schema: str, row: Dict) -> str:
        """回退的 SQL 生成方法"""
        logger.warning("使用回退 SQL 生成方法")
        
        prompt = f"""You are an expert SQL developer. Please generate a SQL query for the following question.

Question: {question}

Database Schema:
{schema}

Please generate a valid SQL query. Only return the SQL statement without any explanation:"""

        try:
            response = self.llm.complete(prompt)
            sql = response.text.strip()
            
            # 清理 SQL 输出
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
            
            return sql.strip()
        except Exception as e:
            logger.error(f"回退 SQL 生成失败: {e}")
            return "SELECT 1"  # 返回默认 SQL

    def _post_process_sql(self, sql: str, question: str, schema: str, db_id: str, db_path: str) -> str:
        """SQL 后处理和验证"""
        if not sql or sql.strip() == "":
            return self._fallback_sql_generation(question, schema, {"question": question})

        # 执行 SQL 验证
        exec_result = self._execute_sql_with_feedback(sql, db_id, db_path)
        
        if not exec_result["success"]:
            logger.warning(f"SQL 执行失败，尝试优化: {exec_result.get('error', 'Unknown error')}")
            # 使用 LLM 优化 SQL
            refined_sql = self._refine_sql_with_llm(question, schema, sql, exec_result)
            
            # 再次验证
            refined_result = self._execute_sql_with_feedback(refined_sql, db_id, db_path)
            if refined_result["success"]:
                logger.info("SQL 优化成功")
                return refined_sql
            else:
                logger.warning("SQL 优化后仍失败，返回原始 SQL")
                return sql
        
        return sql 