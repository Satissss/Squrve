import json
import time
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from loguru import logger
import re

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, load_dataset, save_dataset
from core.utils import parse_schema_from_df
from core.data_manage import single_central_process
from core.db_connect import get_sql_exec_result
from llama_index.core.llms.llm import LLM

# MAC-SQL 常量
MAX_ROUND = 3
SELECTOR_NAME = 'Selector'
DECOMPOSER_NAME = 'Decomposer'
REFINER_NAME = 'Refiner'
SYSTEM_NAME = 'System'


# 工具函数
def parse_json(text: str) -> dict:
    """解析 JSON 格式的文本"""
    # 查找 JSON 块
    start = text.find("```json")
    end = text.find("```", start + 7)
    
    if start != -1 and end != -1:
        json_string = text[start + 7: end].strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}")
            return {}
    
    # 如果没有找到 JSON 块，尝试直接解析整个文本
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    return {}


def parse_sql_from_string(input_string: str) -> str:
    """从字符串中提取 SQL"""
    sql_pattern = r'```sql(.*?)```'
    all_sqls = []
    for match in re.finditer(sql_pattern, input_string, re.DOTALL):
        all_sqls.append(match.group(1).strip())
    if all_sqls:
        return all_sqls[-1]
    else:
        return "error: No SQL found in the input string"


def load_json_file(file_path: str) -> dict:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# MAC-SQL 提示模板
selector_template = '''
As an experienced and professional database administrator, your task is to analyze a user question and a database schema to provide relevant information. The database schema consists of table descriptions, each containing multiple column descriptions. Your goal is to identify the relevant tables and columns based on the user question and evidence provided.

[Instruction]:
1. Discard any table schema that is not related to the user question and evidence.
2. Sort the columns in each relevant table in descending order of relevance and keep the top 6 columns.
3. Ensure that at least 3 tables are included in the final output JSON.
4. The output should be in JSON format.

Requirements:
1. If a table has less than or equal to 10 columns, mark it as "keep_all".
2. If a table is completely irrelevant to the user question and evidence, mark it as "drop_all".
3. Prioritize the columns in each relevant table based on their relevance.

Here is a typical example:

==========
【DB_ID】 banking_system
【Schema】
# Table: account
[
  (account_id, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),
  (date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].),
]
# Table: client
[
  (client_id, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),
  (gender, gender. Value examples: ['M', 'F']. And F：female . M：male ),
  (birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),
]
# Table: loan
[
  (loan_id, the id number identifying the loan data. Value examples: [4959, 4960, 4961].),
  (account_id, the id number identifying the account. Value examples: [10, 80, 55, 43].),
  (date, the date when the loan is approved. Value examples: ['1998-07-12', '1998-04-19'].),
  (amount, the id number identifying the loan data. Value examples: [1567, 7877, 9988].),
  (duration, the id number identifying the loan data. Value examples: [60, 48, 24, 12, 36].),
  (payments, the id number identifying the loan data. Value examples: [3456, 8972, 9845].),
  (status, the id number identifying the loan data. Value examples: ['C', 'A', 'D', 'B'].)
]
# Table: district
[
  (district_id, location of branch. Value examples: [77, 76].),
  (A2, area in square kilometers. Value examples: [50.5, 48.9].),
  (A4, number of inhabitants. Value examples: [95907, 95616].),
  (A5, number of households. Value examples: [35678, 34892].),
  (A6, literacy rate. Value examples: [95.6, 92.3, 89.7].),
  (A7, number of entrepreneurs. Value examples: [1234, 1456].),
  (A8, number of cities. Value examples: [5, 4].),
  (A9, number of schools. Value examples: [15, 12, 10].),
  (A10, number of hospitals. Value examples: [8, 6, 4].),
  (A11, average salary. Value examples: [12541, 11277].),
  (A12, poverty rate. Value examples: [12.4, 9.8].),
  (A13, unemployment rate. Value examples: [8.2, 7.9].),
  (A15, number of crimes. Value examples: [256, 189].)
]
【Foreign keys】
client.`district_id` = district.`district_id`
【Question】
What is the gender of the youngest client who opened account in the lowest average salary branch?
【Evidence】
Later birthdate refers to younger age; A11 refers to average salary
【Answer】
```json
{{
  "account": "keep_all",
  "client": "keep_all",
  "loan": "drop_all",
  "district": ["district_id", "A11", "A2", "A4", "A6", "A7"]
}}
```
Question Solved.

==========

Here is a new example, please start answering:

【DB_ID】 {db_id}
【Schema】
{desc_str}
【Foreign keys】
{fk_str}
【Question】
{query}
【Evidence】
{evidence}
【Answer】
'''

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

decompose_template_spider = '''Given a 【Database schema】 description, and the 【Question】, you need to use valid SQLite and understand the database, and then generate the corresponding SQL.

==========

【Database schema】
# Table: stadium
[
  (Stadium_ID, stadium id. Value examples: [1, 2, 3, 4, 5, 6].),
  (Location, location. Value examples: ['Stirling Albion', 'Raith Rovers', "Queen's Park", 'Peterhead', 'East Fife', 'Brechin City'].),
  (Name, name. Value examples: ["Stark's Park", 'Somerset Park', 'Recreation Park', 'Hampden Park', 'Glebe Park', 'Gayfield Park'].),
  (Capacity, capacity. Value examples: [52500, 11998, 10104, 4125, 4000, 3960].),
  (Highest, highest. Value examples: [4812, 2363, 1980, 1763, 1125, 1057].),
  (Lowest, lowest. Value examples: [1294, 1057, 533, 466, 411, 404].),
  (Average, average. Value examples: [2106, 1477, 864, 730, 642, 638].)
]
# Table: concert
[
  (concert_ID, concert id. Value examples: [1, 2, 3, 4, 5, 6].),
  (concert_Name, concert name. Value examples: ['Week 1', 'Week 2', 'Super bootcamp', 'Home Visits', 'Auditions'].),
  (Theme, theme. Value examples: ['Wide Awake', 'Party All Night', 'Happy Tonight', 'Free choice 2', 'Free choice', 'Bleeding Love'].),
  (Stadium_ID, stadium id. Value examples: ['2', '9', '7', '10', '1'].),
  (Year, year. Value examples: ['2015', '2014'].)
]
【Foreign keys】
concert.`Stadium_ID` = stadium.`Stadium_ID`
【Question】
Show the stadium name and the number of concerts in each stadium.

SQL
```sql
SELECT T1.`Name`, COUNT(*) FROM stadium AS T1 JOIN concert AS T2 ON T1.`Stadium_ID` = T2.`Stadium_ID` GROUP BY T1.`Stadium_ID`
```

Question Solved.

==========

【Database schema】
# Table: singer
[
  (Singer_ID, singer id. Value examples: [1, 2].),
  (Name, name. Value examples: ['Tribal King', 'Timbaland'].),
  (Country, country. Value examples: ['France', 'United States', 'Netherlands'].),
  (Song_Name, song name. Value examples: ['You', 'Sun', 'Love', 'Hey Oh'].),
  (Song_release_year, song release year. Value examples: ['2016', '2014'].),
  (Age, age. Value examples: [52, 43].)
]
# Table: concert
[
  (concert_ID, concert id. Value examples: [1, 2].),
  (concert_Name, concert name. Value examples: ['Super bootcamp', 'Home Visits', 'Auditions'].),
  (Theme, theme. Value examples: ['Wide Awake', 'Party All Night'].),
  (Stadium_ID, stadium id. Value examples: ['2', '9'].),
  (Year, year. Value examples: ['2015', '2014'].)
]
# Table: singer_in_concert
[
  (concert_ID, concert id. Value examples: [1, 2].),
  (Singer_ID, singer id. Value examples: ['3', '6'].)
]
【Foreign keys】
singer_in_concert.`Singer_ID` = singer.`Singer_ID`
singer_in_concert.`concert_ID` = concert.`concert_ID`
【Question】
Show the name and the release year of the song by the youngest singer.

SQL
```sql
SELECT `Song_Name`, `Song_release_year` FROM singer WHERE Age = (SELECT MIN(Age) FROM singer)
```

Question Solved.

==========

【Database schema】
{desc_str}
【Foreign keys】
{fk_str}
【Question】
{query}

SQL

'''

refiner_template = '''【Instruction】
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
【Constraints】
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
【Query】
-- {query}
【Evidence】
{evidence}
【Database info】
{desc_str}
【Foreign keys】
{fk_str}
【old SQL】
```sql
{original_sql}
```
【SQLite error】 
{error}

Now please fixup old SQL and generate new SQL again.
【correct SQL】
'''


class Selector:
    """MAC-SQL Selector Agent: 负责数据库模式选择和剪枝"""
    name = SELECTOR_NAME

    def __init__(self, llm: LLM, dataset_name: str, without_selector: bool = False):
        self.llm = llm
        self.dataset_name = dataset_name
        self.without_selector = without_selector
        self._message = {}

    def _parse_schema_to_bird_format(self, schema: pd.DataFrame) -> Tuple[str, str]:
        """将 DataFrame 格式的 schema 转换为 BIRD 格式的字符串"""
        desc_str = ""
        fk_str = ""

        # 按表名分组
        table_groups = {}
        foreign_keys = []

        for _, row in schema.iterrows():
            table_name = row.get('table_name', '')
            column_name = row.get('column_name', '')
            column_type = row.get('column_type', '')

            if table_name not in table_groups:
                table_groups[table_name] = []

            # 构建列描述
            col_desc = f"  ({column_name}, {column_name}"
            if column_type:
                col_desc += f". Type: {column_type}"
            col_desc += ".),\n"

            table_groups[table_name].append(col_desc)

            # 处理外键
            if 'foreign_key' in row and pd.notna(row['foreign_key']):
                fk_info = row['foreign_key']
                if isinstance(fk_info, str) and '=' in fk_info:
                    foreign_keys.append(fk_info)

        # 构建描述字符串
        for table_name, columns in table_groups.items():
            desc_str += f"# Table: {table_name}\n[\n"
            desc_str += "".join(columns)
            desc_str = desc_str.rstrip(",\n") + "\n]\n"

        # 构建外键字符串
        fk_str = "\n".join(set(foreign_keys))

        return desc_str.strip(), fk_str.strip()

    def _is_need_prune(self, schema_str: str) -> bool:
        """判断是否需要进行模式剪枝"""
        if self.without_selector:
            return False

        # 更精确的启发式规则，基于原始实现
        table_count = schema_str.count("# Table:")
        
        # 计算总列数（更精确的计算）
        column_count = 0
        lines = schema_str.split('\n')
        for line in lines:
            if line.strip().startswith('(') and ',' in line:
                column_count += 1

        # 基于原始实现的判断逻辑
        if table_count <= 3:
            return False
            
        # 如果平均每表列数 <= 6 且总列数 <= 30，则不需要剪枝
        avg_columns = column_count / table_count if table_count > 0 else 0
        if avg_columns <= 6 and column_count <= 30:
            return False
            
        return True

    def _prune_schema(self, db_id: str, query: str, schema_str: str, fk_str: str, evidence: str) -> Dict:
        """使用 LLM 进行模式剪枝"""
        try:
            prompt = selector_template.format(
                db_id=db_id,
                desc_str=schema_str,
                fk_str=fk_str,
                query=query,
                evidence=evidence
            )
            response = self.llm.complete(prompt)
            return parse_json(response.text)
        except Exception as e:
            logger.error(f"模式剪枝失败: {e}")
            return {}

    def talk(self, message: Dict):
        """Selector 智能体的主要逻辑"""
        if message.get('send_to') != self.name:
            return

        self._message = message
        db_id = message.get('db_id', '')
        query = message.get('query', '')
        evidence = message.get('evidence', '')
        schema = message.get('schema')

        if schema is None:
            logger.error("Schema 信息缺失")
            message['send_to'] = SYSTEM_NAME
            return

        # 转换 schema 格式
        if isinstance(schema, pd.DataFrame):
            desc_str, fk_str = self._parse_schema_to_bird_format(schema)
        else:
            logger.error("不支持的 schema 格式")
            message['send_to'] = SYSTEM_NAME
            return

        # 判断是否需要剪枝
        need_prune = self._is_need_prune(desc_str)

        if need_prune:
            logger.debug("开始模式剪枝...")
            extracted_schema = self._prune_schema(db_id, query, desc_str, fk_str, evidence)
            message['extracted_schema'] = extracted_schema
            message['pruned'] = True
            logger.debug(f"剪枝结果: {extracted_schema}")
        else:
            message['extracted_schema'] = {}
            message['pruned'] = False

        message['desc_str'] = desc_str
        message['fk_str'] = fk_str
        message['send_to'] = DECOMPOSER_NAME


class Decomposer:
    """MAC-SQL Decomposer Agent: 负责问题分解和 SQL 生成"""
    name = DECOMPOSER_NAME

    def __init__(self, llm: LLM, dataset_name: str):
        self.llm = llm
        self.dataset_name = dataset_name
        self._message = {}

    def talk(self, message: Dict):
        """Decomposer 智能体的主要逻辑"""
        if message.get('send_to') != self.name:
            return

        self._message = message
        query = message.get('query', '')
        evidence = message.get('evidence', '')
        desc_str = message.get('desc_str', '')
        fk_str = message.get('fk_str', '')

        if not query or not desc_str:
            logger.error("缺少必要的查询或模式信息")
            message['send_to'] = SYSTEM_NAME
            return

        # 选择合适的模板
        if self.dataset_name == 'bird':
            template = decompose_template_bird
            prompt = template.format(
                desc_str=desc_str,
                fk_str=fk_str,
                query=query,
                evidence=evidence
            )
        else:
            template = decompose_template_spider
            prompt = template.format(
                desc_str=desc_str,
                fk_str=fk_str,
                query=query
            )

        try:
            logger.debug("开始问题分解和 SQL 生成...")
            response = self.llm.complete(prompt)
            reply = response.text.strip()

            # 提取最终的 SQL
            final_sql = parse_sql_from_string(reply)

            message['final_sql'] = final_sql
            message['qa_pairs'] = reply
            message['fixed'] = False
            message['send_to'] = REFINER_NAME

            logger.debug(f"生成的 SQL: {final_sql[:100]}...")

        except Exception as e:
            logger.error(f"问题分解失败: {e}")
            message['final_sql'] = "error: Failed to generate SQL"
            message['send_to'] = SYSTEM_NAME


class Refiner:
    """MAC-SQL Refiner Agent: 负责 SQL 执行验证和修复"""
    name = REFINER_NAME

    def __init__(self, llm: LLM, dataset_name: str):
        self.llm = llm
        self.dataset_name = dataset_name
        self._message = {}

    def _execute_sql(self, sql: str, db_type: str, db_path: str, db_id: str, credential: Dict = None) -> Dict:
        """使用 Squrve 框架的 get_sql_exec_result 执行 SQL"""
        try:
            # 构建参数
            exec_args = {
                "sql_query": sql,
                "db_path": db_path,
                "db_id": db_id,
                "credential_path": credential
            }

            # 执行 SQL
            result = get_sql_exec_result(db_type, **exec_args)

            if isinstance(result, tuple):
                if len(result) >= 2:
                    data, error = result[0], result[1]
                    if error:
                        return {"success": False, "error": str(error)}
                    elif data is None or (hasattr(data, 'empty') and data.empty):
                        return {"success": True, "result": [], "row_count": 0}
                    else:
                        return {"success": True, "result": data,
                                "row_count": len(data) if hasattr(data, '__len__') else 1}
                else:
                    return {"success": False, "error": "执行结果格式异常"}
            else:
                return {"success": False, "error": "执行结果格式异常"}

        except Exception as e:
            logger.error(f"SQL 执行失败: {e}")
            return {"success": False, "error": str(e)}

    def _is_need_refine(self, exec_result: Dict) -> bool:
        """判断是否需要修复 SQL"""
        if not exec_result.get("success", False):
            return True

        # 对于 Spider 数据集，即使结果为空也不一定需要修复
        if self.dataset_name == 'spider':
            return False

        # 对于其他数据集，如果结果为空则需要修复
        row_count = exec_result.get("row_count", 0)
        return row_count == 0

    def _refine_sql(self, query: str, evidence: str, desc_str: str, fk_str: str, original_sql: str, error: str) -> str:
        """使用 LLM 修复 SQL"""
        try:
            prompt = refiner_template.format(
                desc_str=desc_str,
                fk_str=fk_str,
                query=query,
                evidence=evidence,
                original_sql=original_sql,
                error=error
            )

            response = self.llm.complete(prompt)
            reply = response.text.strip()
            return parse_sql_from_string(reply)

        except Exception as e:
            logger.error(f"SQL 修复失败: {e}")
            return original_sql

    def talk(self, message: Dict):
        """Refiner 智能体的主要逻辑"""
        if message.get('send_to') != self.name:
            return

        self._message = message
        db_id = message.get('db_id', '')
        db_type = message.get('db_type', 'sqlite')
        db_path = message.get('db_path', '')
        credential = message.get('credential')
        final_sql = message.get('final_sql', '')
        query = message.get('query', '')
        evidence = message.get('evidence', '')
        desc_str = message.get('desc_str', '')
        fk_str = message.get('fk_str', '')

        # 如果 SQL 包含错误信息，直接返回
        if 'error' in final_sql.lower():
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = final_sql
            message['send_to'] = SYSTEM_NAME
            return

        # 执行 SQL
        logger.debug("开始执行 SQL 验证...")
        exec_result = self._execute_sql(final_sql, db_type, db_path, db_id, credential)

        # 判断是否需要修复
        need_refine = self._is_need_refine(exec_result)

        if not need_refine:
            # SQL 执行成功，无需修复
            message['try_times'] = message.get('try_times', 0) + 1
            message['pred'] = final_sql
            message['send_to'] = SYSTEM_NAME
            logger.debug("SQL 执行成功，无需修复")
        else:
            # 需要修复 SQL
            try_times = message.get('try_times', 0)
            if try_times >= MAX_ROUND - 1:
                # 达到最大尝试次数，返回原始 SQL
                message['try_times'] = try_times + 1
                message['pred'] = final_sql
                message['send_to'] = SYSTEM_NAME
                logger.warning(f"达到最大修复次数 {MAX_ROUND}，返回原始 SQL")
            else:
                # 尝试修复 SQL
                logger.debug("开始修复 SQL...")
                error_info = exec_result.get('error', 'Empty result set')
                refined_sql = self._refine_sql(query, evidence, desc_str, fk_str, final_sql, error_info)

                message['try_times'] = try_times + 1
                message['final_sql'] = refined_sql
                message['fixed'] = True
                message['send_to'] = REFINER_NAME
                logger.debug(f"SQL 修复完成: {refined_sql[:100]}...")


class ChatManager:
    """MAC-SQL ChatManager: 管理三个智能体之间的协作"""

    def __init__(self, llm: LLM, dataset_name: str, without_selector: bool = False):
        self.llm = llm
        self.dataset_name = dataset_name
        self.chat_group = [
            Selector(llm=llm, dataset_name=dataset_name, without_selector=without_selector),
            Decomposer(llm=llm, dataset_name=dataset_name),
            Refiner(llm=llm, dataset_name=dataset_name)
        ]

    def _chat_single_round(self, message: Dict):
        """执行单轮对话"""
        for agent in self.chat_group:
            if message.get('send_to') == agent.name:
                agent.talk(message)
                break

    def start(self, user_message: Dict):
        """开始多智能体协作"""
        start_time = time.time()

        if user_message.get('send_to') == SYSTEM_NAME:
            user_message['send_to'] = SELECTOR_NAME

        for round_num in range(MAX_ROUND):
            logger.debug(f"开始第 {round_num + 1} 轮对话，当前发送到: {user_message.get('send_to')}")
            self._chat_single_round(user_message)

            if user_message.get('send_to') == SYSTEM_NAME:
                logger.debug("对话结束")
                break

        end_time = time.time()
        exec_time = end_time - start_time
        logger.info(f"MAC-SQL 协作完成，耗时: {exec_time:.2f} 秒")


class MACSQLGenerator(BaseGenerator):
    """
    MAC-SQL Generator: Multi-Agent Collaborative SQL Generation
    实现 MAC-SQL 方法的端到端 Text-to-SQL 生成
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
            dataset_name: str = "spider",
            without_selector: bool = False,
            db_path: Optional[Union[str, Path]] = None,
            credential: Optional[Dict] = None,
            **kwargs
    ):
        super().__init__()
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = Path(save_dir)
        self.max_round = max_round
        self.dataset_name = dataset_name
        self.without_selector = without_selector

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

    def _validate_inputs(self, item, schema) -> Tuple[bool, str]:
        """验证输入参数的有效性"""
        if self.dataset is None:
            return False, "Dataset 未初始化"

        if self.llm is None:
            return False, "LLM 未初始化"

        try:
            row = self.dataset[item]
            if 'question' not in row:
                return False, "数据样本缺少 'question' 字段"
            if 'db_id' not in row:
                return False, "数据样本缺少 'db_id' 字段"
        except Exception as e:
            return False, f"无法访问数据样本: {e}"

        return True, ""

    def _prepare_schema(self, item, schema) -> pd.DataFrame:
        """准备和标准化 schema"""
        # 加载 schema
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

        if not isinstance(schema, pd.DataFrame):
            raise Exception("无法处理数据库模式格式!")

        return schema

    def _prepare_database_info(self, row) -> Tuple[str, str, str, Dict]:
        """准备数据库连接信息"""
        db_id = row['db_id']
        db_type = row.get('db_type', 'sqlite')

        # 设置数据库路径
        if self.db_path:
            if db_type == 'sqlite':
                db_path = str(Path(self.db_path) / f"{db_id}.sqlite")
            else:
                db_path = str(self.db_path)
        else:
            db_path = ""

        credential = self.credential if self.credential else {}

        return db_id, db_type, db_path, credential

    def _fallback_sql_generation(self, question: str, schema_str: str) -> str:
        """回退的 SQL 生成方法"""
        logger.warning("使用回退 SQL 生成方法")

        prompt = f"""You are an expert SQL developer. Please generate a SQL query for the following question.

Question: {question}

Database Schema:
{schema_str}

Please generate a valid SQL query. Only return the SQL statement:"""

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

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ):
        """实现 MAC-SQL 的端到端 SQL 生成逻辑"""
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        logger.info(f"MACSQLGenerator 开始处理样本 {item}")

        # 验证输入
        is_valid, error_msg = self._validate_inputs(item, schema)
        if not is_valid:
            logger.error(f"输入验证失败: {error_msg}")
            raise Exception(error_msg)

        try:
            # 获取数据样本
            row = self.dataset[item]
            question = row['question']
            evidence = row.get('evidence', '')

            logger.debug(f"处理问题: {question[:100]}...")

            # 准备 schema
            schema_df = self._prepare_schema(item, schema)

            # 准备数据库信息
            db_id, db_type, db_path, credential = self._prepare_database_info(row)

            logger.debug(f"数据库信息: db_id={db_id}, db_type={db_type}")

            # 创建 ChatManager
            chat_manager = ChatManager(
                llm=self.llm,
                dataset_name=self.dataset_name,
                without_selector=self.without_selector
            )

            # 初始化用户消息
            user_message = {
                "idx": row.get('instance_id', item),
                "db_id": db_id,
                "db_type": db_type,
                "db_path": db_path,
                "credential": credential,
                "query": question,
                "evidence": evidence,
                "schema": schema_df,
                "extracted_schema": {},
                "ground_truth": row.get('query', ''),
                "send_to": SYSTEM_NAME
            }

            # 执行 MAC-SQL 流程
            logger.debug("开始 MAC-SQL 多智能体协作...")
            chat_manager.start(user_message)

            # 获取生成的 SQL
            pred_sql = user_message.get('pred', user_message.get('final_sql', ''))

            if not pred_sql or pred_sql.strip() == "":
                logger.warning("MAC-SQL 未生成有效的 SQL，使用回退方法")
                schema_str = parse_schema_from_df(schema_df)
                pred_sql = self._fallback_sql_generation(question, schema_str)

            logger.debug(f"最终生成的 SQL: {pred_sql[:100]}...")

            # 保存结果
            if self.is_save:
                instance_id = row.get("instance_id", item)
                save_path = self.save_dir
                if self.dataset.dataset_index:
                    save_path = save_path / str(self.dataset.dataset_index)
                save_path = save_path / f"{self.NAME}_{instance_id}.sql"

                save_dataset(pred_sql, new_data_source=save_path)
                self.dataset.setitem(item, "pred_sql", str(save_path))
                logger.debug(f"SQL 已保存到: {save_path}")

            logger.info(f"MACSQLGenerator 样本 {item} 处理完成")
            if data_logger:
                data_logger.info(f"{self.NAME}.final_sql | sql={pred_sql}")
                data_logger.info(f"{self.NAME}.act end | item={item}")
            return pred_sql

        except Exception as e:
            logger.error(f"MACSQLGenerator 处理失败: {e}")
            # 尝试回退方法
            try:
                row = self.dataset[item]
                schema_df = self._prepare_schema(item, schema)
                schema_str = parse_schema_from_df(schema_df)
                return self._fallback_sql_generation(row['question'], schema_str)
            except Exception as fallback_e:
                logger.error(f"回退方法也失败: {fallback_e}")
                raise Exception(f"MAC-SQL 生成失败: {e}")
