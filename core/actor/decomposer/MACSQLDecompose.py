# core/actor/decomposer/MACSQLDecompose.py
from llama_index.core.llms.llm import LLM
from typing import Union, List, Dict, Tuple
import pandas as pd
from os import PathLike
from pathlib import Path
import re
from loguru import logger

from core.data_manage import Dataset, single_central_process
from core.actor.decomposer.BaseDecompose import BaseDecomposer
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)


# Import for database connection (for future use if needed)
# from core.db_connect import get_sql_exec_result, execute_sql

@BaseDecomposer.register_actor
class MACSQLDecomposer(BaseDecomposer):
    """
    MAC-SQL Decomposer: Decomposes complex queries into sub-questions with corresponding SQL statements.
    """

    NAME = "MACSQLDecomposer"

    SKILL = """# MACSQLDecomposer

MACSQLDecomposer produces (sub_question, sql) pairs in one shot via BIRD-style template; evidence from row enriches the prompt. Advantage: joint sub-question and SQL output; drawback: BIRD-specific.

## Inputs
- `schema`: Database schema (str/path/dict/list). If absent, loaded from dataset.

## Output
`sub_questions`

## Steps
1. Process schema to BIRD format (desc_str).
2. Load question, evidence from row.
3. generate_decomposition: LLM with DECOMPOSE_TEMPLATE_BIRD → parse_qa_pairs.
4. Save and return sub_questions.
"""

    # Prompt templates from MAC-SQL
    DECOMPOSE_TEMPLATE_BIRD = '''Given a 【Database schema】 description, a knowledge 【Evidence】 and the 【Question】, you need to use valid SQLite and understand the database and knowledge, and then decompose the question into subquestions for text-to-SQL generation.
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

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Union[LLM, List[LLM]] = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/sub_questions",
            dataset_name: str = "bird",  # or "spider"
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, **kwargs)
        self.dataset_name = dataset_name

    def parse_qa_pairs(self, response: str) -> List[Tuple[str, str]]:
        qa_pairs = []
        sub_parts = re.split(r'Sub question \d+:', response)
        for part in sub_parts[1:]:
            if 'SQL' in part:
                sub_q = part.split('SQL')[0].strip()
                sql_match = re.search(r'```sql(.*?)```', part, re.DOTALL)
                sub_sql = sql_match.group(1).strip() if sql_match else ""
                qa_pairs.append((sub_q, sub_sql))
        return qa_pairs

    def generate_decomposition(
            self, llm_: LLM, query: str, desc_str: str, fk_str: str, evidence: str,
            data_logger=None
    ) -> List[Tuple[str, str]]:
        if data_logger:
            data_logger.info(f"{self.NAME}.generate_decomposition input | question={query}")
        prompt = self.DECOMPOSE_TEMPLATE_BIRD.format(query=query, desc_str=desc_str, fk_str=fk_str,
                                                     evidence=evidence)

        response = llm_.complete(prompt).text.strip()
        response = self.parse_qa_pairs(response)
        if data_logger:
            data_logger.info(f"{self.NAME}.generate_sub_questions output | sub_questions={response}")
        return response

    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, data_logger=None, **kwargs):
        """
        Decompose complex queries into sub-questions with corresponding SQL statements.
        
        Returns:
            List[Tuple[str, str]]: List of (sub_question, corresponding_sql) pairs.
            Note: This differs from DINSQLDecomposer which returns List[str] of sub_questions only.
        """
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        row = self.dataset[item]
        query = row.get("question", "")
        evidence = row.get("evidence", "")
        db_id = row.get("db_id", "")

        # Use base class method to process schema
        desc_str = self.process_schema(item, schema)
        fk_str = ""  # TODO: Implement foreign key extraction if needed, or assume it's part of desc_str

        # Use base class method to get LLM
        llm = self.get_llm()
        if llm is None:
            # 如果没有有效的 LLM，返回空结果
            return []

        # Generate decomposition results
        sub_questions = self.generate_decomposition(llm, query, desc_str, fk_str, evidence, data_logger=data_logger)

        # Use base class method to save output
        self.save_output(sub_questions, item, db_id=db_id)
        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return sub_questions
