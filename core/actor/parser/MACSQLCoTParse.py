import json
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict

from core.actor.parser.BaseParse import BaseParser
from core.utils import load_dataset

class MACSQLCoTParser(BaseParser):
    NAME = "MACSQLCoTParser"
    OUTPUT_NAME = "schema_links"

    def __init__(
        self,
        dataset=None,
        llm=None,
        is_save: bool = True,
        save_dir: Union[str, Path] = "../files/schema_links",
        **kwargs
    ):
        super().__init__()
        self.dataset = dataset
        self.llm = llm
        self.is_save = is_save
        self.save_dir = save_dir

    selector_template = '''\nAs an experienced and professional database administrator, your task is to analyze a user question and a database schema to provide relevant information. The database schema consists of table descriptions, each containing multiple column descriptions. Your goal is to identify the relevant tables and columns based on the user question and evidence provided.\n\n[Instruction]:\n1. Discard any table schema that is not related to the user question and evidence.\n2. Sort the columns in each relevant table in descending order of relevance and keep the top 6 columns.\n3. Ensure that at least 3 tables are included in the final output JSON.\n4. The output should be in JSON format.\n\nRequirements:\n1. If a table has less than or equal to 10 columns, mark it as \"keep_all\".\n2. If a table is completely irrelevant to the user question and evidence, mark it as \"drop_all\".\n3. Prioritize the columns in each relevant table based on their relevance.\n\nHere is a typical example:\n\n==========\n【DB_ID】 banking_system\n【Schema】\n# Table: account\n[\n  (account_id, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),\n  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].),\n  (frequency, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),\n  (date, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)\n]\n# Table: client\n[\n  (client_id, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),\n  (gender, gender. Value examples: ['M', 'F']. And F：female . M：male ),\n  (birth_date, birth date. Value examples: ['1987-09-27', '1986-08-13'].),\n  (district_id, location of branch. Value examples: [77, 76, 2, 1, 39].)\n]\n# Table: loan\n[\n  (loan_id, the id number identifying the loan data. Value examples: [4959, 4960, 4961].),\n  (account_id, the id number identifying the account. Value examples: [10, 80, 55, 43].),\n  (date, the date when the loan is approved. Value examples: ['1998-07-12', '1998-04-19'].),\n  (amount, the id number identifying the loan data. Value examples: [1567, 7877, 9988].),\n  (duration, the id number identifying the loan data. Value examples: [60, 48, 24, 12, 36].),\n  (payments, the id number identifying the loan data. Value examples: [3456, 8972, 9845].),\n  (status, the id number identifying the loan data. Value examples: ['C', 'A', 'D', 'B'].)\n]\n# Table: district\n[\n  (district_id, location of branch. Value examples: [77, 76].),\n  (A2, area in square kilometers. Value examples: [50.5, 48.9].),\n  (A4, number of inhabitants. Value examples: [95907, 95616].),\n  (A5, number of households. Value examples: [35678, 34892].),\n  (A6, literacy rate. Value examples: [95.6, 92.3, 89.7].),\n  (A7, number of entrepreneurs. Value examples: [1234, 1456].),\n  (A8, number of cities. Value examples: [5, 4].),\n  (A9, number of schools. Value examples: [15, 12, 10].),\n  (A10, number of hospitals. Value examples: [8, 6, 4].),\n  (A11, average salary. Value examples: [12541, 11277].),\n  (A12, poverty rate. Value examples: [12.4, 9.8].),\n  (A13, unemployment rate. Value examples: [8.2, 7.9].),\n  (A15, number of crimes. Value examples: [256, 189].)\n]\n【Foreign keys】\nclient.`district_id` = district.`district_id`\n【Question】\nWhat is the gender of the youngest client who opened account in the lowest average salary branch?\n【Evidence】\nLater birthdate refers to younger age; A11 refers to average salary\n【Answer】\n```json\n{{\n  \"account\": \"keep_all\",\n  \"client\": \"keep_all\",\n  \"loan\": \"drop_all\",\n  \"district\": [\"district_id\", \"A11\", \"A2\", \"A4\", \"A6\", \"A7\"]\n}}\n```\nQuestion Solved.\n\n==========\n\nHere is a new example, please start answering:\n\n【DB_ID】 {db_id}\n【Schema】\n{desc_str}\n【Foreign keys】\n{fk_str}\n【Question】\n{query}\n【Evidence】\n{evidence}\n【Answer】\n'''

    def _parse_json(self, text: str) -> dict:
        start = text.find("```json")
        end = text.find("```", start + 7)
        if start != -1 and end != -1:
            json_string = text[start + 7: end]
            try:
                return json.loads(json_string)
            except:
                return {}
        return {}

    def _build_desc_str(self, schema_df: pd.DataFrame) -> str:
        desc_str = ""
        grouped = schema_df.groupby('table_name')
        for table_name, group in grouped:
            desc_str += f"# Table: {table_name}\n[\n"
            for _, row in group.iterrows():
                col = row['column_name']
                desc = row.get('column_descriptions', col)
                values = row.get('sample_rows', 'No value examples found.')
                if isinstance(values, list):
                    values = '[' + ', '.join(map(str, values[:5])) + ']'
                else:
                    values = str(values)
                desc_str += f"  ({col}, {desc}. Value examples: {values}.),\n"
            desc_str += "]\n"
        return desc_str

    def _build_fk_str(self, schema_df: pd.DataFrame) -> str:
        fk_str = ""
        for _, row in schema_df.iterrows():
            fk_table = row.get('foreign_key_table')
            fk_col = row.get('foreign_key_column')
            if pd.notna(fk_table) and pd.notna(fk_col):
                fk_str += f"{row['table_name']}.`{row['column_name']}` = {fk_table}.`{fk_col}`\n"
        return fk_str

    def act(self, item, schema: Union[str, Path, Dict, List] = None, **kwargs):
        row = self.dataset[item]
        question = row['question']
        evidence = row.get('evidence', '')
        db_id = row.get('db_id', 'database')

        # Normalize schema to DataFrame
        if isinstance(schema, (str, Path)):
            schema = load_dataset(schema)
        if isinstance(schema, List):
            schema = pd.DataFrame(schema)
        if isinstance(schema, Dict):
            data = []
            for table, cols in schema.items():
                if isinstance(cols, list):
                    for col in cols:
                        data.append({'table_name': table, 'column_name': col})
                elif isinstance(cols, dict):
                    for col, info in cols.items():
                        entry = {'table_name': table, 'column_name': col}
                        entry.update(info)
                        data.append(entry)
            schema = pd.DataFrame(data)
        if not isinstance(schema, pd.DataFrame):
            raise ValueError("Invalid schema format")

        desc_str = self._build_desc_str(schema)
        fk_str = self._build_fk_str(schema)
        prompt = self.selector_template.format(db_id=db_id, desc_str=desc_str, fk_str=fk_str, query=question, evidence=evidence)

        response = self.llm.complete(prompt)
        reply = response.text.strip()
        extracted_dict = self._parse_json(reply)

        # Build schema_links
        schema_links = []
        table_columns = schema.groupby('table_name')['column_name'].apply(list).to_dict()
        for table, value in extracted_dict.items():
            if value == "keep_all":
                if table in table_columns:
                    for col in table_columns[table]:
                        schema_links.append(f"{table}.{col}")
            elif isinstance(value, list):
                for col in value:
                    schema_links.append(f"{table}.{col}")

        schema_links = list(set(schema_links))

        if self.is_save:
            instance_id = row.get("instance_id", item)
            save_path = Path(self.save_dir) / f"{self.NAME}_{instance_id}.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(schema_links, f, ensure_ascii=False, indent=4)
            self.dataset.setitem(item, self.OUTPUT_NAME, str(save_path))

        return schema_links 