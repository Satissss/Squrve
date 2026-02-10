import json
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Optional
from os import PathLike
from loguru import logger

from core.actor.parser.BaseParse import BaseParser, parallel_slice_parse
from core.utils import load_dataset, save_dataset
from core.data_manage import Dataset, single_central_process
from llama_index.core.llms.llm import LLM

@BaseParser.register_actor
class MACSQLCoTParser(BaseParser):
    """
    Extract relevant schema links for a query using MAC-SQL approach with table and column selection.
    """
    NAME = "MACSQLCoTParser"
    OUTPUT_NAME = "schema_links"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Union[LLM, List[LLM]] = None,
            output_format: str = "list",  # output in `list` or `str`
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/schema_links",
            use_external: bool = False,
            **kwargs
    ):
        super().__init__(dataset, llm, output_format, is_save, save_dir, use_external, **kwargs)

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

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

    def _process_single_llm(self, llm_model, prompt, schema):
        """Process a single LLM call and return schema links"""
        try:
            response = llm_model.complete(prompt)
            reply = response.text.strip()
            result = self._parse_json(reply)
            return result
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return {}

    def _extract_schema_links(self, result: dict) -> List[str]:
        """Extract schema links from LLM response"""
        schema_links = []
        for table, columns in result.items():
            if isinstance(columns, list):
                for col in columns:
                    schema_links.append(f"{table}.{col}")
            elif columns == "keep_all":
                # For keep_all, we need to get all columns from the original schema
                # This is a simplified version - in practice you might want to handle this differently
                pass
        return schema_links

    @parallel_slice_parse
    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, data_logger=None, update_dataset=True,
            **kwargs):
        """
        Extract relevant schema links using MAC-SQL approach.
        """
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        row = self.dataset[item]
        query = row.get("question", "")
        evidence = row.get("evidence", "")
        db_id = row.get("db_id", "")

        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external"))
            if external_knowledge:
                query += "\n" + external_knowledge

        # Use base class method to process schema
        schema_df = self.process_schema(item, schema)
        desc_str = self._build_desc_str(schema_df)
        fk_str = self._build_fk_str(schema_df)

        # Use base class method to get LLM
        llm = self.get_llm()
        if llm is None:
            return []

        # Build prompt
        prompt = self.selector_template.format(
            db_id=db_id,
            desc_str=desc_str,
            fk_str=fk_str,
            query=query,
            evidence=evidence
        )

        # Process with LLM
        result = self._process_single_llm(llm, prompt, schema_df)
        schema_links = self._extract_schema_links(result)

        # Deduplicate
        schema_links = list(set(schema_links))
        self.log_schema_links(data_logger, schema_links, stage="final")
        output = self.format_output(schema_links)

        # Use base class method to save output
        if update_dataset:
            self.save_output(output, item)
        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return output

    def merge_results(self, results: List):
        if not results:
            logger.info("Input results empty!")

        merge_result = []
        for row in results:
            if not isinstance(row, List):
                raise TypeError(f"Each row must be a list, but got {type(row)}: {row}")

            merge_result.extend(row)

        return merge_result
