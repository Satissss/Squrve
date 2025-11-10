from llama_index.core.llms.llm import LLM
from typing import Union, List, Dict
import pandas as pd
from os import PathLike
from pathlib import Path

from core.data_manage import Dataset, single_central_process
from core.actor.parser.BaseParse import BaseParser
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)


class DINSQLCoTParser(BaseParser):
    """
    Extract relevant schema links for a query using chain-of-thought prompting in a single pass.
    """

    NAME = "DINSQLCoTParser"

    SCHEMA_LINKING_PROMPT = '''Table advisor, columns = [*,s_ID,i_ID]
Table classroom, columns = [*,building,room_number,capacity]
Table course, columns = [*,course_id,title,dept_name,credits]
Table department, columns = [*,dept_name,building,budget]
Table instructor, columns = [*,ID,name,dept_name,salary]
Table prereq, columns = [*,course_id,prereq_id]
Table section, columns = [*,course_id,sec_id,semester,year,building,room_number,time_slot_id]
Table student, columns = [*,ID,name,dept_name,tot_cred]
Table takes, columns = [*,ID,course_id,sec_id,semester,year,grade]
Table teaches, columns = [*,ID,course_id,sec_id,semester,year]
Table time_slot, columns = [*,time_slot_id,day,start_hr,start_min,end_hr,end_min]
Foreign_keys = [course.dept_name = department.dept_name,instructor.dept_name = department.dept_name,section.building = classroom.building,section.room_number = classroom.room_number,section.course_id = course.course_id,teaches.ID = instructor.ID,teaches.course_id = section.course_id,teaches.sec_id = section.sec_id,teaches.semester = section.semester,teaches.year = section.year,student.dept_name = department.dept_name,takes.ID = student.ID,takes.course_id = section.course_id,takes.sec_id = section.sec_id,takes.semester = section.semester,takes.year = section.year,advisor.s_ID = student.ID,advisor.i_ID = instructor.ID,prereq.prereq_id = course.course_id,prereq.course_id = course.course_id]
Q: "Find the buildings which have rooms with capacity more than 50."
A: Let's think step by step. In the question "Find the buildings which have rooms with capacity more than 50.", we are asked:
"the buildings which have rooms" so we need column = [classroom.capacity]
"rooms with capacity" so we need column = [classroom.building]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [50]. So the Schema_links are:
Schema_links: [classroom.building,classroom.capacity,50]'''

    def __init__(
            self,
            dataset: Dataset = None,
            llm: Union[LLM, List[LLM]] = None,
            output_format: str = "str",  # output in `list` or `str`
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/schema_links",
            use_external: bool = False,
            generate_num: int = 1,
            **kwargs
    ):
        super().__init__(dataset, llm, output_format, is_save, save_dir, use_external, **kwargs)
        self.generate_num = generate_num

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def schema_linking_prompt_maker(self, question: str, schema: str) -> str:
        instruction = "# Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.\n"
        return instruction + self.SCHEMA_LINKING_PROMPT + schema + 'Q: "' + question + '"\nA: Let\'s think step by step.'

    def parse_schema_links(self, response: str) -> List[str]:
        try:
            links_str = response.split("Schema_links: ")[1].strip()
            if links_str.startswith('[') and links_str.endswith(']'):
                links_str = links_str[1:-1]
            links = [link.strip() for link in links_str.split(',')]
            return links
        except IndexError:
            return []

    def generate_schema_links(self, llm_: LLM, question: str, schema_context: str) -> List[str]:
        prompt = self.schema_linking_prompt_maker(question, schema_context)
        response = llm_.complete(prompt).text

        return self.parse_schema_links(response)

    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, data_logger=None, **kwargs):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        row = self.dataset[item]
        question = row["question"]

        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external"))
            if external_knowledge:
                question += "\n" + external_knowledge

        # Use base class method to process schema
        schema_df = self.process_schema(item, schema)
        schema_context = parse_schema_from_df(schema_df)

        # Use base class method to get LLM
        llm = self.get_llm()
        if llm is None:
            # 如果没有有效的 LLM，返回空结果
            return []

        # Generate
        schema_links = []
        for idx in range(self.generate_num):
            links = self.generate_schema_links(llm, question, schema_context)
            schema_links.extend(links)
            self.log_schema_links(data_logger, links, stage=f"generated schema links.{idx}")

        schema_links = list(dict.fromkeys(schema_links))
        self.log_schema_links(data_logger, schema_links, stage="final")

        output = self.format_output(schema_links)

        # Use base class method to save output
        file_ext = ".txt" if self.output_format == "str" else ".json"
        self.save_output(output, item, file_ext=file_ext)

        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return output
