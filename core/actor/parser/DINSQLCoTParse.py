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
            use_llm_scaling: bool = False,
            open_parallel: bool = False,
            max_workers: int = None,
            **kwargs
    ):
        self.dataset: Dataset = dataset
        self.llm: Union[LLM, List[LLM]] = llm
        self.output_format: str = output_format
        self.is_save: bool = is_save
        self.save_dir: Union[str, PathLike] = save_dir
        self.use_external: bool = use_external
        self.generate_num: int = generate_num
        self.use_llm_scaling: bool = use_llm_scaling
        self.open_parallel: bool = open_parallel
        self.max_workers: int = max_workers

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

    def act(self, item, schema: Union[str, PathLike, Dict, List] = None, **kwargs):
        row = self.dataset[item]
        question = row["question"]

        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external"))
            if external_knowledge:
                question += "\n" + external_knowledge

        if isinstance(schema, (str, PathLike)) and Path(schema).exists():
            schema = load_dataset(schema)

        if schema is None:
            instance_schema_path = row.get("instance_schemas", None)
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
            if schema is None:
                schema = self.dataset.get_db_schema(item)
            if schema is None:
                raise Exception("Failed to load a valid database schema for the sample!")
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)

        schema_context = parse_schema_from_df(schema)

        # 在 act 方法内部初始化 llm，考虑 self.llm 是否为列表
        if isinstance(self.llm, list) and self.llm:
            llm = self.llm[0]
        else:
            llm = self.llm

        if llm is None:
            # 如果没有有效的 LLM，返回空结果
            return []

        # 仅使用第一个 LLM 生成 schema links
        schema_links = []
        for _ in range(self.generate_num):
            links = self.generate_schema_links(llm, question, schema_context)
            schema_links.extend(links)

        schema_links = list(dict.fromkeys(schema_links))

        output = str(schema_links) if self.output_format == "str" else schema_links

        if self.is_save:
            instance_id = row.get("instance_id", item)
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            file_ext = ".txt" if self.output_format == "str" else ".json"
            save_path = save_path / f"{self.NAME}_{instance_id}{file_ext}"
            save_dataset(output, new_data_source=save_path)
            self.dataset.setitem(item, "schema_links", str(save_path))

        return output
