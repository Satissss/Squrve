from typing import Union, List, Optional, Dict
from os import PathLike
from pathlib import Path
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core.llms import LLM

from core.actor.decomposer.BaseDecompose import BaseDecomposer
from core.data_manage import Dataset, single_central_process
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)

class DINSQLDecomposer(BaseDecomposer):
    """Decomposer implementation based on DIN-SQL's classification for identifying sub-questions in complex queries."""

    NAME = "DINSQLDecomposer"

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
A: Let\'s think step by step. In the question "Find the buildings which have rooms with capacity more than 50.", we are asked:
"the buildings which have rooms" so we need column = [classroom.capacity]
"rooms with capacity" so we need column = [classroom.building]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [50]. So the Schema_links are:
Schema_links: [classroom.building,classroom.capacity,50]'''

    CLASSIFICATION_PROMPT = '''Q: "Find the buildings which have rooms with capacity more than 50."
schema_links: [classroom.building,classroom.capacity,50]
A: Let\'s think step by step. The SQL query for the question "Find the buildings which have rooms with capacity more than 50." needs these tables = [classroom], so we don\'t need JOIN.
Plus, it doesn\'t require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don\'t need JOIN and don\'t need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

Q: "What are the names of all instructors who advise students in the math depart sorted by total credits of the student."
schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.dept_name,student.tot_cred,math]
A: Let\'s think step by step. The SQL query for the question "What are the names of all instructors who advise students in the math depart sorted by total credits of the student." needs these tables = [advisor,instructor,student], so we need JOIN.
Plus, it doesn\'t need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don\'t need nested queries, then the the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

Q: "How many courses that do not have prerequisite?"
schema_links: [course.*,course.course_id = prereq.course_id]
A: Let\'s think step by step. The SQL query for the question "How many courses that do not have prerequisite?" needs these tables = [course,prereq], so we need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Which courses have prerequisite?"].
So, we need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"'''

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Union[LLM, List[LLM]] = None,
            generate_num: int = 1,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/sub_questions",
            use_llm_scaling: bool = False,
            open_parallel: bool = False,
            max_workers: int = None,
            **kwargs
    ):
        self.dataset = dataset
        self.llm = llm if isinstance(llm, list) else [llm]
        self.generate_num = generate_num
        self.is_save = is_save
        self.save_dir = save_dir
        self.use_llm_scaling = use_llm_scaling
        self.open_parallel = open_parallel
        self.max_workers = max_workers

    def schema_linking_prompt_maker(self, question: str, schema: str) -> str:
        instruction = "# Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.\n"
        return instruction + self.SCHEMA_LINKING_PROMPT + schema + 'Q: "' + question + '"\nA: Let\'s think step by step.'

    def classification_prompt_maker(self, question: str, schema: str, schema_links: str) -> str:
        instruction = "# For the given question, classify it as EASY, NON-NESTED, or NESTED based on nested queries and JOIN.\n"
        instruction += "\nif need nested queries: predict NESTED\n"
        instruction += "elif need JOIN and don't need nested queries: predict NON-NESTED\n"
        instruction += "elif don't need JOIN and don't need nested queries: predict EASY\n\n"
        return instruction + schema + "\n" + self.CLASSIFICATION_PROMPT + 'Q: "' + question + '"\nschema_links: ' + schema_links + '\nA: Let\'s think step by step.'

    def generate_sub_questions(self, llm_: LLM, question: str, schema_str: str) -> List[str]:
        # Step 1: Schema linking
        prompt = self.schema_linking_prompt_maker(question, schema_str)
        response = llm_.complete(prompt).text.strip()
        try:
            schema_links = response.split("Schema_links: ")[1]
        except IndexError:
            schema_links = "[]"

        # Step 2: Classification
        class_prompt = self.classification_prompt_maker(question, schema_str, schema_links)
        classification = llm_.complete(class_prompt).text.strip()

        # Parse
        try:
            predicted_class = classification.split("Label: ")[1]
        except IndexError:
            predicted_class = '"NESTED"'

        sub_questions = []
        if '"NESTED"' in predicted_class:
            try:
                sub_questions_str = classification.split('questions = ["')[1].split('"]')[0]
                sub_questions = [q.strip().replace('"', '').replace("'", '') for q in sub_questions_str.split(', ')]
            except IndexError:
                pass

        return sub_questions

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            **kwargs
    ) -> List[str]:
        row = self.dataset[item]
        question = row['question']

        if schema is None:
            instance_schema_path = row.get("instance_schemas", None)
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
            if schema is None:
                schema = self.dataset.get_db_schema(item)
            if schema is None:
                raise ValueError("Failed to load a valid database schema for the sample!")

        if isinstance(schema, (str, PathLike)):
            schema = load_dataset(schema)

        if isinstance(schema, dict):
            schema = single_central_process(schema)
        if isinstance(schema, list):
            schema = pd.DataFrame(schema)

        schema_str = parse_schema_from_df(schema) if isinstance(schema, pd.DataFrame) else str(schema)

        def process_serial(llm_lis_):
            all_sub_questions = []
            for llm_model in llm_lis_:
                for _ in range(self.generate_num):
                    sub_qs = self.generate_sub_questions(llm_model, question, schema_str)
                    all_sub_questions.extend(sub_qs)
            return all_sub_questions

        def process_parallel(llm_lis_):
            all_sub_questions = []
            max_workers_ = self.max_workers if self.max_workers else len(llm_lis_) * self.generate_num
            with ThreadPoolExecutor(max_workers=max_workers_) as executor:
                futures = []
                for llm_model in llm_lis_:
                    for _ in range(self.generate_num):
                        futures.append(executor.submit(self.generate_sub_questions, llm_model, question, schema_str))
                for future in as_completed(futures):
                    try:
                        sub_qs = future.result()
                        all_sub_questions.extend(sub_qs)
                    except Exception as e:
                        logger.error(f"Error in decomposition: {e}")
            return all_sub_questions

        llm_lis = self.llm if isinstance(self.llm, list) else [self.llm]
        sub_questions = []
        if self.use_llm_scaling and isinstance(self.llm, list):
            sub_questions = process_parallel(llm_lis) if self.open_parallel else process_serial(llm_lis)
        else:
            sub_questions = process_serial([llm_lis[0]])

        # Deduplicate
        sub_questions = list(set(sub_questions))

        if self.is_save:
            instance_id = row.get('instance_id', item)
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            save_path = save_path / f"{self.NAME}_{instance_id}.json"
            save_dataset(sub_questions, new_data_source=save_path)
            self.dataset.setitem(item, self.OUTPUT_NAME, str(save_path))

        return sub_questions 