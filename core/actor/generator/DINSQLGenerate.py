from llama_index.core.llms.llm import LLM
from typing import Union, List, Dict, Optional
import pandas as pd
from os import PathLike
from pathlib import Path
from loguru import logger
import time

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)


class DINSQLGenerator(BaseGenerator):
    """DIN-SQL method implementation for Text-to-SQL generation.

    This class provides a faithful reproduction of the DIN-SQL approach,
    integrated into the Squrve framework without dependencies on external baselines.
    """

    NAME = "DINSQLGenerator"

    # Prompt constants for better organization and readability
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

    CLASSIFICATION_PROMPT = '''Q: "Find the buildings which have rooms with capacity more than 50."
schema_links: [classroom.building,classroom.capacity,50]
A: Let's think step by step. The SQL query for the question "Find the buildings which have rooms with capacity more than 50." needs these tables = [classroom], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

Q: "What are the names of all instructors who advise students in the math depart sorted by total credits of the student."
schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.dept_name,student.tot_cred,math]
A: Let's think step by step. The SQL query for the question "What are the names of all instructors who advise students in the math depart sorted by total credits of the student." needs these tables = [advisor,instructor,student], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

Q: "How many courses that do not have prerequisite?"
schema_links: [course.*,course.course_id = prereq.course_id]
A: Let's think step by step. The SQL query for the question "How many courses that do not have prerequisite?" needs these tables = [course,prereq], so we need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Which courses have prerequisite?"].
So, we need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"'''

    EASY_PROMPT = '''### Here are some reference examples:
# 
Q: "Find the buildings which have rooms with capacity more than 50."
schema_links: [classroom.building,classroom.capacity,50]
SQL: SELECT DISTINCT building FROM classroom WHERE capacity > 50

Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100]
SQL: SELECT room_number, building FROM classroom WHERE capacity BETWEEN 50 AND 100

Q: "Show the status shared by cities with population bigger than 1500 and smaller than 500."
schema_links: [city.Status,city.Population,1500,500]
SQL: SELECT DISTINCT Status FROM city WHERE Population > 1500 AND Population < 500

Q: "Show the id, the date of account opened, the account name, and other account detail for all accounts."
schema_links: [Accounts.account_id,Accounts.account_name,Accounts.other_account_details,Accounts.date_account_opened]
SQL: SELECT account_id, date_account_opened, account_name, other_account_details FROM Accounts

###
'''

    MEDIUM_PROMPT = '''### Here are some reference examples:
# 
Q: "What are the names of all instructors who advise students in the math depart sorted by total credits of the student."
schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.dept_name,student.tot_cred,math]
A: Let's think step by step. The SQL query for the question "What are the names of all instructors who advise students in the math depart sorted by total credits of the student." needs these tables = [advisor,instructor,student], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Intermediate_representation: select instructor.name from instructor join advisor on instructor.id = advisor.i_id join student on advisor.s_id = student.id where student.dept_name = 'math' order by student.tot_cred
SQL: SELECT T1.name FROM instructor AS T1 JOIN advisor AS T2 ON T1.ID = T2.i_ID JOIN student AS T3 ON T2.s_ID = T3.ID WHERE T3.dept_name = 'math' ORDER BY T3.tot_cred

Q: "Find the title, credit, and department name of courses that have more than one prerequisites?"
schema_links: [course.title,course.credits,course.dept_name,course.course_id = prereq.course_id]
A: Let's think step by step. The SQL query for the question "Find the title, credit, and department name of courses that have more than one prerequisites?" needs these tables = [course,prereq], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Intermediate_representation: select course.title , course.credits , course.dept_name from course join prereq on course.course_id = prereq.course_id group by prereq.course_id having count(*) > 1
SQL: SELECT T1.title , T1.credits , T1.dept_name FROM course AS T1 JOIN prereq AS T2 ON T1.course_id = T2.course_id GROUP BY T2.course_id HAVING count(*) > 1

###
'''

    HARD_PROMPT = '''### Here are some reference examples:
# [Question]: "Find the title of courses that have two prerequisites?"
# [Schema links]: [course.title,course.course_id = prereq.course_id]
# [Analysis]: Let's think step by step. "Find the title of courses that have two prerequisites?" can be solved by knowing the answer to the following sub-question "What are the titles for courses with two prerequisites?".
The SQL query for the sub-question "What are the titles for courses with two prerequisites?" is SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2
So, the answer to the question "Find the title of courses that have two prerequisites?" is =
Intermediate_representation: select course.title from course  where  count ( prereq.* )  = 2  group by prereq.course_id
# [Sql]: SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2

# [Question]: "Find the name and building of the department with the highest budget."
# [Schema links]: [department.dept_name,department.building,department.budget]
# [Analysis]: Let's think step by step. "Find the name and building of the department with the highest budget." can be solved by knowing the answer to the following sub-question "What is the department name and corresponding building for the department with the greatest budget?".
The SQL query for the sub-question "What is the department name and corresponding building for the department with the greatest budget?" is SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1
So, the answer to the question "Find the name and building of the department with the highest budget." is =
Intermediate_representation: select department.dept_name , department.building from department  order by department.budget desc limit 1
# [Sql]: SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1

# [Question]: "Give the name and building of the departments with greater than average budget."
# [Schema links]: [department.dept_name,department.building,department.budget]
# [Analysis]: Let's think step by step. "Give the name and building of the departments with greater than average budget." can be solved by knowing the answer to the following sub-question "What is the average budget of departments?".
The SQL query for the sub-question "What is the average budget of departments?" is SELECT avg(budget) FROM department
So, the answer to the question "Give the name and building of the departments with greater than average budget." is =
Intermediate_representation: select department.dept_name , department.building from department  where  @.@ > avg ( department.budget )
# [Sql]: SELECT dept_name ,  building FROM department WHERE budget  >  (SELECT avg(budget) FROM department)

###
'''

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            db_path: Optional[Union[str, PathLike]] = None,
            credential: Optional[Dict] = None,
            **kwargs
    ) -> None:
        """Initialize the DIN-SQL generator with optional parameters."""
        self.dataset: Optional[Dataset] = dataset
        self.llm: Optional[LLM] = llm
        self.is_save = is_save
        self.save_dir: Union[str, PathLike] = save_dir

        # Safely initialize db_path and credential
        self.db_path = db_path or (self.dataset.db_path if self.dataset else None)
        self.credential = credential or (self.dataset.credential if self.dataset else None)

    def schema_linking_prompt_maker(self, question: str, schema: str) -> str:
        instruction = "# Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.\n"
        return instruction + self.SCHEMA_LINKING_PROMPT + schema + 'Q: "' + question + '"\nA: Let\'s think step by step.'

    def classification_prompt_maker(self, question: str, schema: str, schema_links: Union[str, List]) -> str:
        instruction = "# For the given question, classify it as EASY, NON-NESTED, or NESTED based on nested queries and JOIN.\n"
        instruction += "\nif need nested queries: predict NESTED\n"
        instruction += "elif need JOIN and don't need nested queries: predict NON-NESTED\n"
        instruction += "elif don't need JOIN and don't need nested queries: predict EASY\n\n"
        return instruction + schema + "\n" + self.CLASSIFICATION_PROMPT + 'Q: "' + question + '"\nschema_links: ' + str(
            schema_links) + '\nA: Let\'s think step by step.'

    def easy_prompt_maker(self, question: str, schema: str, schema_links: Union[str, List]) -> str:
        instruction = "# Use the the schema links to generate the SQL queries for each of the questions.\n"
        return instruction + schema + "\n" + self.EASY_PROMPT + 'Q: "' + question + '"\nSchema_links: ' + str(
            schema_links) + '\nSQL:'

    def medium_prompt_maker(self, question: str, schema: str, schema_links: Union[str, List]) -> str:
        instruction = "# Use the the schema links and Intermediate_representation to generate the SQL queries for each of the questions.\n"
        return instruction + schema + "\n" + self.MEDIUM_PROMPT + 'Q: "' + question + '"\nSchema_links: ' + str(
            schema_links) + '\nA: Let\'s think step by step.'

    def hard_prompt_maker(self, question: str, schema: str, schema_links: Union[str, List], sub_questions: str) -> str:
        instruction = "# Use the intermediate representation and the schema links to generate the SQL queries for each of the questions.\n"
        stepping = f'\nA: Let\'s think step by step. "{question}" can be solved by knowing the answer to the following sub-question "{sub_questions}".'
        return instruction + schema + "\n" + self.HARD_PROMPT + 'Q: "' + question + '"' + '\nschema_links: ' + str(
            schema_links) + stepping + '\nThe SQL query for the sub-question"'

    def debug_prompt_maker(self, question: str, schema: str, sql: str) -> str:
        instruction = """#### For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given SQLite SQL QUERY for any issues. If there are any problems, fix them. If there are no issues, return the SQLite SQL QUERY as is.
#### Use the following instructions for fixing the SQL QUERY:
1) Use the database values that are explicitly mentioned in the question.
2) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
3) Use DESC and DISTINCT when needed.
4) Pay attention to the columns that are used for the GROUP BY statement.
5) Pay attention to the columns that are used for the SELECT statement.
6) Only change the GROUP BY clause when necessary (Avoid redundant columns in GROUP BY).
7) Use GROUP BY on one column only.

"""
        return instruction + schema + '#### Question: ' + question + '\n#### SQLite SQL QUERY\n' + sql + '\n#### SQLite FIXED SQL QUERY\nSELECT'

    def llm_generation(self, prompt: str) -> str:
        """Generate response using LLM with retry mechanism."""
        if self.llm is None:
            raise ValueError("LLM is not initialized")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.complete(prompt)
                return response.text.strip()
            except Exception as e:
                logger.warning(f"LLM generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Max retries exceeded for LLM generation.")
                    raise

    def llm_debug(self, prompt: str) -> str:
        """Debug SQL using LLM with retry mechanism."""
        if self.llm is None:
            raise ValueError("LLM is not initialized")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.complete(prompt)
                return response.text.strip().replace("\n", " ")
            except Exception as e:
                logger.warning(f"LLM debug attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error("Max retries exceeded for LLM debug.")
                    raise

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            **kwargs
    ) -> str:
        """Process a single item and generate SQL."""
        logger.info(f"DIN_SQLGenerator processing sample {item}")
        row = self.dataset[item]
        question = row['question']
        db_type = row['db_type']
        db_id = row["db_id"]
        db_path = Path(self.db_path) / (db_id + ".sqlite") if self.db_path else self.db_path
        logger.debug(f"Processing question: {question[:100]}... (DB: {db_id}, Type: {db_type})")

        # Load and process schema
        logger.debug("Processing database schema...")
        if isinstance(schema, (str, PathLike)) and Path(schema).exists():
            schema = load_dataset(schema)

        if schema is None:
            instance_schema_path = row.get("instance_schemas")
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
                logger.debug(f"Loaded schema from: {instance_schema_path}")
            else:
                logger.debug("Fetching schema from dataset")
                schema = self.dataset.get_db_schema(item)

            if schema is None:
                raise ValueError("Failed to load a valid database schema for the sample!")

        # Normalize schema type
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            schema = parse_schema_from_df(schema)
        else:
            raise ValueError("Invalid schema format")

        logger.debug("Database schema processed")

        # Step 1: Schema linking
        logger.debug("Starting schema linking...")
        if schema_links is None:
            schema_link_path = row.get("schema_links", None)
            if schema_link_path:
                schema_links = load_dataset(schema_link_path)
                logger.debug(f"Loaded schema links from: {schema_link_path}")
            else:
                logger.debug("Generating schema links using DIN-SQL")
                schema_linking_prompt = self.schema_linking_prompt_maker(question, schema)
                schema_links_raw = self.llm_generation(schema_linking_prompt)
                try:
                    schema_links = schema_links_raw.split("Schema_links: ")[1].strip("[]")
                except IndexError:
                    logger.warning("Schema linking parsing failed, using default")
                    schema_links = "[]"

        # Step 2: Difficulty classification
        logger.debug("Starting difficulty classification...")
        try:
            class_prompt = self.classification_prompt_maker(question, schema, schema_links)
            classification = self.llm_generation(class_prompt)
            logger.debug("Difficulty classification completed")
        except Exception as e:
            logger.error(f"Difficulty classification failed: {e}")
            raise

        try:
            predicted_class = classification.split("Label: ")[1]
        except IndexError:
            logger.warning("Classification parsing failed, defaulting to NESTED")
            predicted_class = '"NESTED"'

        # Step 3: SQL generation based on difficulty
        logger.debug("Starting SQL generation...")
        if '"EASY"' in predicted_class:
            logger.debug("Processing EASY category query")
            sql = self.llm_generation(self.easy_prompt_maker(question, schema, schema_links))
        elif '"NON-NESTED"' in predicted_class:
            logger.debug("Processing NON-NESTED category query")
            sql = self.llm_generation(self.medium_prompt_maker(question, schema, schema_links))
            try:
                sql = sql.split("SQL: ")[1]
            except IndexError:
                logger.warning("SQL parsing failed")
                sql = "SELECT"
        else:
            logger.debug("Processing NESTED category query")
            try:
                sub_questions = classification.split('questions = ["')[1].split('"]')[0]
            except IndexError:
                logger.warning("Sub-questions parsing failed")
                sub_questions = "What is the answer?"

            sql = self.llm_generation(self.hard_prompt_maker(question, schema, schema_links, sub_questions))
            try:
                sql = sql.split("SQL: ")[1]
            except IndexError:
                logger.warning("SQL parsing failed")
                sql = "SELECT"

        # Step 4: SQL debugging
        logger.debug("Starting SQL debugging...")
        try:
            debug_prompt = self.debug_prompt_maker(question, schema, sql)
            debugged_sql = self.llm_debug(debug_prompt)
            sql = "SELECT " + debugged_sql
            logger.debug("SQL debugging completed")
        except Exception as e:
            logger.error(f"SQL debugging failed: {e}")
            # Keep original SQL if debugging fails

        # SQL post-process
        # if self.sql_post_process_function:
        #     sql = self.sql_post_process_function(sql, self.dataset)

        logger.debug(f"Final SQL: {sql[:100]}...")

        if self.is_save:
            instance_id = row.get("instance_id")
            save_path = Path(self.save_dir)
            save_path = save_path / str(self.dataset.dataset_index) if self.dataset.dataset_index else save_path
            save_path = save_path / f"{self.name}_{instance_id}.sql"

            save_dataset(sql, new_data_source=save_path)
            self.dataset.setitem(item, "pred_sql", str(save_path))
            logger.debug(f"SQL saved to: {save_path}")

        logger.info(f"DIN_SQLGenerator sample {item} processed")
        return sql