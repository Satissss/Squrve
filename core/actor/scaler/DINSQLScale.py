import re
from typing import Union, List, Optional, Dict
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.scaler.BaseScale import BaseScaler
from core.data_manage import Dataset
from core.utils import parse_schema_from_df, load_dataset
from llama_index.core.llms.llm import LLM


class DINSQLScaler(BaseScaler):
    """Scaler implementation based on DIN-SQL's SQL generation strategy for producing multiple SQL candidates."""

    NAME = "DINSQLScaler"

    # DIN-SQL prompts for different difficulty levels
    EASY_PROMPT_TEMPLATE = '''### Here are some reference examples:
# 
Q: "Find the buildings which have rooms with capacity more than 50."
schema_links: [classroom.building,classroom.capacity,50]
SQL: SELECT DISTINCT building FROM classroom WHERE capacity  >  50

Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100]
SQL: SELECT building ,  room_number FROM classroom WHERE capacity BETWEEN 50 AND 100

Q: "Give the name of the student in the History department with the most credits."
schema_links: [student.name,student.dept_name,student.tot_cred,History]
SQL: SELECT name FROM student WHERE dept_name  =  'History' ORDER BY tot_cred DESC LIMIT 1

Q: "Find the total budgets of the Marketing or Finance department."
schema_links: [department.budget,department.dept_name,Marketing,Finance]
SQL: SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'

Q: "Find the department name of the instructor whose name contains 'Soisalon'."
schema_links: [instructor.dept_name,instructor.name,Soisalon]
SQL: SELECT dept_name FROM instructor WHERE name LIKE '%Soisalon%'

Q: "What is the name of the department with the most credits?"
schema_links: [course.dept_name,course.credits]
SQL: SELECT dept_name FROM course GROUP BY dept_name ORDER BY sum(credits) DESC LIMIT 1

Q: "How many instructors teach a course in the Spring of 2010?"
schema_links: [teaches.ID,teaches.semester,teaches.YEAR,Spring,2010]
SQL: SELECT COUNT (DISTINCT ID) FROM teaches WHERE semester  =  'Spring' AND YEAR  =  2010

Q: "Find the name of the students and their department names sorted by their total credits in ascending order."
schema_links: [student.name,student.dept_name,student.tot_cred]
SQL: SELECT name ,  dept_name FROM student ORDER BY tot_cred

Q: "Find the year which offers the largest number of courses."
schema_links: [SECTION.YEAR,SECTION.*]
SQL: SELECT YEAR FROM SECTION GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1

Q: "What are the names and average salaries for departments with average salary higher than 42000?"
schema_links: [instructor.dept_name,instructor.salary,42000]
SQL: SELECT dept_name ,  AVG (salary) FROM instructor GROUP BY dept_name HAVING AVG (salary)  >  42000

###

{schema}

Q: "{question}"
schema_links: {schema_links}
SQL: '''

    MEDIUM_PROMPT_TEMPLATE = '''### Here are some reference examples:
# 
Q: "Find the total budgets of the Marketing or Finance department."
Schema_links: [department.budget,department.dept_name,Marketing,Finance]
A: Let's think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select sum(department.budget) from department  where  department.dept_name = "Marketing"  or  department.dept_name = "Finance"
SQL: SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'

Q: "Find the name and building of the department with the highest budget."
Schema_links: [department.budget,department.dept_name,department.building]
A: Let's think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select department.dept_name , department.building from department  order by department.budget desc limit 1
SQL: SELECT dept_name ,  building FROM department ORDER BY budget DESC LIMIT 1

Q: "What is the name and building of the departments whose budget is more than the average budget?"
Schema_links: [department.budget,department.dept_name,department.building]
A: Let's think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation:  select department.dept_name , department.building from department  where  @.@ > avg ( department.budget ) 
SQL: SELECT dept_name ,  building FROM department WHERE budget  >  (SELECT avg(budget) FROM department)

Q: "Find the total number of students and total number of instructors for each department."
Schema_links: [department.dept_name = student.dept_name,student.id,department.dept_name = instructor.dept_name,instructor.id]
A: Let's think step by step. For creating the SQL for the given question, we need to join these tables = [department,student,instructor]. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: "select count( distinct student.ID) , count( distinct instructor.ID) , department.dept_name from department  group by instructor.dept_name
SQL: SELECT count(DISTINCT T2.id) ,  count(DISTINCT T3.id) ,  T3.dept_name FROM department AS T1 JOIN student AS T2 ON T1.dept_name  =  T2.dept_name JOIN instructor AS T3 ON T1.dept_name  =  T3.dept_name GROUP BY T3.dept_name

Q: "Find the title of courses that have two prerequisites?"
Schema_links: [course.title,course.course_id = prereq.course_id]
A: Let's think step by step. For creating the SQL for the given question, we need to join these tables = [course,prereq]. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select course.title from course  where  count ( prereq.* )  = 2  group by prereq.course_id
SQL: SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2

###

{schema}

Q: "{question}"
Schema_links: {schema_links}
A: Let's think step by step. For creating the SQL for the given question, we need to join these tables = []. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: '''

    HARD_PROMPT_TEMPLATE = '''### Here are some reference examples:
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

{schema}

# [Question]: "{question}"
# [Schema links]: {schema_links}
# [Analysis]: Let's think step by step. "{question}" can be solved by knowing the answer to the following sub-question "{sub_question}".
The SQL query for the sub-question "{sub_question}" is '''

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Union[LLM, List[LLM]] = None,
            generate_num: int = 10,
            temperature: float = 0.7,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            open_parallel: bool = True,
            max_workers: int = None,
            use_external: bool = True,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.generate_num = generate_num
        self.temperature = temperature
        self.use_external = use_external

    def _generate_single_sql(self, llm_: LLM, question: str, schema: str, schema_links: str, difficulty: str = "EASY", sub_questions: Union[str, List[str]] = None) -> Optional[str]:
        """Generate a single SQL candidate using DIN-SQL approach"""
        try:
            if difficulty == "EASY":
                prompt = self.EASY_PROMPT_TEMPLATE.format(
                    schema=schema,
                    question=question,
                    schema_links=schema_links
                )
            elif difficulty == "NON-NESTED":
                prompt = self.MEDIUM_PROMPT_TEMPLATE.format(
                    schema=schema,
                    question=question,
                    schema_links=schema_links
                )
            else:  # NESTED (HARD)
                # Use provided sub_questions if available, otherwise generate a generic one
                if sub_questions:
                    if isinstance(sub_questions, list):
                        sub_question = sub_questions[0] if sub_questions else f"What is the answer to {question}?"
                    else:
                        sub_question = str(sub_questions)
                else:
                    sub_question = f"What is the answer to {question}?"
                prompt = self.HARD_PROMPT_TEMPLATE.format(
                    schema=schema,
                    question=question,
                    schema_links=schema_links,
                    sub_question=sub_question
                )
                
            response = llm_.complete(prompt, temperature=self.temperature).text
            
            # Extract SQL from response
            if "SQL:" in response:
                sql = response.split("SQL:")[-1].strip()
            elif "# [Sql]:" in response:
                sql = response.split("# [Sql]:")[-1].strip()
            else:
                # Look for SELECT statements
                lines = response.split('\n')
                for line in lines:
                    if line.strip().upper().startswith('SELECT'):
                        sql = line.strip()
                        break
                else:
                    return None
            
            # Clean up SQL
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
            
            sql = sql.strip()
            if sql and sql.upper().startswith('SELECT'):
                return sql
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to generate SQL candidate: {e}")
            return None

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            sub_questions: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ) -> List[str]:
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        row = self.dataset[item]
        question = row['question']
        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external", None))
            if external_knowledge:
                question += "\n" + external_knowledge
                logger.debug("已加载外部知识")
        
        # Load and process schema using base class method
        schema = self.process_schema(schema, item)
        
        # Process schema_links
        if isinstance(schema_links, list):
            schema_links_str = str(schema_links)
        elif schema_links is None:
            schema_links_str = "[]"
        else:
            schema_links_str = str(schema_links)

        # Get LLM
        if isinstance(self.llm, list) and self.llm:
            llm = self.llm[0]
        else:
            llm = self.llm

        if llm is None:
            logger.warning("No LLM available for SQL generation")
            return []

        # Generate multiple SQL candidates using different difficulty levels
        pred_sqls = []
        difficulties = ["EASY", "NON-NESTED", "NESTED"]
        
        # Distribute generation across different difficulty levels
        samples_per_difficulty = max(1, self.generate_num // len(difficulties))
        
        for difficulty in difficulties:
            for _ in range(samples_per_difficulty):
                sql = self._generate_single_sql(llm, question, schema, schema_links_str, difficulty, sub_questions)
                if sql:
                    pred_sqls.append(sql)
        
        # Fill remaining slots with EASY difficulty if needed
        while len(pred_sqls) < self.generate_num:
            sql = self._generate_single_sql(llm, question, schema, schema_links_str, "NESTED", sub_questions)
            if sql:
                pred_sqls.append(sql)
            else:
                break

        # Deduplicate
        pred_sqls = list(dict.fromkeys(pred_sqls))

        # Ensure at least one SQL result
        if not pred_sqls:
            logger.warning(f"No SQL candidates generated for item {item}, creating default SQL")
            pred_sqls = ["SELECT * FROM table LIMIT 1"]

        logger.info(f"DINSQLScaler: Generated {len(pred_sqls)} SQL candidates for item {item}")
        if data_logger:
            data_logger.info(f"{self.NAME}.candidates | count={len(pred_sqls)}")

        # Save results using base class method
        self.save_output(pred_sqls, item, row.get("instance_id"))

        if data_logger:
            data_logger.info(f"{self.NAME}.pred_sqls | output={pred_sqls}")
            data_logger.info(f"{self.NAME}.act end | item={item}")
        return pred_sqls
