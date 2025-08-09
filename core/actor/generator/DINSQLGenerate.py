from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from loguru import logger
from llama_index.core.llms.llm import LLM

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.utils import load_dataset, parse_schema_from_df, save_dataset


class SQLComplexity(Enum):
    """SQL query complexity classification enumeration"""
    EASY = "EASY"
    NON_NESTED = "NON-NESTED"
    NESTED = "NESTED"


@dataclass(frozen=True)
class PromptTemplates:
    """DIN-SQL prompt template constants"""

    SCHEMA_LINKING = '''Table advisor, columns = [*,s_ID,i_ID]
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

    CLASSIFICATION = '''Q: "Find the buildings which have rooms with capacity more than 50."
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

    EASY = '''### Here are some reference examples:
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

    MEDIUM = '''### Here are some reference examples:
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

    HARD = '''### Here are some reference examples:
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

    DEBUG_INSTRUCTION = """#### For the given question, use the provided tables, columns, foreign keys, and primary keys to fix the given SQLite SQL QUERY for any issues. If there are any problems, fix them. If there are no issues, return the SQLite SQL QUERY as is.
#### Use the following instructions for fixing the SQL QUERY:
1) Use the database values that are explicitly mentioned in the question.
2) Pay attention to the columns that are used for the JOIN by using the Foreign_keys.
3) Use DESC and DISTINCT when needed.
4) Pay attention to the columns that are used for the GROUP BY statement.
5) Pay attention to the columns that are used for the SELECT statement.
6) Only change the GROUP BY clause when necessary (Avoid redundant columns in GROUP BY).
7) Use GROUP BY on one column only.

"""


class DIN_SQLGenerator(BaseGenerator):
    """
    DIN-SQL method implementation for Text-to-SQL generation

    This class provides an optimized implementation of the DIN-SQL approach,
    integrated into the Squrve framework with improved error handling,
    logging, and code organization.
    """

    NAME = "DIN_SQLGenerator"

    # Configuration constants
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 3
    MIN_EXTERNAL_KNOWLEDGE_LENGTH = 50

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = Path("../files/pred_sql"),
            use_external: bool = True,
            use_few_shot: bool = True,
            sql_post_process_function: Optional[Callable[[str, Dataset], str]] = None,
            db_path: Optional[Union[str, Path]] = None,
            credential: Optional[Dict[str, Any]] = None,
            max_retries: int = DEFAULT_MAX_RETRIES,
            retry_delay: float = DEFAULT_RETRY_DELAY,
            **kwargs
    ) -> None:
        """
        Initialize the DIN-SQL generator

        Args:
            dataset: Dataset instance
            llm: Language model instance
            is_save: Whether to save results
            save_dir: Save directory path
            use_external: Whether to use external knowledge
            use_few_shot: Whether to use few-shot learning
            sql_post_process_function: SQL post-processing function
            db_path: Database path
            credential: Credential information
            max_retries: Maximum retry attempts
            retry_delay: Retry delay time in seconds
        """
        super().__init__()

        # Core components
        self.dataset = dataset
        self.llm = llm

        # Configuration parameters
        self.is_save = is_save
        self.save_dir = Path(save_dir)
        self.use_external = use_external
        self.use_few_shot = use_few_shot
        self.sql_post_process_function = sql_post_process_function

        # Database related
        self.db_path = self._resolve_db_path(db_path)
        self.credential = credential or (dataset.credential if dataset else None)

        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Prompt templates
        self.templates = PromptTemplates()

        # Validate necessary components
        self._validate_initialization()

    def _resolve_db_path(self, db_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolve database path"""
        if db_path:
            return Path(db_path)
        if self.dataset and hasattr(self.dataset, 'db_path') and self.dataset.db_path:
            return Path(self.dataset.db_path)
        return None

    def _validate_initialization(self) -> None:
        """Validate initialization parameters"""
        if not self.llm:
            logger.warning("LLM not initialized, some features may be unavailable")

        if self.is_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Save directory created: {self.save_dir}")

    @classmethod
    def load_external_knowledge(
            cls,
            external: Optional[Union[str, Path]] = None
    ) -> Optional[str]:
        """
        Load external knowledge

        Args:
            external: External knowledge file path

        Returns:
            Formatted external knowledge string, None if invalid
        """
        if not external:
            return None

        try:
            external_data = load_dataset(external)
            if external_data and len(str(external_data)) > cls.MIN_EXTERNAL_KNOWLEDGE_LENGTH:
                return f"####[External Prior Knowledge]:\n{external_data}"
            logger.debug("External knowledge content too short, ignored")
            return None
        except Exception as e:
            logger.warning(f"Failed to load external knowledge: {e}")
            return None

    def _build_schema_linking_prompt(self, question: str, schema: str) -> str:
        """Build schema linking prompt"""
        instruction = "# Find the schema_links for generating SQL queries for each question based on the database schema and Foreign keys.\n"
        return f"{instruction}{self.templates.SCHEMA_LINKING}{schema}Q: \"{question}\"\nA: Let's think step by step."

    def _build_classification_prompt(self, question: str, schema: str, schema_links: Union[str, List]) -> str:
        """Build classification prompt"""
        instruction = (
            "# For the given question, classify it as EASY, NON-NESTED, or NESTED based on nested queries and JOIN.\n"
            "\nif need nested queries: predict NESTED\n"
            "elif need JOIN and don't need nested queries: predict NON-NESTED\n"
            "elif don't need JOIN and don't need nested queries: predict EASY\n\n"
        )
        return f"{instruction}{schema}\n{self.templates.CLASSIFICATION}Q: \"{question}\"\nschema_links: {schema_links}\nA: Let's think step by step."

    def _build_easy_prompt(self, question: str, schema: str, schema_links: Union[str, List]) -> str:
        """Build easy query prompt"""
        instruction = "# Use the the schema links to generate the SQL queries for each of the questions.\n"
        return f"{instruction}{schema}\n{self.templates.EASY}Q: \"{question}\"\nSchema_links: {schema_links}\nSQL:"

    def _build_medium_prompt(self, question: str, schema: str, schema_links: Union[str, List]) -> str:
        """Build medium complexity query prompt"""
        instruction = "# Use the the schema links and Intermediate_representation to generate the SQL queries for each of the questions.\n"
        return f"{instruction}{schema}\n{self.templates.MEDIUM}Q: \"{question}\"\nSchema_links: {schema_links}\nA: Let's think step by step."

    def _build_hard_prompt(self, question: str, schema: str, schema_links: Union[str, List], sub_questions: str) -> str:
        """Build hard query prompt"""
        instruction = "# Use the intermediate representation and the schema links to generate the SQL queries for each of the questions.\n"
        stepping = f'\nA: Let\'s think step by step. "{question}" can be solved by knowing the answer to the following sub-question "{sub_questions}".'
        return f"{instruction}{schema}\n{self.templates.HARD}Q: \"{question}\"\nschema_links: {schema_links}{stepping}\nThe SQL query for the sub-question\""

    def _build_debug_prompt(self, question: str, sql: str, schema: str) -> str:
        """Build debug prompt"""
        return f"{self.templates.DEBUG_INSTRUCTION}{schema}#### Question: {question}\n#### SQLite SQL QUERY\n{sql}\n#### SQLite FIXED SQL QUERY\nSELECT"

    def _execute_llm_with_retry(
            self,
            prompt: str,
            operation_name: str = "generation",
            clean_response: bool = True
    ) -> str:
        """
        Execute LLM with retry mechanism

        Args:
            prompt: Prompt text
            operation_name: Operation name (for logging)
            clean_response: Whether to clean response text

        Returns:
            LLM response text

        Raises:
            ValueError: LLM not initialized
            RuntimeError: Maximum retry attempts exceeded
        """
        if not self.llm:
            raise ValueError("LLM not initialized")

        for attempt in range(self.max_retries):
            try:
                response = self.llm.complete(prompt)
                result = response.text.strip()

                if clean_response and operation_name == "debug":
                    result = result.replace("\n", " ")

                logger.debug(f"{operation_name} operation completed successfully")
                return result

            except Exception as e:
                logger.warning(f"{operation_name} operation attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    error_msg = f"{operation_name} operation exceeded max retries ({self.max_retries})"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def _process_schema(
            self,
            schema: Union[str, Path, Dict, List],
            item: int
    ) -> str:
        """
        Process database schema

        Args:
            schema: Schema data
            item: Data item index

        Returns:
            Processed schema string

        Raises:
            ValueError: Unable to load valid database schema
        """
        # If schema is file path, load data
        if isinstance(schema, (str, Path)):
            schema = load_dataset(schema)

        # If schema is None, try to get from dataset
        if schema is None:
            row = self.dataset[item]
            instance_schema_path = row.get("instance_schemas")

            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
                logger.debug(f"Loaded from instance schema path: {instance_schema_path}")
            else:
                logger.debug("Getting schema from dataset")
                schema = self.dataset.get_db_schema(item)

        if schema is None:
            raise ValueError(f"Unable to load valid database schema for sample {item}")

        # Normalize schema format
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            return parse_schema_from_df(schema)
        elif isinstance(schema, str):
            return schema
        else:
            raise ValueError(f"Unsupported schema format: {type(schema)}")

    def _extract_response_content(
            self,
            response: str,
            delimiter: str,
            default_value: str = "",
            operation_name: str = "parsing"
    ) -> str:
        """
        Extract specific content from response

        Args:
            response: Response text
            delimiter: Delimiter string
            default_value: Default value
            operation_name: Operation name (for logging)

        Returns:
            Extracted content
        """
        try:
            return response.split(delimiter)[1]
        except IndexError:
            logger.warning(f"{operation_name} failed, using default value: {default_value}")
            return default_value

    def _determine_sql_complexity(self, classification_response: str) -> SQLComplexity:
        """
        Determine SQL query complexity

        Args:
            classification_response: Classification response

        Returns:
            SQL complexity enumeration value
        """
        predicted_class = self._extract_response_content(
            classification_response,
            "Label: ",
            f'"{SQLComplexity.NESTED.value}"',
            "complexity classification parsing"
        )

        for complexity in SQLComplexity:
            if f'"{complexity.value}"' in predicted_class:
                return complexity

        logger.warning(f"Unrecognized complexity classification: {predicted_class}, defaulting to NESTED")
        return SQLComplexity.NESTED

    def _generate_sql_by_complexity(
            self,
            complexity: SQLComplexity,
            question: str,
            schema: str,
            schema_links: Union[str, List],
            classification_response: str = ""
    ) -> str:
        """
        Generate SQL based on complexity

        Args:
            complexity: SQL complexity
            question: Question
            schema: Database schema
            schema_links: Schema links
            classification_response: Classification response (for extracting sub-questions)

        Returns:
            Generated SQL query
        """
        if complexity == SQLComplexity.EASY:
            logger.debug("Processing EASY query")
            return self._execute_llm_with_retry(
                self._build_easy_prompt(question, schema, schema_links),
                "easy SQL generation"
            )

        elif complexity == SQLComplexity.NON_NESTED:
            logger.debug("Processing NON-NESTED query")
            sql_response = self._execute_llm_with_retry(
                self._build_medium_prompt(question, schema, schema_links),
                "medium SQL generation"
            )
            return self._extract_response_content(
                sql_response,
                "SQL: ",
                "SELECT",
                "medium SQL parsing"
            )

        else:  # NESTED
            logger.debug("Processing NESTED query")

            # Extract sub-questions
            sub_questions = self._extract_response_content(
                classification_response,
                'questions = ["',
                "What is the answer?",
                "sub-question parsing"
            )
            if sub_questions != "What is the answer?":
                sub_questions = sub_questions.split('"]')[0]

            sql_response = self._execute_llm_with_retry(
                self._build_hard_prompt(question, schema, schema_links, sub_questions),
                "hard SQL generation"
            )
            return self._extract_response_content(
                sql_response,
                "SQL: ",
                "SELECT",
                "hard SQL parsing"
            )

    def _save_sql_result(self, sql: str, item: int) -> None:
        """
        Save SQL result

        Args:
            sql: SQL query
            item: Data item index
        """
        if not self.is_save or not self.dataset:
            return

        try:
            row = self.dataset[item]
            instance_id = row.get("instance_id", f"item_{item}")

            # Build save path
            save_path = self.save_dir
            if hasattr(self.dataset, 'dataset_index') and self.dataset.dataset_index:
                save_path = save_path / str(self.dataset.dataset_index)

            save_path = save_path / f"{self.NAME}_{instance_id}.sql"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save file
            save_dataset(sql, new_data_source=save_path)

            # Update dataset
            self.dataset.setitem(item, "pred_sql", str(save_path))

            logger.debug(f"SQL saved to: {save_path}")

        except Exception as e:
            logger.error(f"Failed to save SQL result: {e}")

    def _process_schema_links(
            self,
            schema_links: Union[str, List[str]],
            question: str,
            schema: str,
            row: Dict[str, Any]
    ) -> Union[str, List]:
        """
        Process schema links

        Args:
            schema_links: Pre-provided schema links
            question: Question
            schema: Processed database schema
            row: Data row information

        Returns:
            Processed schema links
        """
        if schema_links is not None:
            logger.debug("Using pre-provided schema links")
            return schema_links

        # Try to load from data row
        schema_link_path = row.get("schema_links")
        if schema_link_path:
            try:
                schema_links = load_dataset(schema_link_path)
                logger.debug(f"Loaded schema links from path: {schema_link_path}")
                return schema_links
            except Exception as e:
                logger.warning(f"Failed to load schema links from path: {e}")

        # Generate schema links using DIN-SQL
        logger.debug("Generating schema links using DIN-SQL")
        schema_linking_prompt = self._build_schema_linking_prompt(question, schema)
        schema_links_response = self._execute_llm_with_retry(
            schema_linking_prompt,
            "schema linking generation"
        )

        return self._extract_response_content(
            schema_links_response,
            "Schema_links: ",
            "[]",
            "schema links parsing"
        )

    def act(
            self,
            item: int,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            **kwargs
    ) -> str:
        """
        Process a single data item and generate SQL

        Args:
            item: Data item index
            schema: Database schema
            schema_links: Schema links
            **kwargs: Other parameters

        Returns:
            Generated SQL query

        Raises:
            ValueError: Dataset not initialized or unable to load schema
            RuntimeError: LLM operation failed
        """
        if not self.dataset:
            raise ValueError("Dataset not initialized")

        logger.info(f"DIN_SQLGenerator started processing sample {item}")

        try:
            # 1. Prepare question and basic information
            row = self.dataset[item]
            question = row['question']
            db_type = row.get('db_type', 'unknown')
            db_id = row.get('db_id', 'unknown')

            logger.debug(f"Processing question: {question[:100]}... (DB: {db_id}, Type: {db_type})")

            # 2. Load external knowledge (if enabled)
            if self.use_external:
                external_knowledge = self.load_external_knowledge(row.get("external"))
                if external_knowledge:
                    question = f"{question}\n{external_knowledge}"
                    logger.debug("External knowledge loaded")

            # 3. Process database schema
            logger.debug("Processing database schema...")
            processed_schema = self._process_schema(schema, item)
            logger.debug("Database schema processing completed")

            # 4. Schema linking step
            logger.debug("Starting schema linking...")
            processed_schema_links = self._process_schema_links(
                schema_links, question, processed_schema, row
            )

            # 5. Difficulty classification
            logger.debug("Starting difficulty classification...")
            classification_response = self._execute_llm_with_retry(
                self._build_classification_prompt(question, processed_schema, processed_schema_links),
                "difficulty classification"
            )
            complexity = self._determine_sql_complexity(classification_response)
            logger.debug(f"Query complexity: {complexity.value}")

            # 6. Generate SQL based on complexity
            logger.debug("Starting SQL generation...")
            sql = self._generate_sql_by_complexity(
                complexity, question, processed_schema, processed_schema_links, classification_response
            )

            # 7. SQL debugging and optimization
            logger.debug("Starting SQL debugging...")
            try:
                debug_prompt = self._build_debug_prompt(question, sql, processed_schema)
                debugged_sql = self._execute_llm_with_retry(debug_prompt, "SQL debugging")
                sql = f"SELECT {debugged_sql}"
                logger.debug("SQL debugging completed")
            except Exception as e:
                logger.warning(f"SQL debugging failed, keeping original SQL: {e}")

            # 8. SQL post-processing
            if self.sql_post_process_function:
                try:
                    sql = self.sql_post_process_function(sql, self.dataset)
                    logger.debug("SQL post-processing completed")
                except Exception as e:
                    logger.warning(f"SQL post-processing failed: {e}")

            # 9. Save result
            self._save_sql_result(sql, item)

            logger.debug(f"Final SQL: {sql[:100]}...")
            logger.info(f"DIN_SQLGenerator sample {item} processing completed")

            return sql

        except Exception as e:
            logger.error(f"Error occurred while processing sample {item}: {e}")
            raise

    def generate_batch(
            self,
            items: List[int],
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            **kwargs
    ) -> List[str]:
        """
        Batch generate SQL queries

        Args:
            items: List of data item indices
            schema: Database schema
            schema_links: Schema links
            **kwargs: Other parameters

        Returns:
            List of generated SQL queries
        """
        logger.info(f"Started batch processing {len(items)} samples")
        results = []

        for i, item in enumerate(items):
            try:
                logger.debug(f"Batch processing progress: {i + 1}/{len(items)}")
                sql = self.act(item, schema, schema_links, **kwargs)
                results.append(sql)
            except Exception as e:
                logger.error(f"Batch processing sample {item} failed: {e}")
                results.append("SELECT -- Error occurred during generation")

        success_count = len([r for r in results if not r.startswith('SELECT -- Error')])
        logger.info(f"Batch processing completed, successfully processed {success_count} samples")
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get generator statistics

        Returns:
            Statistics dictionary
        """
        return {
            "name": self.NAME,
            "dataset_size": len(self.dataset) if self.dataset else 0,
            "use_external": self.use_external,
            "use_few_shot": self.use_few_shot,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "save_enabled": self.is_save,
            "save_directory": str(self.save_dir),
            "llm_initialized": self.llm is not None,
            "dataset_initialized": self.dataset is not None
        }

    def __repr__(self) -> str:
        """Return string representation of the object"""
        return (
            f"{self.__class__.__name__}("
            f"dataset={'initialized' if self.dataset else 'None'}, "
            f"llm={'initialized' if self.llm else 'None'}, "
            f"save_dir='{self.save_dir}', "
            f"use_external={self.use_external})"
        )