from llama_index.core.llms.llm import LLM
from typing import Union, List, Callable, Dict, Optional
import pandas as pd
from os import PathLike
from pathlib import Path
from loguru import logger

from core.actor.generator.BaseGenerate import BaseGenerator
from core.data_manage import Dataset, single_central_process
from core.actor.reducer.LinkAlignReduce import LinkAlignReducer
from core.actor.parser.LinkAlignParse import LinkAlignParser
from core.actor.generator.sql_debug import sql_debug_by_experience, sql_debug_by_feedback
from core.utils import (
    parse_schema_from_df,
    load_dataset,
    save_dataset
)


class LinkAlignGenerator(BaseGenerator):
    """ We adapt the DIN-SQL method to scalable real-world db environment by applying the LinkAlign framework """

    NAME = "LinkAlignGenerator"

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            reducer: Optional[LinkAlignReducer] = None,
            parser: Optional[LinkAlignParser] = None,
            is_save: bool = True,
            save_dir: Union[str, PathLike] = "../files/pred_sql",
            use_external: bool = True,
            use_few_shot: bool = True,
            sql_post_process_function: Optional[Callable] = None,
            use_feedback_debug: bool = True,
            debug_turn_n: int = 3,
            db_path: Optional[Union[str, PathLike]] = None,
            credential: Optional[Dict] = None,
            **kwargs
    ):
        self.dataset: Optional[Dataset] = dataset
        self.llm: Optional[LLM] = llm
        self.reducer = reducer
        self.parser = parser
        self.is_save = is_save
        self.save_dir: Union[str, PathLike] = save_dir
        self.use_external: bool = use_external
        self.use_few_shot: bool = use_few_shot

        self.sql_post_process_function: Optional[Callable] = sql_post_process_function
        self.use_feedback_debug: bool = use_feedback_debug
        self.debug_turn_n: int = debug_turn_n

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

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    @property
    def classification_prompt(self):
        classification_prompt = '''### Here are some reference examples:
# 
Q: "How many courses that do not have prerequisite?"
schema_links: [course.*,course.course_id = prereq.course_id]
A: Let’s think step by step. The SQL query for the question "How many courses that do not have prerequisite?" needs these tables = [course,prereq], so we need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Which courses have prerequisite?"].
So, we need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"
#
Q: "Find the title of course that is provided by both Statistics and Psychology departments."
schema_links: [course.title,course.dept_name,Statistics,Psychology]
A: Let’s think step by step. The SQL query for the question "Find the title of course that is provided by both Statistics and Psychology departments." needs these tables = [course], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Find the titles of courses that is provided by Psychology departments"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"
#
Q: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
schema_links: [teaches.id,teaches.semester,teaches.year,Fall,2009,Spring,2010]
A: Let’s think step by step. The SQL query for the question "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010." needs these tables = [teaches], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Find the id of instructors who taught a class in Spring 2010"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"
#
Q: "Give the name and building of the departments with greater than average budget."
schema_links: [department.budget,department.dept_name,department.building]
A: Let’s think step by step. The SQL query for the question "Give the name and building of the departments with greater than average budget." needs these tables = [department], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["What is the average budget of the departments"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"
#
'''
        return classification_prompt

    def classification_prompt_maker(
            self,
            question: str,
            schema: str,
            schema_links: Union[str, List] = "None",
    ):

        instruction = """# [Instruction]
For the given question, classify it as NESTED. 
Break down the problem into sub-problems and list them in the `List` format: questions = [q1,q2,q3..], e.g. questions = ['Which courses have prerequisite?']
"""
        prompt = (
            f"{instruction}"
            f"{schema}\n"
            f"{self.classification_prompt}\n"
            f"Question: {question}\n"
            f"schema_links: {schema_links}\n"
            "A: Let’s think step by step."
        )

        return prompt

    @property
    def hard_prompt(self):
        hard_prompt = '''### Here are some reference examples:
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

# [Question]: "Find the title, credit, and department name of courses that have more than one prerequisites?"
# [Schema links]: [course.title,course.credits,course.dept_name,course.course_id = prereq.course_id]
# [Analysis]: Let's think step by step. "Find the title, credit, and department name of courses that have more than one prerequisites?" can be solved by knowing the answer to the following sub-question "What is the title, credit value, and department name for courses with more than one prerequisite?".
The SQL query for the sub-question "What is the title, credit value, and department name for courses with more than one prerequisite?" is SELECT T1.title ,  T1.credits , T1.dept_name FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  >  1
So, the answer to the question "Find the name and building of the department with the highest budget." is =
Intermediate_representation: select course.title , course.credits , course.dept_name from course  where  count ( prereq.* )  > 1  group by prereq.course_id 
# [Sql]: SELECT T1.title ,  T1.credits , T1.dept_name FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  >  1

# [Question]: "Give the name and building of the departments with greater than average budget."
# [Schema links]: [department.dept_name,department.building,department.budget]
# [Analysis]: Let's think step by step. "Give the name and building of the departments with greater than average budget." can be solved by knowing the answer to the following sub-question "What is the average budget of departments?".
The SQL query for the sub-question "What is the average budget of departments?" is SELECT avg(budget) FROM department
So, the answer to the question "Give the name and building of the departments with greater than average budget." is =
Intermediate_representation: select department.dept_name , department.building from department  where  @.@ > avg ( department.budget )
# [Sql]: SELECT dept_name ,  building FROM department WHERE budget  >  (SELECT avg(budget) FROM department)

###
'''
        return hard_prompt

    def hard_prompt_maker(
            self,
            question: str,
            schema: str,
            sub_questions: str,
            schema_links: Union[str, List] = "None",
            reasoning_examples: str = None
    ) -> str:
        instruction = """[Instructions]
Use the intermediate representation, schema links, and the provided prior knowledge (including field and table information) to generate the correct SQL queries for each question. The SQL queries must be syntactically correct and logically aligned with the requirements of the question. 
You need to follow below requirements:
1. Understand the question: Carefully analyze the question to identify the relevant data and the required result.
2. Consult the schema: Use the schema links provided to identify the tables, fields, and relationships (including foreign keys and primary keys) necessary to answer the question.
3. Leverage prior knowledge: Utilize any domain-specific knowledge, field names, table relationships, and query logic to craft an accurate SQL query.
4. Use intermediate representations: Where applicable, break down the query into logical components such as CTEs (Common Table Expressions), subqueries, and joins, ensuring that each part of the query is clearly derived from the question and schema.
5. Adhere to DBMS syntax: Ensure that the SQL queries comply with the syntax specifications of {dbms_name}. Pay attention to common SQL conventions, such as SELECT, JOIN, WHERE, GROUP BY, and ORDER BY clauses, and ensure correct use of aggregate functions and data types.
6. Correct complex queries: For complex queries, use appropriate techniques (e.g., CTEs, subqueries) to avoid errors and improve readability.
7. Return only the SQL query: Provide the final, corrected SQL query without any explanations.
"""

        example_prompt = reasoning_examples if reasoning_examples else self.hard_prompt
        step_reasoning = (
            f"Let's think step by step. "
            f"Question can be solved by knowing the answer to the following sub-question \"{sub_questions}\"."
        )
        prompt = (
            f"{instruction}\n\n"
            f"### [Question]: {question}\n"
            f"### [Provided Database Schema]:\n{schema}\n"
            f"### [Relevant Examples]: \n{example_prompt}\n\n"
            "### [Process Begin]\n"
            f'# [Question]: "{question}"\n'
            f"# [Schema links]: {str(schema_links)}\n"
            f"# [Analysis]: {step_reasoning}\n"
            "# Only output SQL query:"
        )

        return prompt

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        logger.info(f"LinkAlignGenerator 开始处理样本 {item}")
        row = self.dataset[item]
        question = row['question']
        db_type = row['db_type']
        db_id = row["db_id"]
        db_path = Path(self.db_path) / (db_id + ".sqlite") if self.db_path else self.db_path
        logger.debug(f"处理问题: {question[:100]}... (数据库: {db_id}, 类型: {db_type})")

        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external", None))
            if external_knowledge:
                question += "\n" + external_knowledge
                logger.debug("已加载外部知识")

        # Use LinkAlign to reduce the dimensionality of database schema
        logger.debug("开始处理数据库模式...")
        if isinstance(schema, (str, PathLike)) and Path(schema).exists():
            schema = load_dataset(schema)

        # Try to load schema if not provided
        if schema is None:
            instance_schema_path = row.get("instance_schemas")
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
                logger.debug(f"从实例模式路径加载模式: {instance_schema_path}")

            if schema is None:
                logger.debug("从数据集获取数据库模式")
                schema = self.dataset.get_db_schema(item)
                reducer = self.reducer or LinkAlignReducer(self.dataset, self.llm)
                logger.debug("使用 LinkAlignReducer 降维模式")
                schema = reducer.act(item, schema)

            if schema is None:
                raise Exception("Failed to load a valid database schema for the sample!")

        # Normalize schema type
        if isinstance(schema, dict):
            schema = single_central_process(schema)
        elif isinstance(schema, list):
            schema = pd.DataFrame(schema)

        if isinstance(schema, pd.DataFrame):
            origin_schema = schema
            schema = parse_schema_from_df(schema)
        else:
            raise Exception("Failed to load a valid database schema for the sample!")

        logger.debug("数据库模式处理完成")

        # The following follows the DIN-SQL processing procedure
        # Step 1: schema linking
        logger.debug("开始模式链接...")
        if schema_links is None:
            schema_link_path = row.get("schema_links", None)
            if schema_link_path:
                schema_links = load_dataset(schema_link_path)
                logger.debug(f"从路径加载模式链接: {schema_link_path}")
            else:
                logger.debug("使用 LinkAlignParser 生成模式链接")
                parser = LinkAlignParser(self.dataset, self.llm) if not self.parser else self.parser
                schema_links = parser.act(item, origin_schema)

        if data_logger:
            data_logger.info(f"{self.NAME}.schema linking output | {schema_links}")
        # Step 2: difficulty classification
        logger.debug("开始难度分类...")
        try:
            class_prompt = self.classification_prompt_maker(question, schema, schema_links)
            classification = self.llm.complete(class_prompt).text
            logger.debug("难度分类完成")
        except Exception as e:
            logger.error(f"难度分类失败: {e}")
            print(e)
            raise e
        try:
            sub_questions = classification.split('questions = [')[1].split(']')[0]
            logger.debug(f"解析子问题: {sub_questions}")
        except Exception as e:
            logger.warning(f'解析子问题时出错，作为非嵌套处理: {e}')
            print('warning: error when parsing sub_question. treat it as Non-Nested. error:', e)
            sub_questions = classification

        # step 3: SQL generation
        logger.debug("开始 SQL 生成...")
        # load reasoning examples
        reasoning_examples = None
        if self.use_few_shot:
            reasoning_example_path = row.get("reasoning_examples", None)
            if reasoning_example_path:
                reasoning_examples = load_dataset(reasoning_example_path)
                logger.debug(f"加载推理示例: {reasoning_example_path}")

        try:
            hard_prompt = self.hard_prompt_maker(question, schema, sub_questions, schema_links, reasoning_examples)
            sql = self.llm.complete(hard_prompt).text
            sql_list = [sql]
            logger.debug("SQL 生成完成")
        except Exception as e:
            logger.error(f"SQL 生成失败: {e}")
            print(e)
            raise e

        if data_logger:
            data_logger.info(f"{self.NAME}.predict sql output | {sql_list}")

        # LinkAlign: SQL debugging by feedback
        if self.use_feedback_debug:
            if data_logger:
                data_logger.info(f"{self.NAME}: begin use_feedback_debug")
            for idx, sql in enumerate(sql_list):
                debug_args = {
                    "question": question,
                    "schema": schema,
                    "sql_query": sql,
                    "llm": self.llm,
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_type": db_type,
                    "credential": self.credential,
                    "debug_turn_n": self.debug_turn_n
                }
                if data_logger:
                    data_logger.info(f"{self.NAME}: debug arguments | {str(debug_args)}")
                _, debugged_sql = sql_debug_by_feedback(**debug_args)
                sql_list[idx] = debugged_sql
            logger.debug("基于反馈的 SQL 调试完成")

        # Select the Winner SQL
        pred_sql = sql_list[0]
        logger.debug(f"最终 SQL: {pred_sql[:100]}...")

        pred_sql = self.save_output(pred_sql, item, row.get("instance_id"))

        logger.info(f"LinkAlignGenerator 样本 {item} 处理完成")
        if data_logger:
            data_logger.info(f"{self.NAME}.final_sql | sql={pred_sql}")
            data_logger.info(f"{self.NAME}.act end | item={item}")
        return pred_sql
