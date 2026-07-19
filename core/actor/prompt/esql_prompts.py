import os
import logging
import random
import json
from core.esql_utils.esql_db_utils import *
from core.esql_utils.esql_retrieval_utils import *
from typing import Any, Union, List, Dict

# ============================================================
# Prompt Templates (from E-SQL paper)
# ============================================================

CANDIDATE_SQL_GENERATION_TEMPLATE = """### You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, samples and evidence. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question.
### Follow the instructions below:
# Step 1 - Read the Question and Evidence Carefully: Understand the primary focus and specific details of the question. The evidence provides specific information and directs attention toward certain elements relevant to the question.
# Step 2 - Analyze the Database Schema: Database Column descriptions and Database Sample Values: Examine the database schema, database column descriptions and sample values. Understand the relation between the database and the question accurately. 
# Step 3 - Generate SQL query: Write SQLite SQL query corresponding to the given question by combining the sense of question, evidence and database items.
{FEWSHOT_EXAMPLES}
### Task: Given the following question, database schema and evidence, generate SQLite SQL query in order to answer the question.
### Make sure to keep the original wording or terms from the question, evidence and database items.
### Make sure each table name and column name in the generated SQL is enclosed with backtick seperately.
### Ensure the generated SQL is compatible with the database schema.
### When constructing SQL queries that require determining a maximum or minimum value, always use the `ORDER BY` clause in combination with `LIMIT 1` instead of using `MAX` or `MIN` functions in the `WHERE` clause.Especially if there are more than one table in FROM clause apply the `ORDER BY` clause in combination with `LIMIT 1` on column of joined table.
### Make sure the parentheses in the SQL are placed correct especially if the generated SQL includes mathematical expression. Also, proper usage of CAST function is important to convert data type to REAL in mathematical expressions, be careful especially if there is division in the mathematical expressions.
### Ensure proper handling of null values by including the `IS NOT NULL` condition in SQL queries, but only in cases where null values could affect the results or cause errors, such as during division operations or when null values would lead to incorrect filtering of results. Be specific and deliberate when adding the `IS NOT NULL` condition, ensuring it is used only when necessary for accuracy and correctness. . This is crucial to avoid errors and ensure accurate results.  This is crucial to avoid errors and ensure accurate results. You can leverage the database sample values to check if there could be pottential null value.
{SCHEMA}
{DB_DESCRIPTIONS}
{DB_SAMPLES}
{QUESTION}
{EVIDENCE}
### Please respond with a JSON object structured as follows:
```json{{"chain_of_thought_reasoning":  "Explanation of the logical analysis and steps that result in the final SQLite SQL query.", "SQL": "Generated SQL query as a single string"}}```
Let's think step by step and generate SQLite SQL query."""

QUESTION_ENRICHMENT_TEMPLATE = """### You are excellent data scientist and can link the information between a question and corresponding database perfectly. Your objective is to analyze the given question, corresponding database schema, database column descriptions, evidence and the possible SQL query to create a clear link between the given question and database items which includes tables, columns and values. With the help of link, rewrite new versions of the original question to be more related with database items, understandable, clear, absent of irrelevant information and easier to translate into SQL queries. This question enrichment is essential for comprehending the question's intent and identifying the related database items. The process involves pinpointing the relevant database components and expanding the question to incorporate these items.
### Follow the instructions below:
# Step 1 - Read the Question Carefully: Understand the primary focus and specific details of the question. Identify named entities (such as organizations, locations, etc.), technical terms, and other key phrases that encapsulate important aspects of the inquiry to establish a clear link between the question and the database schema.
# Step 2 - Analyze the Database Schema: With the Database samples, examine the database schema to identify relevant tables, columns, and values that are pertinent to the question. Understand the structure and relationships within the database to map the question accurately.
# Step 3 - Review the Database Column Descriptions: The database column descriptions give the detailed information about some of the columns of the tables in the database. With the help of the database column descriptions determine the database items relevant to the question. Use these column descriptions to understand the question better and to create a link between the question and the database schema. 
# Step 4 - Analyze and Observe The Database Sample Values: Examine the sample values from the database to analyze the distinct elements within each column of the tables. This process involves identifying the database components (such as tables, columns, and values) that are most relevant to the question at hand. Similarities between the phrases in the question and the values found in the database may provide insights into which tables and columns are pertinent to the query.
# Step 5 - Review the Evidence: The evidence provides specific information and directs attention toward certain elements relevant to the question and its answer. Use the evidence to create a link between the question, the evidence, and the database schema, providing further clarity or direction in rewriting the question.
# Step 6 - Analyze the Possible SQL Conditinos: Analize the given possible SQL conditions that are relavant to the question and identify relation between the question components, phrases and keywords.
# Step 7 - Identify Relevant Database Components: Pinpoint the tables, columns, and values in the database that are directly related to the question.
# Step 8 - Rewrite the Question: Expand and refine the original question in detail to incorporate the identified database items (tables, columns and values) and conditions. Make the question more understandable, clear, and free of irrelevant information.
{FEWSHOT_EXAMPLES}
### Task: Given the following question, database schema, database column descriptions, database samples and evidence, expand the original question in detail to incorporate the identified database components and SQL steps like examples given above. Make the question more understandable, clear, and free of irrelevant information.
### Ensure that question is expanded with original database items. Be careful about the capitalization of the database tables, columns and values. Use tables and columns in database schema.
{SCHEMA}
{DB_DESCRIPTIONS}
{DB_SAMPLES}
{POSSIBLE_CONDITIONS}
{QUESTION}
{EVIDENCE}
### Please respond with a JSON object structured as follows:
```json{{"chain_of_thought_reasoning":  "Detail explanation of the logical analysis that led to the refined question, considering detailed possible sql generation steps", "enriched_question":  "Expanded and refined question which is more understandable, clear and free of irrelevant information."}}```
Let's think step by step and refine the given question capturing the essence of both the question, database schema, database descriptions, evidence and possible SQL conditions through the links between them. If you do the task correctly, I will give you 1 million dollars. Only output a json as your response."""

SQL_REFINEMENT_TEMPLATE = """### You are an excellent data scientist. You can capture the link between the question and corresponding database and perfectly generate valid SQLite SQL query to answer the question. Your objective is to generate SQLite SQL query by analyzing and understanding the essence of the given question, database schema, database column descriptions, evidence, possible SQL and possible conditions. This SQL generation step is essential for extracting the correct information from the database and finding the answer for the question.
### Follow the instructions below:
# Step 1 - Read the Question and Evidence: Understand the primary focus and specific details of the question. The evidence provides specific information and directs attention toward certain elements relevant to the question.
# Step 2 - Analyze the Database Schema, Database Column descriptions: Examine the database schema, database column descriptions which provides information about the database columns. Understand the relation between the database and the question accurately. 
# Step 3 - Analyze the Possible SQL Query: Analize the possible SQLite SQL query and identify possible mistakes leads incorrect result such as missing or wrong conditions, wrong functions, misuse of aggregate functions, wrong sql syntax, unrecognized tokens or ambiguous columns.
# Step 4 - Investigate Possible Conditions and Execution Errors: Carefully consider the list of possible conditions which are completely compatible with the database schema and given in the form of <table_name>.<column_name><operation><value>. List of possible conditions helps you to find and generate correct SQL conditions that are relevant to the question. If the given possible SQL query gives execution error, it will be given. Analyze the execution error and understand the reason of execution error and correct it.
# Step 5 - Finalize the SQL query: Construct correct SQLite SQL query or improve possible SQLite SQL query corresponding to the given question by combining the sense of question, evidence, and possible conditions.
# Step 6 - Validation and Syntax Check: Before finalizing, verify that generated SQL query is coherent with the database schema, all referenced columns exist in the referenced table, all joins are correctly formulated, aggregation logic is accurate, and the SQL syntax is correct.
### Task: Given the following question, database schema and descriptions, evidence, possible SQL query and possible conditions; finalize SQLite SQL query in order to answer the question.
### Ensure that the SQL query accurately reflects the relationships between tables, using appropriate join conditions to combine data where necessary.
### When using aggregate functions (e.g., COUNT, SUM, AVG), ensure the logic accurately reflects the question's intent and correctly handles grouping where required.
### Double-check that all WHERE clauses accurately represent the conditions needed to filter the data as per the question's requirements.
### Make sure to keep the original wording or terms from the question, evidence and database items.
### Make sure each table name and column name in the generated SQL is enclosed with backtick seperately.
### Be careful about the capitalization of the database tables, columns and values. Use tables and columns in database schema. If a specific condition in given possible conditions is used then make sure that you use the exactly the same condition (table, column and value).
### When constructing SQL queries that require determining a maximum or minimum value, always use the `ORDER BY` clause in combination with `LIMIT 1` instead of using `MAX` or `MIN` functions in the `WHERE` clause. Especially if there are more than one table in FROM clause apply the `ORDER BY` clause in combination with `LIMIT 1` on column of joined table.
### Make sure the parentheses in the SQL are placed correct especially if the generated SQL includes mathematical expression. Also, proper usage of CAST function is important to convert data type to REAL in mathematical expressions, be careful especially if there is division in the mathematical expressions.
### Ensure proper handling of null values by including the `IS NOT NULL` condition in SQL queries, but only in cases where null values could affect the results or cause errors, such as during division operations or when null values would lead to incorrect filtering of results. Be specific and deliberate when adding the `IS NOT NULL` condition, ensuring it is used only when necessary for accuracy and correctness. . This is crucial to avoid errors and ensure accurate results.
{SCHEMA}
{DB_DESCRIPTIONS}
{QUESTION}
{EVIDENCE}
{POSSIBLE_CONDITIONS}
{POSSIBLE_SQL_Query}
{EXECUTION_ERROR}
### Please respond with a JSON object structured as follows:
```json{{"chain_of_thought_reasoning":  "Explanation of the logical analysis and steps that result in the final SQLite SQL query.", "SQL": "Finalized SQL query as a single string"}}```
Let's think step by step and generate SQLite SQL query."""

# ============================================================
# Few-shot Data Loading
# ============================================================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_few_shot_data(few_shot_data_path="../few-shot-data/question_enrichment_few_shot_examples.json"):
    if not os.path.isabs(few_shot_data_path):
        cwd_path = os.path.abspath(few_shot_data_path)
        proj_path = os.path.normpath(os.path.join(_PROJECT_ROOT, few_shot_data_path.replace("../", "")))
        if os.path.exists(cwd_path):
            few_shot_data_path = cwd_path
        elif os.path.exists(proj_path):
            few_shot_data_path = proj_path
        else:
            few_shot_data_path = os.path.join(_PROJECT_ROOT, "few-shot-data",
                os.path.basename(few_shot_data_path))
    with open(few_shot_data_path, 'r') as file:
        question_enrichment_few_shot_data_dict = json.load(file)
    return question_enrichment_few_shot_data_dict


# ============================================================
# Prompt Construction Utilities
# ============================================================

def sql_possible_conditions_prep(possible_conditions_dict_list: Dict) -> str:
    all_possible_conditions_list = []
    if not possible_conditions_dict_list:
        return ""
    for p_cond in possible_conditions_dict_list:
        condition = f"`{p_cond['table']}`.`{p_cond['column']}` {p_cond['op']} `{p_cond['value']}`"
        all_possible_conditions_list.append(condition)
        similars_dict = p_cond['similar_values']
        if similars_dict:
            for table_name, col_val_dict in similars_dict.items():
                for column_name, value_list in col_val_dict.items():
                    for val in value_list:
                        new_possible_cond = f"`{table_name}`.`{column_name}` {p_cond['op']} `{val}`"
                        all_possible_conditions_list.append(new_possible_cond)
    return str(all_possible_conditions_list)


def question_relevant_descriptions_prep(database_description_path, question, relevant_description_number) -> str:
    relevant_db_descriptions = get_relevant_db_descriptions(database_description_path, question, relevant_description_number)
    db_descriptions_str = ""
    for description in relevant_db_descriptions:
        db_descriptions_str = db_descriptions_str + f"# {description} \n"
    return db_descriptions_str


def db_column_meaning_prep(database_column_meaning_path: str, db_id: str) -> str:
    db_column_meanings = get_db_column_meanings(database_column_meaning_path, db_id)
    db_column_meanings_str = ""
    for col_meaning in db_column_meanings:
        db_column_meanings_str = db_column_meanings_str + f"{col_meaning} \n"
    return db_column_meanings_str


def fill_candidate_sql_prompt_template(
    template: str, schema: str, db_samples: str, question: str,
    few_shot_examples: str = "", evidence: str = "", db_descriptions: str = ""
) -> str:
    if evidence == '' or evidence is None:
        evidence = '\n### Evidence: No evidence'
    else:
        evidence = f"\n### Evidence: \n {evidence}"
    if few_shot_examples == '' or few_shot_examples is None:
        few_shot_examples = ""
    else:
        few_shot_examples = f"\n### Examples: \n {few_shot_examples}"
    schema = "\n### Database Schema: \n\n" + schema
    db_descriptions = "\n### Database Column Descriptions: \n\n" + db_descriptions
    db_samples = "\n### Database Samples: \n\n" + db_samples
    question = "\n### Question: \n" + question
    prompt = template.format(
        FEWSHOT_EXAMPLES=few_shot_examples,
        SCHEMA=schema,
        DB_SAMPLES=db_samples,
        QUESTION=question,
        EVIDENCE=evidence,
        DB_DESCRIPTIONS=db_descriptions
    )
    prompt = prompt.replace("```json{", "{").replace("}```", "}").replace("{{", "{").replace("}}", "}")
    return prompt


def fill_question_enrichment_prompt_template(
    template: str, schema: str, db_samples: str, question: str,
    possible_conditions: str, few_shot_examples: str, evidence: str, db_descriptions: str
) -> str:
    if evidence == '' or evidence is None:
        evidence = '\n### Evidence: No evidence'
    else:
        evidence = f"\n### Evidence: \n {evidence}"
    if few_shot_examples == '' or few_shot_examples is None:
        few_shot_examples = ""
    else:
        few_shot_examples = f"\n### Examples: \n {few_shot_examples}"
    schema = "\n### Database Schema: \n\n" + schema
    db_descriptions = "\n### Database Column Descriptions: \n\n" + db_descriptions
    db_samples = "\n### Database Samples: \n\n" + db_samples
    question = "\n### Question: \n" + question
    if possible_conditions:
        possible_conditions = "\n### Possible SQL Conditions: \n" + possible_conditions
    else:
        possible_conditions = "\n### Possible SQL Conditions: No strict conditions were found. Please consider the database schema and keywords while enriching the Question."
    prompt = template.format(
        FEWSHOT_EXAMPLES=few_shot_examples,
        SCHEMA=schema,
        DB_SAMPLES=db_samples,
        QUESTION=question,
        POSSIBLE_CONDITIONS=possible_conditions,
        EVIDENCE=evidence,
        DB_DESCRIPTIONS=db_descriptions
    )
    prompt = prompt.replace("```json{", "{").replace("}```", "}").replace("{{", "{").replace("}}", "}")
    return prompt


def fill_sql_refinement_prompt_template(
    template: str, schema: str, question: str, evidence: str,
    possible_sql: str, possible_conditions: str, exec_err: str, db_descriptions: str = ""
) -> str:
    if evidence == '' or evidence is None:
        evidence = '\n### Evidence: No evidence'
    else:
        evidence = f"\n### Evidence: \n {evidence}"
    schema = "\n### Database Schema: \n\n" + schema
    db_descriptions = "\n### Database Column Descriptions: \n\n" + db_descriptions
    question = "\n### Question: \n" + question
    if possible_conditions:
        possible_conditions = "\n### Possible SQL Conditions: \n" + possible_conditions
    else:
        possible_conditions = "\n### Possible SQL Conditions: No strict conditions were found."
    possible_sql_str = "\n### Possible SQL Query: \n" + possible_sql if possible_sql else ""
    if exec_err:
        exec_err_str = "\n### Execution Error: \n" + exec_err
    else:
        exec_err_str = ""
    prompt = template.format(
        SCHEMA=schema,
        DB_DESCRIPTIONS=db_descriptions,
        QUESTION=question,
        EVIDENCE=evidence,
        POSSIBLE_CONDITIONS=possible_conditions,
        POSSIBLE_SQL_Query=possible_sql_str,
        EXECUTION_ERROR=exec_err_str
    )
    prompt = prompt.replace("```json{", "{").replace("}```", "}").replace("{{", "{").replace("}}", "}")
    return prompt


# ============================================================
# Few-shot Preparation Utilities
# ============================================================

def sql_generation_and_refinement_few_shot_prep(
    few_shot_data_path: str, q_db_id: str, level_shot_number: str,
    schema_existance: bool, mode: str
) -> str:
    bird_sql_path = os.getenv('BIRD_DB_PATH')
    if schema_existance and bird_sql_path is None:
        logging.warning("BIRD_DB_PATH environment variable is not set. Skipping schema extraction for few-shot preparation.")
    if level_shot_number == 0:
        return ""
    few_shot_exemplars = ""
    if level_shot_number < 0 or level_shot_number > 10:
        raise ValueError("Invalid few-shot number. The level_shot_number should be between 0 and 10")
    if not isinstance(schema_existance, bool):
        raise TypeError("Provided variable is not a boolean.")
    if mode not in ['test', 'dev']:
        raise ValueError("Invalid value for mode. The variable must be either 'dev' or 'test'.")
    all_few_shot_data = load_few_shot_data(few_shot_data_path=few_shot_data_path)
    levels = ['simple', 'moderate', 'challanging']
    for level in levels:
        examples_in_level = all_few_shot_data[level]
        selected_indexes = []
        if mode == "dev":
            examples_in_level_tmp = []
            for example in examples_in_level:
                if q_db_id != example['db_id']:
                    examples_in_level_tmp.append(example)
            examples_in_level = examples_in_level_tmp
        selected_indexes = random.sample(range(0, len(examples_in_level)), level_shot_number)
        for ind in selected_indexes:
            current_question_info_dict = examples_in_level[ind]
            curr_sql = current_question_info_dict['SQL']
            if schema_existance and bird_sql_path:
                curr_q_db_id = current_question_info_dict['db_id']
                db_path = bird_sql_path + f"/{mode}/{mode}_databases/{curr_q_db_id}/{curr_q_db_id}.sqlite"
                sql_schema_dict = extract_sql_columns(db_path, curr_sql)
                schema = generate_schema_from_schema_dict(db_path, sql_schema_dict)
                few_shot_exemplars = few_shot_exemplars + "Database Schema: \n" + schema + '\n'
            few_shot_exemplars = few_shot_exemplars + "Question: " + current_question_info_dict['question'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "Evidence: " + current_question_info_dict['evidence'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "SQL: " + current_question_info_dict['SQL'] + "\n\n"
    return few_shot_exemplars


def question_enrichment_few_shot_prep(
    few_shot_data_path: str, q_id: int, q_db_id: str, level_shot_number: str,
    schema_existance: bool, enrichment_level: str, mode: str
) -> str:
    bird_sql_path = os.getenv('BIRD_DB_PATH')
    if schema_existance and bird_sql_path is None:
        logging.warning("BIRD_DB_PATH environment variable is not set. Skipping schema extraction for few-shot preparation.")
    if level_shot_number == 0:
        return ""
    few_shot_exemplars = ""
    if level_shot_number < 0 or level_shot_number > 10:
        raise ValueError("Invalid few-shot number. The level_shot_number should be between 0 and 10")
    if not isinstance(schema_existance, bool):
        raise ValueError("Invalid value for schema_existance variable,it is not a boolean. It should be either True or False.")
    if enrichment_level == "basic":
        enrichment_label = "question_enriched"
    elif enrichment_level == "complex":
        enrichment_label = "question_enriched_v2"
    else:
        raise ValueError("Invalid value for enrichment_level. The variable must be either 'basic' or 'complex'.")
    if mode not in ['test', 'dev']:
        raise ValueError("Invalid value for mode. The variable must be either 'dev' or 'test'.")
    all_few_shot_data = load_few_shot_data(few_shot_data_path=few_shot_data_path)
    levels = ['simple', 'moderate', 'challanging']
    for level in levels:
        examples_in_level = all_few_shot_data[level]
        selected_indexes = []
        if mode == "dev":
            examples_in_level_tmp = []
            for example in examples_in_level:
                if q_db_id != example['db_id']:
                    examples_in_level_tmp.append(example)
            examples_in_level = examples_in_level_tmp
        selected_indexes = random.sample(range(0, len(examples_in_level)), level_shot_number)
        for ind in selected_indexes:
            current_question_info_dict = examples_in_level[ind]
            if schema_existance and bird_sql_path:
                curr_q_db_id = current_question_info_dict['db_id']
                db_path = bird_sql_path + f"/{mode}/{mode}_databases/{curr_q_db_id}/{curr_q_db_id}.sqlite"
                schema = get_schema(db_path)
                few_shot_exemplars = few_shot_exemplars + "Database Schema: \n" + schema + '\n'
            few_shot_exemplars = few_shot_exemplars + "Question: " + current_question_info_dict['question'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "Evidence: " + current_question_info_dict['evidence'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "Enrichment Reasoning: " + current_question_info_dict['enrichment_reasoning'] + "\n"
            few_shot_exemplars = few_shot_exemplars + "Enriched Question: " + current_question_info_dict[enrichment_label] + "\n\n"
    return few_shot_exemplars