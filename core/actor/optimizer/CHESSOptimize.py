from typing import Union, List, Optional, Dict
from pathlib import Path
from loguru import logger
from llama_index.core.llms.llm import LLM
import re
import concurrent.futures
import pandas as pd

from core.actor.optimizer.BaseOptimize import BaseOptimizer
from core.data_manage import Dataset, load_dataset, save_dataset, single_central_process
from core.db_connect import get_sql_exec_result
from core.utils import sql_clean, parse_schema_from_df


class CHESSOptimizer(BaseOptimizer):
    """Optimizer that debugs and refines SQL queries using feedback-based methods from CHESS-SQL."""

    NAME = "CHESSOptimizer"

    REVISE_TEMPLATE = '''**Task Description:**
You are an SQL database expert tasked with correcting a SQL query. A previous attempt to run a query did not yield the correct results, either due to errors in execution or because the result returned was empty or unexpected. Your role is to analyze the error based on the provided database schema and the details of the failed execution, and then provide a corrected version of the SQL query.

**Procedure:**
1. Review Database Schema:
	- Examine the table creation statements to understand the database structure.
2. Analyze Query Requirements:
	- Original Question: Consider what information the query is supposed to retrieve.
	- Hint: Use the provided hints to understand the relationships and conditions relevant to the query.
	- Executed SQL Query: Review the SQL query that was previously executed and led to an error or incorrect result.
	- Execution Result: Analyze the outcome of the executed query to identify why it failed (e.g., syntax errors, incorrect column references, logical mistakes).
3. Correct the Query: 
	- Modify the SQL query to address the identified issues, ensuring it correctly fetches the requested data according to the database schema and query requirements.


**Output Format:**

Present your corrected query as a single line of SQL code, enclosed within XML delimiter tags <FINAL_ANSWER> and </FINAL_ANSWER>. Ensure there are no line breaks within the query.


Here are some examples:
======= Example 1 =======
**************************
【Table creation statements】
CREATE TABLE generalinfo
(
	id_restaurant INTEGER not null primary key,
	food_type TEXT null, -- examples: `thai`| `food type` description: the food type
	city TEXT null, -- description: the city where the restaurant is located in
);

CREATE TABLE location
(
	id_restaurant INTEGER not null primary key,
	street_name TEXT null, -- examples: `ave`, `san pablo ave`, `pablo ave`| `street name` description: the street name of the restaurant
	city TEXT null, -- description: the city where the restaurant is located in
	foreign key (id_restaurant) references generalinfo (id_restaurant) on update cascade on delete cascade,
);

**************************
【Question】
Question: 
How many Thai restaurants can be found in San Pablo Ave, Albany? 

Evidence:
Thai restaurant refers to food_type = 'thai'; San Pablo Ave Albany refers to street_name = 'san pablo ave' AND T1.city = 'albany'

The SQL query executed was:
SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND T2.street = 'san pablo ave'

The execution result:
Error: no such column: T2.street

**************************

Step 1: Review Database Schema
The database comprises two tables:
1. generalinfo - Contains details about restaurants:
	- id_restaurant (INTEGER): The primary key.
	- food_type (TEXT): The type of food the restaurant serves.
	- city (TEXT): The city where the restaurant is located.
	- location - Contains the location specifics of each restaurant:

2. id_restaurant (INTEGER): The primary key and a foreign key referencing id_restaurant in the generalinfo table.
	- street_name (TEXT): The street where the restaurant is located.
	- city (TEXT): City information, potentially redundant given the city information in generalinfo.

Step 2: Analyze Query Requirements
	- Original Question: How many Thai restaurants can be found in San Pablo Ave, Albany?
	- Hints for Construction:
		- "Thai restaurant" is defined by food_type = 'thai'.
		- "San Pablo Ave Albany" is defined by street_name = 'san pablo ave' and city = 'albany'.
	- Executed SQL Query:
		- SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND T2.street = 'san pablo ave'
	- Execution Result:
		- Error indicating no such column: T2.street.
	- Analysis of Error:
		- The error message no such column: T2.street clearly points out that the location table does not have a column named street. Instead, it has a column named street_name. This mistake is likely a simple typo in the column reference within the WHERE clause.

Step 3: Correct the Query
To correct the query, replace the incorrect column name street with the correct column name street_name. Also, ensure that the city condition (T1.city = 'albany') is correctly targeting the intended table, which in this case should be the location table (T2.city), as it's more specific to the address.
<FINAL_ANSWER>
SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND T2.street_name = 'san pablo ave'
</FINAL_ANSWER> 

===== Example 2 ========
**************************
【Table creation statements】
CREATE TABLE businesses
(
        `business_id` INTEGER NOT NULL,
        `name` TEXT NOT NULL, -- description: the name of the eatery
        PRIMARY KEY (`business_id`),
);

CREATE TABLE inspections
(
        `business_id` INTEGER NOT NULL, -- `business id` description: the unique id of the business
        `score` INTEGER DEFAULT NULL, -- description: the inspection score
        `date` DATE NOT NULL, -- examples: `2014-01-24`
        FOREIGN KEY (`business_id`) REFERENCES `businesses` (`business_id`),
);

CREATE TABLE violations
(
        `business_id` INTEGER NOT NULL,
        `date` DATE NOT NULL, -- examples: `2016-05-03`
        FOREIGN KEY (`business_id`) REFERENCES `businesses` (`business_id`),
);


**************************
【Question】
Question: 
What are the names of the establishments that met all the required standards for 4 consecutive years? 

Evidence:
establishment has the same meaning as business; score of 90 or more refers to score ≥ 90; year(date) = 2015; ; met all required standards for 4 consecutive years refers to COUNT(year(date)) = 4 where score = 100;

The SQL query executed was:
SELECT DISTINCT T1.name FROM ( SELECT T3.name, T3.years, row_number() OVER (PARTITION BY T3.name ORDER BY T3.years) AS rowNumber FROM ( SELECT DISTINCT name, STRFTIME('%Y', `date`) AS years FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score = 100 ) AS T3 ) AS T4 GROUP BY T4.name, date(T4.years || '-01-01', '-' || (T4.rowNumber - 1) || ' years') HAVING COUNT(T4.years) = 4

The execution result:
Error: T1.name is not found in table T1

**************************

Step 1: Review Database Schema
The database includes three tables relevant to the query:
1. businesses:
	- business_id (INTEGER): Unique identifier for each business, primary key.
	- name (TEXT): Name of the business.

2. inspections:
	- business_id (INTEGER): Links to business_id in the businesses table.
	- score (INTEGER): Inspection score.
	- date (DATE): Date of the inspection.

3. violations:
	- business_id (INTEGER): Links to business_id in the businesses table.
	- date (DATE): Date of the violation.

Step 2: Analyze Query Requirements
	- Original Question: What are the names of the establishments that met all the required standards for 4 consecutive years?
	- Hints for Construction:
		- "Establishment" refers to businesses.
		- "Score of 90 or more" and the requirement of "score = 100" for meeting standards.
		- The businesses must have met the standards over four consecutive years.
	- Executed SQL Query:
		- SELECT DISTINCT T1.name FROM ( SELECT T3.name, T3.years, row_number() OVER (PARTITION BY T3.name ORDER BY T3.years) AS rowNumber FROM ( SELECT DISTINCT name, STRFTIME('%Y', `date`) AS years FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score = 100 ) AS T3 ) AS T4 GROUP BY T4.name, date(T4.years || '-01-01', '-' || (T4.rowNumber - 1) || ' years') HAVING COUNT(T4.years) = 4
	- Execution Result:
		- Error: T1.name is not found in table T1.
	- Analysis of Error
		- The error arises because the alias T1 is used outside its scope, causing confusion about which table or subquery the name column should be sourced from.

Step 3: Correct the Query
The objective is to simplify the query and correctly refer to column names and aliases.
<FINAL_ANSWER>
SELECT DISTINCT T4.name FROM ( SELECT T3.name, T3.years, row_number() OVER (PARTITION BY T3.name ORDER BY T3.years) AS rowNumber FROM ( SELECT DISTINCT name, STRFTIME('%Y', `date`) AS years FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score = 100 ) AS T3 ) AS T4 GROUP BY T4.name, date(T4.years || '-01-01', '-' || (T4.rowNumber - 1) || ' years') HAVING COUNT(T4.years) = 4
</FINAL_ANSWER>

======= Example 3 =======
**************************
【Database Info】
CREATE TABLE games
(
	id INTEGER not null primary key,
	games_year INTEGER default NULL, -- `games year` description: the year of the game
);

CREATE TABLE games_city
(
	games_id INTEGER default NULL,
	city_id INTEGER default NULL, -- `city id` description: the id of the city that held the game Maps to city(id)
	foreign key (city_id) references city(id),
	foreign key (games_id) references games(id),
);

CREATE TABLE city
(
	id INTEGER not null primary key,
	city_name TEXT default NULL, -- examples: `London`
);

**************************
【Question】
Question:
From 1900 to 1992, how many games did London host?

Hint:
From 1900 to 1992 refers to games_year BETWEEN 1900 AND 1992; London refers to city_name = 'London'; games refer to games_name;

The SQL query executed was:
SELECT COUNT(T3.id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = 'london' AND T3.games_year BETWEEN 1900 AND 1992

The execution result:
[]

**************************

Step 1: Review Database Schema
The database includes three tables that are relevant to the query:
1. games:
	- id (INTEGER): Primary key, representing each game's unique identifier.
	- games_year (INTEGER): The year the game was held.

2. games_city:
	- games_id (INTEGER): Foreign key linking to games(id).
	- city_id (INTEGER): Foreign key linking to city(id).

3.city:
	- id (INTEGER): Primary key, representing each city's unique identifier.
	- city_name (TEXT): Name of the city.

Step 2: Analyze Query Requirements
	- Original Question: From 1900 to 1992, how many games did London host?
	- Hints for Construction:
		- Time frame specified as 1900 to 1992.
		- London is specified by city_name = 'London'.
	- Executed SQL Query:
		- SELECT COUNT(T3.id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = 'london' AND T3.games_year BETWEEN 1900 AND 1992
	- Execution Result:
		- The result returned an empty set [].
	- Analysis of Error:
		- The query was structurally correct but failed to return results possibly due to:
			- Case sensitivity in SQL: The city name 'london' was used instead of 'London', which is case-sensitive and might have caused the query to return no results if the database treats strings as case-sensitive.
			- Data availability or integrity issues, which we cannot verify without database access, but for the purpose of this exercise, we will focus on correcting potential issues within the query itself.

Step 3: Correct the Query
Correcting the potential case sensitivity issue and ensuring the query is accurately targeted:
<FINAL_ANSWER>
SELECT COUNT(T3.id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = 'London' AND T3.games_year BETWEEN 1900 AND 1992
</FINAL_ANSWER>

======= Your task =======
**************************
【Table creation statements】
{DATABASE_SCHEMA}

**************************
The original question is:
Question: 
{QUESTION}

Evidence:
{HINT}

The SQL query executed was:
{QUERY}

The execution result:
{RESULT}

**************************
Based on the question, table schemas and the previous query, analyze the result try to fix the query.

Give very detailed analysis first. When you are OK with the fixed query, output the query string ONLY inside the xml delimiter <FINAL_ANSWER></FINAL_ANSWER>.
Inside the xml delimiter it should be the query in plain text. You cannot modify the database schema or the question, just output the corrected query.
Make sure you only output one single query. The query should be a one liner without any line breaks.

Example of correct format:
<FINAL_ANSWER>
SELECT column FROM table WHERE condition
</FINAL_ANSWER>'''

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/optimized_sql",
            debug_turn_n: int = 3,
            open_parallel: bool = True,
            max_workers: Optional[int] = None,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.debug_turn_n = debug_turn_n

    def _revise_sql(
            self,
            question: str,
            schema: str,
            sql: str,
            result: str,
            evidence: str = ""
    ) -> str:
        """Revise SQL query based on execution feedback using CHESS prompt."""
        prompt = self.REVISE_TEMPLATE.format(
            DATABASE_SCHEMA=schema,
            QUESTION=question,
            HINT=evidence,
            QUERY=sql,
            RESULT=result
        )

        response = self.llm.complete(prompt).text

        match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', response, re.DOTALL)
        if match:
            return sql_clean(match.group(1).strip())
        return sql

    def _optimize_single_sql(
            self,
            sql: str,
            question: str,
            schema: str,
            evidence: str,
            db_type: str,
            db_id: Optional[str] = None,
            db_path: Optional[Union[str, Path]] = None,
            credential: Optional[Dict] = None
    ) -> str:
        """Optimize a single SQL query using feedback from execution."""
        debug_args = {
            "db_type": db_type,
            "sql_query": sql_clean(sql),
            "db_path": db_path,
            "db_id": db_id
        }
        # Add credential_path for any database type if credential is provided
        if isinstance(credential, dict) and db_type in credential:
            debug_args["credential_path"] = credential.get(db_type)
        else:
            debug_args["credential_path"] = credential

        for turn in range(self.debug_turn_n):
            exec_result = get_sql_exec_result(**debug_args)

            if isinstance(exec_result, tuple):
                if len(exec_result) == 3:
                    res, err, _ = exec_result
                elif len(exec_result) == 2:
                    res, err = exec_result
                else:
                    res = exec_result
                    err = None
            else:
                res = exec_result
                err = None

            if err is None and res is not None and (not isinstance(res, pd.DataFrame) or not res.empty):
                return debug_args["sql_query"]

            result_str = str(err) if err else "[]"

            revised_sql = self._revise_sql(question, schema, debug_args["sql_query"], result_str, evidence)

            debug_args["sql_query"] = revised_sql

        return debug_args["sql_query"]

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            pred_sql: Union[str, Path, List[str], List[Path]] = None,
            **kwargs
    ):
        logger.info(f"CHESSOptimizer processing item {item}")

        if self.dataset is None:
            raise ValueError("Dataset is required for CHESSOptimizer")

        row = self.dataset[item]
        question = row['question']
        db_type = row.get('db_type', 'sqlite')
        db_id = row.get("db_id")
        evidence = row.get('evidence', '')

        db_path = None
        if hasattr(self.dataset, 'db_path') and self.dataset.db_path:
            db_path = Path(self.dataset.db_path) / f"{db_id}.sqlite"

        credential = self.dataset.credential if hasattr(self.dataset, 'credential') else None

        # Load and process schema using base class method
        schema_str = self.process_schema(schema, item)

        # Load pred_sql using base class method
        sql_list, is_single = self.load_pred_sql(pred_sql, item)

        def process_sql(sql):
            return self._optimize_single_sql(
                sql, question, schema_str, evidence, db_type, db_id, db_path, credential
            )

        optimized_sqls = []
        if self.open_parallel and len(sql_list) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_sql, sql) for sql in sql_list]
                for future in concurrent.futures.as_completed(futures):
                    optimized_sqls.append(future.result())
        else:
            for sql in sql_list:
                optimized_sqls.append(process_sql(sql))

        # Save results using base class method
        output = self.save_results(optimized_sqls, is_single, item, row.get("instance_id"))

        logger.info(f"CHESSOptimizer completed processing item {item}")
        return output
