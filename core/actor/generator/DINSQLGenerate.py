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
Schema_links: [classroom.building,classroom.capacity,50]

Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Q: "How many heads of the departments are older than 56 ?"
A: Let's think step by step. In the question "How many heads of the departments are older than 56 ?", we are asked:
"How many heads of the departments" so we need column = [head.*]
"older" so we need column = [head.age]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [56]. So the Schema_links are:
Schema_links: [head.*,head.age,56]

Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
Table head, columns = [*,head_ID,name,born_state,age]
Table management, columns = [*,department_ID,head_ID,temporary_acting]
Foreign_keys = [management.head_ID = head.head_ID,management.department_ID = department.Department_ID]
Q: "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"
A: Let's think step by step. In the question "what are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?", we are asked:
"distinct creation years of the departments" so we need column = [department.Creation]
"departments managed by" so we need column = [management.department_ID]
"born in" so we need column = [head.born_state]
Based on the columns and tables, we need these Foreign_keys = [department.Department_ID = management.department_ID,management.head_ID = head.head_ID].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = ['Alabama']. So the Schema_links are:
Schema_links: [department.Creation,department.Department_ID = management.department_ID,head.head_ID = management.head_ID,head.born_state,'Alabama']

Table Addresses, columns = [*,address_id,line_1,line_2,city,zip_postcode,state_province_county,country]
Table Candidate_Assessments, columns = [*,candidate_id,qualification,assessment_date,asessment_outcome_code]
Table Candidates, columns = [*,candidate_id,candidate_details]
Table Courses, columns = [*,course_id,course_name,course_description,other_details]
Table People, columns = [*,person_id,first_name,middle_name,last_name,cell_mobile_number,email_address,login_name,password]
Table People_Addresses, columns = [*,person_address_id,person_id,address_id,date_from,date_to]
Table Student_Course_Attendance, columns = [*,student_id,course_id,date_of_attendance]
Table Student_Course_Registrations, columns = [*,student_id,course_id,registration_date]
Table Students, columns = [*,student_id,student_details]
Foreign_keys = [Students.student_id = People.person_id,People_Addresses.address_id = Addresses.address_id,People_Addresses.person_id = People.person_id,Student_Course_Registrations.course_id = Courses.course_id,Student_Course_Registrations.student_id = Students.student_id,Student_Course_Attendance.student_id = Student_Course_Registrations.student_id,Student_Course_Attendance.course_id = Student_Course_Registrations.course_id,Candidates.candidate_id = People.person_id,Candidate_Assessments.candidate_id = Candidates.candidate_id]
Q: "List the id of students who never attends courses?"
A: Let's think step by step. In the question "List the id of students who never attends courses?", we are asked:
"id of students" so we need column = [Students.student_id]
"never attends courses" so we need column = [Student_Course_Attendance.student_id]
Based on the columns and tables, we need these Foreign_keys = [Students.student_id = Student_Course_Attendance.student_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
Schema_links: [Students.student_id = Student_Course_Attendance.student_id]

Table city, columns = [*,City_ID,Official_Name,Status,Area_km_2,Population,Census_Ranking]
Table competition_record, columns = [*,Competition_ID,Farm_ID,Rank]
Table farm, columns = [*,Farm_ID,Year,Total_Horses,Working_Horses,Total_Cattle,Oxen,Bulls,Cows,Pigs,Sheep_and_Goats]
Table farm_competition, columns = [*,Competition_ID,Year,Theme,Host_city_ID,Hosts]
Foreign_keys = [farm_competition.Host_city_ID = city.City_ID,competition_record.Farm_ID = farm.Farm_ID,competition_record.Competition_ID = farm_competition.Competition_ID]
Q: "Show the status of the city that has hosted the greatest number of competitions."
A: Let's think step by step. In the question "Show the status of the city that has hosted the greatest number of competitions.", we are asked:
"the status of the city" so we need column = [city.Status]
"greatest number of competitions" so we need column = [farm_competition.*]
Based on the columns and tables, we need these Foreign_keys = [farm_competition.Host_city_ID = city.City_ID].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = []. So the Schema_links are:
Schema_links: [city.Status,farm_competition.Host_city_ID = city.City_ID,farm_competition.*]

Table advisor, columns = [*,s_ID,i_ID]
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
Q: "Give the title of the course offered in Chandler during the Fall of 2010."
A: Let's think step by step. In the question "Give the title of the course offered in Chandler during the Fall of 2010.", we are asked:
"title of the course" so we need column = [course.title]
"course offered in Chandler" so we need column = [SECTION.building]
"during the Fall" so we need column = [SECTION.semester]
"of 2010" so we need column = [SECTION.year]
Based on the columns and tables, we need these Foreign_keys = [course.course_id = SECTION.course_id].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [Chandler,Fall,2010]. So the Schema_links are:
Schema_links: [course.title,course.course_id = SECTION.course_id,SECTION.building,SECTION.year,SECTION.semester,Chandler,Fall,2010]

Table city, columns = [*,City_ID,Official_Name,Status,Area_km_2,Population,Census_Ranking]
Table competition_record, columns = [*,Competition_ID,Farm_ID,Rank]
Table farm, columns = [*,Farm_ID,Year,Total_Horses,Working_Horses,Total_Cattle,Oxen,Bulls,Cows,Pigs,Sheep_and_Goats]
Table farm_competition, columns = [*,Competition_ID,Year,Theme,Host_city_ID,Hosts]
Foreign_keys = [farm_competition.Host_city_ID = city.City_ID,competition_record.Farm_ID = farm.Farm_ID,competition_record.Competition_ID = farm_competition.Competition_ID]
Q: "Show the status shared by cities with population bigger than 1500 and smaller than 500."
A: Let's think step by step. In the question "Show the status shared by cities with population bigger than 1500 and smaller than 500.", we are asked:
"the status shared by cities" so we need column = [city.Status]
"cities with population" so we need column = [city.Population]
Based on the columns and tables, we need these Foreign_keys = [].
Based on the tables, columns, and Foreign_keys, The set of possible cell values are = [1500,500]. So the Schema_links are:
Schema_links: [city.Status,city.Population,1500,500]'''

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

Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100]
A: Let's think step by step. The SQL query for the question "Find the room number of the rooms which can sit 50 to 100 students and their buildings." needs these tables = [classroom], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

Q: "How many courses that do not have prerequisite?"
schema_links: [course.*,course.course_id = prereq.course_id]
A: Let's think step by step. The SQL query for the question "How many courses that do not have prerequisite?" needs these tables = [course,prereq], so we need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Which courses have prerequisite?"].
So, we need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: "Find the title of course that is provided by both Statistics and Psychology departments."
schema_links: [course.title,course.dept_name,Statistics,Psychology]
A: Let's think step by step. The SQL query for the question "Find the title of course that is provided by both Statistics and Psychology departments." needs these tables = [course], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Find the titles of courses that is provided by Psychology departments"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
schema_links: [teaches.id,teaches.semester,teaches.year,Fall,2009,Spring,2010]
A: Let's think step by step. The SQL query for the question "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010." needs these tables = [teaches], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["Find the id of instructors who taught a class in Spring 2010"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"

Q: "Find the name of the department that offers the highest total credits?"
schema_links: [course.dept_name,course.credits]
A: Let's think step by step. The SQL query for the question "Find the name of the department that offers the highest total credits?." needs these tables = [course], so we don't need JOIN.
Plus, it doesn't require nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we don't need JOIN and don't need nested queries, then the the SQL query can be classified as "EASY".
Label: "EASY"

Q: "What is the name of the instructor who advises the student with the greatest number of total credits?"
schema_links: [advisor.i_id = instructor.id,advisor.s_id = student.id,instructor.name,student.tot_cred ]
A: Let's think step by step. The SQL query for the question "What is the name of the instructor who advises the student with the greatest number of total credits?" needs these tables = [advisor,instructor,student], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

Q: "Find the total number of students and total number of instructors for each department."
schema_links = [department.dept_name = instructor.dept_name,student.id,student.dept_name = department.dept_name,instructor.id]
A: Let's think step by step. The SQL query for the question "Find the total number of students and total number of instructors for each department." needs these tables = [department,instructor,student], so we need JOIN.
Plus, it doesn't need nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = [""].
So, we need JOIN and don't need nested queries, then the the SQL query can be classified as "NON-NESTED".
Label: "NON-NESTED"

Q: "Give the name and building of the departments with greater than average budget."
schema_links: [department.budget,department.dept_name,department.building]
A: Let's think step by step. The SQL query for the question "Give the name and building of the departments with greater than average budget." needs these tables = [department], so we don't need JOIN.
Plus, it requires nested queries with (INTERSECT, UNION, EXCEPT, IN, NOT IN), and we need the answer to the questions = ["What is the average budget of the departments"].
So, we don't need JOIN and need nested queries, then the the SQL query can be classified as "NESTED".
Label: "NESTED"'''

    EASY_PROMPT = '''### Here are some reference examples:
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

Q: "How many rooms in each building have a capacity of over 50?"
schema_links: [classroom.*,classroom.building,classroom.capacity,50]
SQL: SELECT count(*) ,  building FROM classroom WHERE capacity  >  50 GROUP BY building

Q: "Find the names of the top 3 departments that provide the largest amount of courses?"
schema_links: [course.dept_name,course.*]
SQL: SELECT dept_name FROM course GROUP BY dept_name ORDER BY count(*) DESC LIMIT 3

Q: "Find the maximum and average capacity among rooms in each building."
schema_links: [classroom.building,classroom.capacity]
SQL: SELECT max(capacity) ,  avg(capacity) ,  building FROM classroom GROUP BY building

Q: "Find the title of the course that is offered by more than one department."
schema_links: [course.title]
SQL: SELECT title FROM course GROUP BY title HAVING count(*)  >  1

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

Q: "Find the name of students who took any class in the years of 2009 and 2010."
Schema_links: [student.name,student.id = takes.id,takes.YEAR,2009,2010]
A: Let's think step by step. For creating the SQL for the given question, we need to join these tables = [student,takes]. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select  distinct student.name from student  where  takes.year = 2009  or  takes.year = 2010
SQL: SELECT DISTINCT T1.name FROM student AS T1 JOIN takes AS T2 ON T1.id  =  T2.id WHERE T2.YEAR  =  2009 OR T2.YEAR  =  2010

Q: "list in alphabetic order all course names and their instructors' names in year 2008."
Schema_links: [course.title,course.course_id = teaches.course_id,teaches.id = instructor.id,instructor.name,teaches.year,2008]
A: Let's think step by step. For creating the SQL for the given question, we need to join these tables = [course,teaches,instructor]. First, create an intermediate representation, then use it to construct the SQL query.
Intermediate_representation: select course.title , instructor.name from course  where  teaches.year = 2008  order by course.title asc
SQL: SELECT T1.title ,  T3.name FROM course AS T1 JOIN teaches AS T2 ON T1.course_id  =  T2.course_id JOIN instructor AS T3 ON T2.id  =  T3.id WHERE T2.YEAR  =  2008 ORDER BY T1.title

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

# [Question]: "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010."
# [Schema links]: [teaches.id,teaches.semester,teaches.YEAR,Fall,2009,Spring,2010]
# [Analysis]: Let's think step by step. "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010." can be solved by knowing the answer to the following sub-question "Find the id of instructors who taught a class in Spring 2010".
The SQL query for the sub-question "Find the id of instructors who taught a class in Spring 2010" is SELECT id FROM teaches WHERE semester  =  'Spring' AND YEAR  =  2010
So, the answer to the question "Find the id of instructors who taught a class in Fall 2009 but not in Spring 2010." is = 
Intermediate_representation: select teaches.ID from teaches  where  teaches.semester = "Fall"  and  teaches.year = 2009  and  teaches.semester != "Spring"  and  teaches.year = 2010 
# [Sql]: SELECT id FROM teaches WHERE semester  =  'Fall' AND YEAR  =  2009 EXCEPT SELECT id FROM teaches WHERE semester  =  'Spring' AND YEAR  =  2010

# [Question]: "Find the name of the courses that do not have any prerequisite?"
# [Schema links]: [course.title,course.course_id]
# [Analysis]: Let's think step by step. "Find the name of the courses that do not have any prerequisite?" can be solved by knowing the answer to the following sub-question "What are the courses that have any prerequisite?".
The SQL query for the sub-question "What are the courses that have any prerequisite?" is SELECT course_id FROM prereq
So, the answer to the question "Find the name of the courses that do not have any prerequisite?" is =
Intermediate_representation: select course.title from course  where  @.@ not in prereq.course_id 
# [Sql]: SELECT title FROM course WHERE course_id NOT IN (SELECT course_id FROM prereq)

# [Question]: "Find the salaries of all distinct instructors that are less than the largest salary."
# [Schema links]: [instructor.salary]
# [Analysis]: Let's think step by step. "Find the salaries of all distinct instructors that are less than the largest salary." can be solved by knowing the answer to the following sub-question "What is the largest salary of instructors".
The SQL query for the sub-question "What is the largest salary of instructors" is SELECT max(salary) FROM instructor
So, the answer to the question "Find the salaries of all distinct instructors that are less than the largest salary." is =
Intermediate_representation: select  distinct instructor.salary from instructor  where  @.@ < max ( instructor.salary )
# [Sql]: SELECT DISTINCT salary FROM instructor WHERE salary  <  (SELECT max(salary) FROM instructor)

# [Question]: "Find the names of students who have taken any course in the fall semester of year 2003."
# [Schema links]: [student.id,student.name,takes.id,takes.semester,fall,2003]
# [Analysis]: Let's think step by step. "Find the names of students who have taken any course in the fall semester of year 2003." can be solved by knowing the answer to the following sub-question "Find the students who have taken any course in the fall semester of year 2003.".
The SQL query for the sub-question "Find the students who have taken any course in the fall semester of year 2003." is SELECT id FROM takes WHERE semester  =  'Fall' AND YEAR  =  2003
So, the answer to the question "Find the names of students who have taken any course in the fall semester of year 2003." is =
Intermediate_representation: select student.name from student  where  takes.semester = "Fall"  and  takes.year = 2003
# [Sql]: SELECT name FROM student WHERE id IN (SELECT id FROM takes WHERE semester  =  'Fall' AND YEAR  =  2003)

# [Question]: "Find the minimum salary for the departments whose average salary is above the average payment of all instructors."
# [Schema links]: [instructor.salary,instructor.dept_name]
# [Analysis]: Let's think step by step. "Find the minimum salary for the departments whose average salary is above the average payment of all instructors." can be solved by knowing the answer to the following sub-question "What is the average payment of all instructors.".
The SQL query for the sub-question "What is the average payment of all instructors." is SELECT avg(salary) FROM instructor
So, the answer to the question "Find the minimum salary for the departments whose average salary is above the average payment of all instructors." is =
Intermediate_representation: select min(instructor.salary) , instructor.dept_name from instructor  where  avg ( instructor.salary )  > avg ( instructor.salary )   group by instructor.dept_name
# [Sql]: SELECT min(salary) ,  dept_name FROM instructor GROUP BY dept_name HAVING avg(salary)  >  (SELECT avg(salary) FROM instructor)

# [Question]: "What is the course title of the prerequisite of course Mobile Computing?"
# [Schema links]: [course.title,course.course_id = prereq.course_id,prereq.prereq_id,course.title,Mobile Computing]
# [Analysis]: Let's think step by step. "What is the course title of the prerequisite of course Mobile Computing?" can be solved by knowing the answer to the following sub-question "What are the ids of the prerequisite of course Mobile Computing?".
The SQL query for the sub-question "What are the ids of the prerequisite of course Mobile Computing?" is SSELECT T1.prereq_id FROM prereq AS T1 JOIN course AS T2 ON T1.course_id  =  T2.course_id WHERE T2.title  =  'Mobile Computing'
So, the answer to the question "What is the course title of the prerequisite of course Mobile Computing?" is =
Intermediate_representation: select course.title from course  where  @.@ in prereq.*  and  course.title = "Mobile Computing"
# [Sql]: SELECT title FROM course WHERE course_id IN (SELECT T1.prereq_id FROM prereq AS T1 JOIN course AS T2 ON T1.course_id  =  T2.course_id WHERE T2.title  =  'Mobile Computing')

# [Question]: "Give the title and credits for the course that is taught in the classroom with the greatest capacity."
# [Schema links]: [classroom.capacity,classroom.building = SECTION.building,classroom.room_number = SECTION.room_number,course.title,course.credits,course.course_id = SECTION.course_id]
# [Analysis]: Let's think step by step. "Give the title and credits for the course that is taught in the classroom with the greatest capacity." can be solved by knowing the answer to the following sub-question "What is the capacity of the largest room?".
The SQL query for the sub-question "What is the capacity of the largest room?" is (SELECT max(capacity) FROM classroom)
So, the answer to the question "Give the title and credits for the course that is taught in the classroom with the greatest capacity." is =
Intermediate_representation: select course.title , course.credits from classroom  order by classroom.capacity desc limit 1"
# [Sql]: SELECT T3.title ,  T3.credits FROM classroom AS T1 JOIN SECTION AS T2 ON T1.building  =  T2.building AND T1.room_number  =  T2.room_number JOIN course AS T3 ON T2.course_id  =  T3.course_id WHERE T1.capacity  =  (SELECT max(capacity) FROM classroom)

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
        return instruction + self.SCHEMA_LINKING_PROMPT + "\n" + schema + '\nQ: "' + question + '"\nA: Let\'s think step by step.'

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
        return instruction + schema + '#### Question: ' + question + '\n#### Original SQLite SQL QUERY\n' + sql + '\n#### Output only the fixed SQL query, without any explanation or extra text:'

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
        return None

    def act(
            self,
            item,
            schema: Union[str, PathLike, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ) -> str:
        """Process a single item and generate SQL."""
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
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
        if data_logger:
            data_logger.info(f"{self.NAME}.classification | predicted_class={predicted_class}")

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

        # Step 4: SQL debugging
        logger.debug("Starting SQL debugging...")
        try:
            debug_prompt = self.debug_prompt_maker(question, schema, sql)
            sql = self.llm_debug(debug_prompt)
            logger.debug("SQL debugging completed")
        except Exception as e:
            logger.error(f"SQL debugging failed: {e}")
            # Keep original SQL if debugging fails

        # SQL post-process
        # if self.sql_post_process_function:
        #     sql = self.sql_post_process_function(sql, self.dataset)

        if data_logger:
            data_logger.info(f"{self.NAME}.final_sql | sql={sql}")

        sql = self.save_output(sql, item, row.get("instance_id"))

        logger.info(f"DIN_SQLGenerator sample {item} processed")
        if data_logger:
            data_logger.info(f"{self.NAME}.act end | item={item}")
        return sql
