def template_agent_prompt():
    return """You are the {agent_name} agent, in a team of agents trying to collaboratively transform a natural language question into a SQL query.
Your task is to {task}.

You have access to the following tools. You need to call them in the correct order to complete your task.

<tools>
{tools}
</tools>

You will be given a system state, which contains the current state of the workflow.

Please identify the next tool you need to call in enclosed in <tool_call> tags.

You will then be given the result of the tool call, and the new system state.

Continue to identify the next tool you need to call until you have completed your task. When you are done, just return "DONE".

"""


def template_evaluate():
    return """** Instructions: **
Given the following question, database schema, a candidate SQL query response, and unit tests, evaluate whether or not the response passes each unit test.

- In your evaluation, you should consider how the responses align with the a given unit test.
- Provide reasoning before you return your evaluation inside the <Thinking> tags.
- At the end of your evaluation, you must finish with a list of verdicts corresponding to each candidate responses in <Answer> and </Answer> tags.
- You must include a verdict with one of these formatted options: '[Passed]' or '[Failed]'
- Here is an example of the output format:
<Thinking> Your step by step reasoning here. </Thinking>
<Answer>
Candidate Response #1: [Passed]
Candidate Response #2: [Failed]
Candidate Response #3: [Passed]
....
</Answer>
- Each verdict should be on a new line and correspond to the candidate response in the same order as they are provided.
- Here is the question, database schema, candidate responses, and the unit test to evaluate the responses:

** Database Schema: **
{DATABASE_SCHEMA}

** Question: **π
Question: {QUESTION} (Hint: {HINT})

** Candidate SQL Queries: **
{CANDIDATE_RESPONSES}

** Unit Test: **
{UNIT_TEST}

** Output Format: **
<Thinking> Your step by step reasoning here. </Thinking>
<Answer>
Candidate Response #1: [Passed]
Candidate Response #2: [Failed]
Candidate Response #3: [Passed]
...
</Answer>"""


def template_extract_keywords():
    return """Objective: Analyze the given question and hint to identify and extract keywords, keyphrases, and named entities. These elements are crucial for understanding the core components of the inquiry and the guidance provided. This process involves recognizing and isolating significant terms and phrases that could be instrumental in formulating searches or queries related to the posed question.

Instructions:

Read the Question Carefully: Understand the primary focus and specific details of the question. Look for any named entities (such as organizations, locations, etc.), technical terms, and other phrases that encapsulate important aspects of the inquiry.

Analyze the Hint: The hint is designed to direct attention toward certain elements relevant to answering the question. Extract any keywords, phrases, or named entities that could provide further clarity or direction in formulating an answer.

List Keyphrases and Entities: Combine your findings from both the question and the hint into a single Python list. This list should contain:

Keywords: Single words that capture essential aspects of the question or hint.
Keyphrases: Short phrases or named entities that represent specific concepts, locations, organizations, or other significant details.
Ensure to maintain the original phrasing or terminology used in the question and hint.

Example 1:
Question: "What is the annual revenue of Acme Corp in the United States for 2022?"
Hint: "Focus on financial reports and U.S. market performance for the fiscal year 2022."

["annual revenue", "Acme Corp", "United States", "2022", "financial reports", "U.S. market performance", "fiscal year"]

Example 2:
Question: "In the Winter and Summer Olympics of 1988, which game has the most number of competitors? Find the difference of the number of competitors between the two games."
Hint: "the most number of competitors refer to MAX(COUNT(person_id)); SUBTRACT(COUNT(person_id where games_name = '1988 Summer'), COUNT(person_id where games_name = '1988 Winter'));"

["Winter Olympics", "Summer Olympics", "1988", "1988 Summer", "Summer", "1988 Winter", "Winter", "number of competitors", "difference", "MAX(COUNT(person_id))", "games_name", "person_id"]

Example 3:
Question: "How many Men's 200 Metres Freestyle events did Ian James Thorpe compete in?"
Hint: "Men's 200 Metres Freestyle events refer to event_name = 'Swimming Men''s 200 metres Freestyle'; events compete in refers to event_id;"

["Swimming Men's 200 metres Freestyle", "Ian James Thorpe", "Ian", "James", "Thorpe", "compete in", "event_name", "event_id"]

Task:
Given the following question and hint, identify and list all relevant keywords, keyphrases, and named entities.

Question: {QUESTION}
Hint: {HINT}

Please provide your findings as a Python list, capturing the essence of both the question and hint through the identified terms and phrases. 
Only output the Python list, no explanations needed. """

def template_filter_column():
    return """You are a detail-oriented data scientist tasked with evaluating the relevance of database column information for answering specific SQL query question based on provided hint.

Your goal is to assess whether the given column details are pertinent to constructing an SQL query to address the question informed by the hint. Label the column information as "relevant" if it aids in query formulation, or "irrelevant" if it does not.

Procedure:
1. Carefully examine the provided column details.
2. Understand the question about the database and its associated hint.
3. Decide if the column details are necessary for the SQL query based on your analysis.

Here are some examples of how to determine if the column information is relevant or irrelevant to the question and the hint:

Example 1:
Column information:
Table name: `movies`
Original column name: `movie_title`
Data type: TEXT
Description: Name of the movie
Example of values in the column: `La Antena`


Question:
Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity.

HINT:
released in the year 1945 refers to movie_release_year = 1945;

```json
{{
  "chain_of_thought_reasoning": "The question specifically asks for movie titles from a particular year and to sort them by popularity. The column movie_title directly provides the names of movies, which is exactly what is required to list the movie titles as requested in the question.",
  "is_column_information_relevant": "Yes"
}}
```

Example 2:
Column information:
Table name: `movies`
Original column name: `movie_release_year`
Data type: INTEGER
Description: Release year of the movie
Example of values in the column: `2007`


Question:
List all movie title rated in April 2020 from user who was a trialist.

HINT:
movie title rated in April 2020 refers to rating_timestamp_utc LIKE '%2020-04-%'; user is a trial list refers to user_trialist = 1;

```json
{{
  "chain_of_thought_reasoning": "The question and hint focus on movies rated in a specific month and year and by a specific type of user (trialist), neither of which relates to the movie_release_year column. This column only provides the year movies were released, which is not what is being queried.",
  "is_column_information_relevant": "No"
}}
```

Example 3:
Column information:
Table name: `ratings_users`
Original column name: `user_has_payment_method`
Data type: INTEGER
Description: whether the user was a paying subscriber when he rated the movie
Value description: 1 = the user was a paying subscriber when he rated the movie  0 = the user was not a paying subscriber when he rated
Example of values in the column: `0`


Question:
How many users, who were a paying subscriber when they rated the movie, gave the movie that was released in 1924 and directed by Erich von Stroheim a rating score of 5?

HINT:
Directed by Buster Keaton refers to director_name; released in 1924 refers to movie_release_year = 1924; paying subscriber refers to user_has_payment_method = 1; rating score of 5 refers to rating_score = 5;

```json
{{
  "chain_of_thought_reasoning": "The question asks about users who were paying subscribers and rated a specific movie from 1924 directed by a specific director. The user_has_payment_method column indicates whether a user was a paying subscriber at the time of rating, which is directly relevant to the question and the hint focusing on subscribers.",
  "is_column_information_relevant": "Yes"
}}
```

Example 4:
Column information:
Table name: `movies`
Original column name: `director_name`
Data type: TEXT
Description: Full Name of the movie director
Example of values in the column: `Stanley Kubrick`


Question:
What is the average number of Mubi users who love movies directed by Stanley Kubrick?

HINT:
average = AVG(movie_popularity); number of Mubi users who loves the movie refers to movie_popularity;

```json
{{
  "chain_of_thought_reasoning": "The question requires filtering movies directed by `Stanley Kubrick` to calculate the average popularity. The director_name column provides the director's name, and as shown in the example values, it includes `Stanley Kubrick`, which is essential for filtering movies directed by this specific director.",
  "is_column_information_relevant": "Yes"
}}
```

Example 5:
Column information:
Table name: `movies`
Original column name: `movie_title`
Data type: TEXT
Description: Name of the movie
Example of values in the column: `La Antena`


Question:
How many movies directed by Francis Ford Coppola have a popularity of more than 1,000? Indicate what is the highest amount of likes that each critic per movie has received, if there's any.

HINT:
Francis Ford Coppola refers to director_name; popularity of more than 1,000 refers to movie_popularity >1000;highest amount of likes that each critic per movie has received refers to MAX(critic_likes)
  
```json  
{{
  "chain_of_thought_reasoning": "The question involves counting movies directed by a specific director with a high popularity score. The movie_title column is relevant because it allows for the identification of movie titles, which is necessary for aggregating and analyzing data on specific movies as mentioned in the hint.",
  "is_column_information_relevant": "Yes"
}}
```

Example 6:
Column information:
Table name: `lists_users`
Original column name: `list_creation_date_utc`
Data type: TEXT
Description: Creation date for the list
Value description: YYYY-MM-DD
Example of values in the column: `2009-12-18`


Question:
Provide list titles created by user who are eligible for trial when he created the list.

HINT:
eligible for trial refers to user_eligible_for_trial = 1

```json
{{
  "chain_of_thought_reasoning": "The question asks for list titles created by users eligible for a trial. The list_creation_date_utc column, which provides the creation dates of lists, is irrelevant because the hint and the question are concerned with the trial status of the users, not the dates the lists were created.",
  "is_column_information_relevant": "No"
}}
```

Example 7:
Column information:
Table name: `playstore`
Original column name: `Installs`
Data type: TEXT
Description: Number of user downloads/installs for the app (as when scraped)
Value description: 1,000,000+ 15% 10,000,000+ 12% Other (8010) 74%
Example of values in the column: `10,000+`


Question:
Name the Apps with a sentiment objectivity of 0.3 and include their number of installs.

HINT:
FALSE;

```json
{{
  "chain_of_thought_reasoning": "The question asks for apps with a specific sentiment objectivity and their number of installs. The Installs column is relevant because it provides data on how many times each app has been installed, which is crucial for answering the question as per the hint.",
  "is_column_information_relevant": "Yes"
}}
```

Example 8:
Column information:
Table name: `movies`
Original column name: `movie_title`
Data type: TEXT
Description: Name of the movie
Example of values in the column: `La Antena`


Question:
What is Jeannot Szwarc's most popular movie and what is its average rating score?

HINT:
Jeannot Szwarc's refers to director_name = 'Jeannot Szwarc'; most popular movie refers to MAX(movie_popularity); average rating score refers to avg(rating_score)

```json
{{
  "chain_of_thought_reasoning": "The question seeks the most popular movie by a specific director and its average rating score. The movie_title column is relevant because it provides the names of movies, which are essential for identifying the most popular movie directed by Jeannot Szwarc.",
  "is_column_information_relevant": "Yes"
}}
```

Example 9:
Column information:
Table name: `ratings`
Original column name: `user_subscriber`
Data type: INTEGER
Example of values in the column: `0`


Question:
What is the percentage of the ratings were rated by user who was a subcriber?

HINT:
user is a subscriber refers to user_subscriber = 1; percentage of ratings = DIVIDE(SUM(user_subscriber = 1), SUM(rating_score)) as percent;

```json
{{
  "chain_of_thought_reasoning": "The question asks about the percentage of ratings from subscribers. The user_subscriber column, indicating whether a user is a subscriber (1) or not (0), is directly relevant as it enables filtering the necessary data to calculate the percentages mentioned in the hint.",
  "is_column_information_relevant": "Yes"
}}
```

Example 10:
Column information:
Table name: `lists`
Original column name: `list_followers`
Data type: INTEGER
Description: Number of followers on the list
Example of values in the column: `5`


Question:
How many users who created a list in the February of 2016 were eligible for trial when they created the list? Indicate the user id of the user who has the most number of followers in his list in February of 2016.

HINT:
created a list in the February of 2016 refer to list_creation_date_utc BETWEEN 2/1/2016 and 2/29/2016; eligible for trial refers to user_eligible_for_trial = 1;

```json
{{
  "chain_of_thought_reasoning": "The question involves finding users who created a list in a specific month and year, with additional focus on those who had the most followers. The list_followers column directly applies because it provides the exact data needed to identify which user's list had the most followers during the specified time.",
  "is_column_information_relevant": "Yes"
}}
```

Example 11:
Column information:
Table name: `user_reviews`
Original column name: `Sentiment_Subjectivity`
Expanded column name: `Sentiment Subjectivity`
Data type: TEXT
Description: Sentiment subjectivity score
Value description: commonsense evidence: more subjectivity refers to less objectivity, vice versa.
Example of values in the column: `0.53`


Question:
What is the average rating of comic category apps? How many users hold positive attitude towards this app?

HINT:
average rating = AVG(Rating where Category = 'COMICS'); number of users who hold a positive attitude towards the app refers to SUM(Sentiment = 'Positive');

```json
{{
  "chain_of_thought_reasoning": "The question involves the average rating and user attitudes towards apps in a specific category. The Sentiment_Subjectivity column, while related to sentiment, does not provide information on user attitudes or ratings, making it irrelevant to the question and hint.",
  "is_column_information_relevant": "No"
}}

Example 12:
Column information:
Table name: `movies`
Original column name: `movie_title`
Data type: TEXT
Description: Name of the movie
Example of values in the column: `La Antena`


Question:
List ther users who gave the worst rating for movie 'Love Will Tear Us Apart'.

HINT:
worst rating refers to rating_score = 1;

```json
{{
  "chain_of_thought_reasoning": "The question is looking for users who rated a specific movie with the worst score. The movie_title column provides the necessary data to identify the movie by its title, which directly aligns with the hint that refers to the movie title 'Love Will Tear Us Apart'.",
  "is_column_information_relevant": "Yes"
}}
```

Example 13:
Column information:
Table name: `ratings`
Original column name: `rating_score`
Data type: INTEGER
Description: Rating score ranging from 1 (lowest) to 5 (highest)
Value description: commonsense evidence: The score is proportional to the user's liking. The higher the score is, the more the user likes the movie
Example of values in the column: `3`


Question:
What is the URL to the user profile image on Mubi of the user who gave the movie id of 1103 a 5 ratinng score on 4/19/2020?

HINT:
URL to the user profile image on Mubi  refers to user_avatar_image_url;  4/19/2020 refers to rating_date_utc

```json
{{
  "chain_of_thought_reasoning": "The question seeks the URL for the user profile of someone who rated a specific movie highly on a particular date. The rating_score column, indicating the score given to movies, is relevant because it allows filtering for ratings of 5, directly addressing the hint's requirement for identifying high ratings.
",
  "is_column_information_relevant": "Yes"
}}
```

Example 14:
Column information:
Table name: `movies`
Original column name: `movie_release_year`
Data type: INTEGER
Description: Release year of the movie
Example of values in the column: `2007`


Question:
When was the first movie of the director who directed the highest number of movies released and what is the user id of the user who received the highest number of comments related to the critic made by the user rating the movie?

HINT:
comments refer to critic_comments

```json
{{
  "chain_of_thought_reasoning": "The question asks for the release year of the first movie by the director who has directed the most films. The movie_release_year column directly provides this necessary information, as it lists the release years of movies. This column is essential to determine when that first movie was released, making it relevant to the question despite the hint focusing solely on comments related to critic ratings.",
  "is_column_information_relevant": "Yes"
}}
```

Example 15:
Column information:
Table name: `playstore`
Original column name: `Price`
Data type: TEXT
Description: Price of the app (as when scraped)
Value description: 0 93% $0.99 1% Other (653) 6%  commonsense evidence: Free means the price is 0.
Example of values in the column: `0`


Question:
Which of the app is the best selling app and what is the sentiments polarity of it?

HINT:
best selling app = MAX(MULTIPLY(Price, Installs));

```json
{{
  "chain_of_thought_reasoning": "The question seeks to identify the best-selling app and its sentiments polarity, with the hint specifying the calculation for "best selling" as the maximum product of Price and Installs. The Price column is crucial for this computation as it provides the price at which each app is sold, which, when multiplied by the number of installs, helps determine the app's total revenue. This makes the Price column directly relevant to identifying the best-selling app according to the hint's criteria.",
  "is_column_information_relevant": "Yes"
}}

Example 16:
Column information:
Table name: `person`
Original column name: `full_name`
Expanded column name: `full name`
Data type: TEXT
Description: the full name of the person
Value description: commonsense evidence: A person's full name is the complete name that they are known by, which typically includes their first name, middle name (if applicable), and last name. 
Example of values in the column: `Dagfinn Sverre Aarskog`


Question:
Tell the weight of Dagfinn Sverre Aarskog.

HINT:


```json
{{
  "chain_of_thought_reasoning": "The question explicitly asks for the weight of a person named `Dagfinn Sverre Aarskog`. As shown in the column information, `Dagfinn Sverre Aarskog` is one of the example values in the `full_name` column, therefore this column can be used to identify the person and retrieve their weight.", 
  "is_column_information_relevant": "Yes"
}}
```

Now, its your turn to determine whether the provided column information can help formulate a SQL query to answer the given question, based on the provided hint.

The following guidelines are VERY IMPORTANT to follow. Make sure to check each of them carefully before making your decision:
1. You're given only one column's information, which alone isn't enough to answer the full query. Concentrate solely on this provided data and assess its relevance to the question and hint without considering any missing information.
2. Read the column information carefully and understand the description of it, then see if the question or the hint is asking or referring to the same information. If yes then the column information is relevant, otherwise it is irrelevant.
3. Look beyond mere keywords. Assess whether there is a meaningful, semantic connection between the column information and the needs of the question or hint. Mere word matches do not necessarily imply relevance.
4. If the question refers to applying a logic on a data such as average, sum, max, min, or any other operation, and the column information is a part of that logic, then the column information is relevant.
5. Pay attention to the provided `Example of values in the column`. If you see a shared keyword between the example and the question or hint, then the column information is relevant. (VERY IMPORTANT)
6. If you see the column name appeared in the hint, then it is definitely relevant. (VERY IMPORTANT)
7. Note that it does not matter if the question is asking for other information not contained in the column, as long as this column's information is useful for crafting a SQL query answering the question, you should consider this column as relevant.

Column information:
{COLUMN_PROFILE}

Question:
{QUESTION}

HINT:
{HINT}


Take a deep breath and provide your answer in the following json format:

```json
{{
  "chain_of_thought_reasoning": "One line explanation of why or why not the column information is relevant to the question and the hint.",
  "is_column_information_relevant": "Yes" or "No"
}}
```

Only output a json as your response."""

def template_generate_candidate_finetuned():
    return """You are an experienced database expert.
Now you need to generate a SQL query given the database information, a question and some additional information.
The database structure is defined by the following table schemas (comments after '--' provide additional column descriptions).
Note that the "Example Values" are actual values from the column. Some column might contain the values that are directly related to the question. Use it to help you justify which columns to use.

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

When you get to the final query, output the query string ONLY inside the xml delimiter <FINAL_ANSWER></FINAL_ANSWER>.

Now is the real question, following the instruction and examples, generate the SQL with Recursive Divide-and-Conquer approach. Make sure you only output one single query.
**************************
【Table creation statements】
{DATABASE_SCHEMA}

**************************
【Question】
Question: 
{QUESTION}

Evidence:
{HINT}

**************************
【Answer】"""

def template_generate_candidate_one():
    return """You are an experienced database expert.
Now you need to generate a SQL query given the database information, a question and some additional information.
The database structure is defined by the following table schemas (comments after '--' provide additional column descriptions).
Note that the "Example Values" are actual values from the column. Some column might contain the values that are directly related to the question. Use it to help you justify which columns to use.

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "recursive divide-and-conquer approach to SQL query generation from natural language".

Here is a high level description of the steps.
1. **Divide (Decompose Sub-question with Pseudo SQL):** The complex natural language question is recursively broken down into simpler sub-questions. Each sub-question targets a specific piece of information or logic required for the final SQL query. 
2. **Conquer (Real SQL for sub-questions):**  For each sub-question (and the main question initially), a "pseudo-SQL" fragment is formulated. This pseudo-SQL represents the intended SQL logic but might have placeholders for answers to the decomposed sub-questions. 
3. **Combine (Reassemble):** Once all sub-questions are resolved and their corresponding SQL fragments are generated, the process reverses. The SQL fragments are recursively combined by replacing the placeholders in the pseudo-SQL with the actual generated SQL from the lower levels.
4. **Final Output:** This bottom-up assembly culminates in the complete and correct SQL query that answers the original complex question. 

Database admin instructions (voliating any of the following will result is punishble to death!):
1. **SELECT Clause:** 
    - Only select columns mentioned in the user's question. 
    - Avoid unnecessary columns or values.
2. **Aggregation (MAX/MIN):**
    - Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:**
    - Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC` to ensure distinct values.
4. **Handling NULLs:**
    - If a column may contain NULL values (indicated by "None" in value examples or explicitly), use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:**
    - Only include tables essential to answer the question.
6. **Strictly Follow Hints:**
    - Adhere to all provided hints.
7. **Thorough Question Analysis:**
    - Address all conditions mentioned in the question.
8. **DISTINCT Keyword:**
    - Use `SELECT DISTINCT` when the question requires unique values (e.g., IDs, URLs). 
    - Refer to column statistics ("Value Statics") to determine if `DISTINCT` is necessary.
9. **Column Selection:**
    - Carefully analyze column descriptions and hints to choose the correct column when similar columns exist across tables.
10. **String Concatenation:**
    - Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause. 
11. **JOIN Preference:**
    - Prioritize `INNER JOIN` over nested `SELECT` statements.
12. **SQLite Functions Only:**
    - Use only functions available in SQLite.
13. **Date Processing:**
    - Utilize `STRFTIME()` for date manipulation (e.g., `STRFTIME('%Y', SOMETIME)` to extract the year).

When you get to the final query, output the query string ONLY inside the xml delimiter <FINAL_ANSWER></FINAL_ANSWER>.

Here are some examples

======= Example =======
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


**************************
【Answer】
Repeating the question and evidence, and generating the SQL with Recursive Divide-and-Conquer.
**Question**: How many Thai restaurants can be found in San Pablo Ave, Albany? 
**Evidence**: Thai restaurant refers to food_type = 'thai'; San Pablo Ave Albany refers to street_name = 'san pablo ave' AND T1.city = 'albany'

**1. Divide and Conquer:**

* **Main Question:** How many Thai restaurants can be found in San Pablo Ave, Albany? 
    * **Analysis:** The question asks for a count of restaurants, so we'll use `COUNT()` for that. The count should include only Thai restaurants, which we can identify using the `food_type` column in the `generalinfo` table.  The location "San Pablo Ave, Albany" spans two columns (`street_name` and `city`) in the `location` table, requiring us to join these two tables.
    * **Pseudo SQL:** SELECT COUNT(`T1`.`id_restaurant`) FROM `generalinfo` AS `T1` INNER JOIN `location` AS `T2` ON `T1`.`id_restaurant` = `T2`.`id_restaurant` WHERE  <Thai restaurant> AND <in San Pablo Ave, Albany>

    * **Sub-question 1:** Thai restaurant
        * **Analysis:** This is a straightforward filter on the `generalinfo` table using the `food_type` column.
        * **Pseudo SQL:** `T1`.`food_type` = 'thai'

    * **Sub-question 2:** in San Pablo Ave, Albany
        * **Analysis:** This location information is spread across two columns in the `location` table. We need to combine these conditions with an "AND" operator to ensure both are met.
        * **Pseudo SQL:** `T2`.`street_name` = 'san pablo ave' AND `T2`.`city` = 'albany' 

**2. Assembling SQL:**

* **Sub-question 1 (Thai restaurant):**
    * **SQL:** `T1`.`food_type` = 'thai'

* **Sub-question 2 (in San Pablo Ave, Albany):**
    * **SQL:** `T2`.`street_name` = 'san pablo ave' AND `T2`.`city` = 'albany'

* **Main Question (count of restaurants):**
    * **SQL:** SELECT COUNT(`T1`.`id_restaurant`) FROM `generalinfo` AS `T1` INNER JOIN `location` AS `T2` ON `T1`.`id_restaurant` = `T2`.`id_restaurant` WHERE `T1`.`food_type` = 'thai' AND `T2`.`street_name` = 'san pablo ave' AND `T2`.`city` = 'albany'

**3. Simplification and Optimization:**

* The SQL query from step 2 is already quite efficient. We've used `INNER JOIN` to combine the tables based on their relationship, and the `WHERE` clause clearly defines our filtering criteria. There's no need for nested queries or complex sub-selections in this case.

**Final Optimized SQL Query:**

<FINAL_ANSWER>
SELECT COUNT(T1.id_restaurant) FROM generalinfo AS T1 INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant WHERE T1.food_type = 'thai' AND T1.city = 'albany' AND T2.street_name = 'san pablo ave'
</FINAL_ANSWER> 

===========
Example 1
**************************
【Database Info】
CREATE TABLE account (
    account_id INT PRIMARY KEY,
    district_id INT REFERENCES district(district_id),
    frequency VARCHAR(255) NOT NULL,
    date DATE NOT NULL
);
CREATE TABLE client (
    client_id INT PRIMARY KEY,
    gender CHAR(1) NOT NULL,
    birth_date DATE NOT NULL,
    district_id INT REFERENCES district(district_id)
);
CREATE TABLE district (
    district_id INT PRIMARY KEY,
    a4 VARCHAR(255) NOT NULL, -- Assuming A4 and A11 are strings due to examples
    a11 VARCHAR(255) NOT NULL
);
**************************
【Question】
Question: What is the gender of the youngest client who opened account in the lowest average salary branch?
Hint: Given that Later birthdate refers to younger age; A11 refers to average salary

**************************
【Answer】
Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer.
**Question**: What is the gender of the youngest client who opened account in the lowest average salary branch?
**Hint**: Given that Later birthdate refers to younger age; A11 refers to average salary

**1. Divide and Conquer:**

* **Main Question:** What is the gender of the youngest client who opened account in the lowest average salary branch?
    * **Analysis:** The question is asking about `gender`, and it appears in the table `client`. We will use this as the output column, selecting it from the youngest client in the lowest average salary branch.
    * **Pseudo SQL:** SELECT `T1`.`gender` FROM `client` AS `T1` WHERE <youngest client in the lowest average salary branch> 

    * **Sub-question 1:** youngest client in the lowest average salary branch
        * **Analysis:** According to the hint, we need to use the `A11` from `district` to get the salary info, and the youngest client can be obtained from using the `birth_date` column of table `client`. The items between these two tables can be INNER JOIN using district_id.
        * **Pseudo SQL:** SELECT `T1`.`client_id` FROM `client` AS `T1` INNER JOIN `district` AS `T2` ON `T1`.`district_id` = `T2`.`district_id` WHERE <lowest average salary branch> ORDER BY `T1`.`birth_date` DESC NULLS LAST LIMIT 1

        * **Sub-question 1.1:** lowest average salary branch
            * **Analysis:** We can get the lowest average salary branch using order by `A11` ASC and pick top 1. The column `A11` is not NULLABLE, so we do not need to add "IS NOT NULL" filter
            * **Pseudo SQL:**  SELECT `district_id` FROM `district` ORDER BY `A11` ASC LIMIT 1 

**2. Assembling SQL:**

* **Sub-question 1.1 (lowest average salary branch):**
    * **SQL:** SELECT `district_id` FROM `district` ORDER BY `A11` ASC LIMIT 1 

* **Sub-question 1 (youngest client in the lowest average salary branch):**
    * **SQL:** SELECT `T1`.`client_id` FROM `client` AS `T1` INNER JOIN `district` AS `T2` ON `T1`.`district_id` = `T2`.`district_id` WHERE `T2`.`district_id` IN (SELECT `district_id` FROM `district` ORDER BY `A11` ASC LIMIT 1) ORDER BY `T1`.`birth_date` DESC NULLS LAST LIMIT 1

* **Main Question (gender of the client):**
    * **SQL:** SELECT `T1`.`gender` FROM `client` AS `T1` WHERE `T1`.`client_id` = (SELECT `T1`.`client_id` FROM `client` AS `T1` INNER JOIN `district` AS `T2` ON `T1`.`district_id` = `T2`.`district_id` WHERE `T2`.`district_id` IN (SELECT `district_id` FROM `district` ORDER BY `A11` ASC LIMIT 1) ORDER BY `T1`.`birth_date` DESC NULLS LAST LIMIT 1) 

**3. Simplification and Optimization:**

* The final SQL query from step 2 can be simplified and optimized. The nested queries can be combined using a single `INNER JOIN` and the filtering can be done within a single `ORDER BY` clause.

**Final Optimized SQL Query:**

<FINAL_ANSWER>
SELECT `T1`.`gender`
  FROM `client` AS `T1`
  INNER JOIN `district` AS `T2`
  ON `T1`.`district_id` = `T2`.`district_id`
  ORDER BY `T2`.`A11` ASC, `T1`.`birth_date` DESC NULLS LAST
  LIMIT 1
</FINAL_ANSWER>

===========
Example 2 (dividing into two parallel sub-questions)
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

**************************
【Answer】

Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer.
**Question**: From 1900 to 1992, how many games did London host?
**Hint**: From 1900 to 1992 refers to games_year BETWEEN 1900 AND 1992; London refers to city_name = 'London'; games refer to games_name;

**1. Divide and Conquer:**

* **Main Question:** From 1900 to 1992, how many games did London host?
    * **Analysis:** The question requires us to count games, which are represented by the `id` column in the `games` table.  We need to filter these games based on two criteria: they were hosted in London and occurred between 1900 and 1992.
    * **Pseudo SQL:** SELECT COUNT(`T1`.`id`) FROM `games` AS `T1`  WHERE  <games are in London> AND <games year between 1900 and 1992>

    * **Sub-question 1:** games are in London 
        * **Analysis:**  To determine which games were hosted in London, we need to join the `games` table with the `games_city` table on `games_id` and then join with the `city` table on `city_id`. We'll use `INNER JOIN` to ensure only matching records are considered.  The filtering on 'London' will be applied to the `city_name` column.
        * **Pseudo SQL:**  `T1`.`id` IN (SELECT `T1`.`games_id` FROM `games_city` AS `T1` INNER JOIN `city` AS `T2` ON `T1`.`city_id` = `T2`.`id` WHERE `T2`.`city_name` = 'London')

    * **Sub-question 2:** games year between 1900 and 1992
        * **Analysis:** This involves filtering the `games` table directly based on the `games_year` column using the `BETWEEN` operator.
        * **Pseudo SQL:** `T1`.`games_year` BETWEEN 1900 AND 1992

**2. Assembling SQL:**

* **Sub-question 1 (games are in London):**
    * **SQL:**  `T1`.`id` IN (SELECT `T1`.`games_id` FROM `games_city` AS `T1` INNER JOIN `city` AS `T2` ON `T1`.`city_id` = `T2`.`id` WHERE `T2`.`city_name` = 'London')

* **Sub-question 2 (games year between 1900 and 1992):**
    * **SQL:**  `T1`.`games_year` BETWEEN 1900 AND 1992

* **Main Question (count of games):**
    * **SQL:** SELECT COUNT(`T1`.`id`) FROM `games` AS `T1` WHERE `T1`.`id` IN (SELECT `T1`.`games_id` FROM `games_city` AS `T1` INNER JOIN `city` AS `T2` ON `T1`.`city_id` = `T2`.`id` WHERE `T2`.`city_name` = 'London') AND `T1`.`games_year` BETWEEN 1900 AND 1992

**3. Simplification and Optimization:**

* The nested query can be converted into a more efficient `JOIN` operation. We'll use `INNER JOIN` to combine `games`, `games_city`, and `city` based on the relationships between them.

**Final Optimized SQL Query:**

<FINAL_ANSWER>
SELECT COUNT(T3.id) FROM games_city AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.id INNER JOIN games AS T3 ON T1.games_id = T3.id WHERE T2.city_name = 'London' AND T3.games_year BETWEEN 1900 AND 1992
</FINAL_ANSWER> 

===========

Example 3 (When it's not clear which column should be used for a string matching, use a loosen condition such as string LIKE and OR condition to cover multiple possible columns.)
**************************
【Table creation statements】
CREATE TABLE `student_programs` (
    `Program Type` text, -- Program Type Example values: ['Summer School', 'After School Program', 'Special Education']
    `Participants (Ages 10-15)` double precision, -- Participants (Ages 10-15) Example values: ['1250.0', '500.0', '75.0']
    `Total Enrollment (Ages 10-15)` double precision, -- Total Enrollment (Ages 10-15) Example values: ['500.0', '1800.0', '1000.0']
    `School Category` text, --  Example values: ['Charter Schools', 'Private Schools', 'Magnet Schools']
);
**************************
【Question】
Question: Please list the lowest three participation rates for students aged 10-15 in online programs. 
Hint: Participation rate for students aged 10-15 = `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`
**************************
【Answer】
Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer.
**Question:** Please list the lowest three participation rates for students aged 10-15 in online programs. 
**Hint:** Participation rate for students aged 10-15 = `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`

**1. Divide and Conquer:**

* **Main Question:** Please list the lowest three participation rates for students aged 10-15 in online programs.
    * **Analysis:** The question is asking about the ratio between `Participants (Ages 10-15)` and `Total Enrollment (Ages 10-15)`. We need to filter the data to only include online programs.
    * **Pseudo SQL:** SELECT (`Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`) FROM `student_programs` WHERE <online programs> ORDER BY (`Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`) ASC NULLS LAST LIMIT 3

    * **Sub-question 1:** online programs 
        * **Analysis:** We will get the information from the table `student_programs`.
        * **Pseudo SQL:** SELECT program_id FROM `student_programs` WHERE <condition for online programs>

        * **Sub-question 1.1:** condition for online programs (Note: This requires external knowledge or database schema information. We need to identify which column(s) indicate "online programs".)
            * **Analysis:** We'll assume either "School Category" or "Program Type" columns might contain the term "online."
            * **Pseudo SQL:**  LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%'

**2. Assembling SQL:**

* **Sub-question 1.1 (condition for online programs):**
    * **SQL:** LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%' 

* **Sub-question 1 (online programs):**
    * **SQL:** SELECT program_id FROM `student_programs` WHERE LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%'

* **Main Question (lowest three participation rates):**
    * **SQL:** SELECT (`Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`) FROM `student_programs` WHERE program_id IN (SELECT program_id FROM `student_programs` WHERE LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%') ORDER BY (`Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)`) ASC NULLS LAST LIMIT 3

**3. Simplification and Optimization:**

* We can directly incorporate the condition for online programs into the main query. 

**Final Optimized SQL Query:**
<FINAL_ANSWER>
SELECT `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)` FROM `student_programs` 
  WHERE LOWER(`School Category`) LIKE '%online%' OR LOWER(`Program Type`) LIKE '%online%'
  AND `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)` IS NOT NULL 
  ORDER BY `Participants (Ages 10-15)` / `Total Enrollment (Ages 10-15)` ASC NULLS LAST LIMIT 3;
</FINAL_ANSWER>

=============

Example 4
**************************
【Table creation statements】
CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    department_id INT,
    salary INT 
);
**************************
【Question】
Question: How many employees earn over $100,000?

【Answer】
Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer.
**Question:** How many employees earn over $100,000?

**1. Divide and Conquer:**

* **Main Question:** How many employees earn over $100,000?

    * **Pseudo SQL:** SELECT COUNT(*) FROM employees WHERE <employees earning over 100000>
    * **Analysis:** The question is asking about the COUNT of employees. We need to filter the data to only include employees earning over $100,000.

    * **Sub-question 1:** employees earning over 100000
        * **Analysis:** Simple condition on the `salary` column.
        * **Pseudo SQL:** SELECT employee_id FROM employees WHERE salary > 100000

**2. Assembling SQL:**

* **Sub-question 1 (employees earning over 100000):** 
    * **SQL:** SELECT employee_id FROM employees WHERE salary > 100000

* **Main Question (count of employees):**
    * **SQL:** SELECT COUNT(*) FROM employees WHERE employee_id IN (SELECT employee_id FROM employees WHERE salary > 100000)

**3. Simplification and Optimization:**

* We can achieve the same result more efficiently within a single WHERE clause.

**Final Optimized SQL Query:**

<FINAL_ANSWER>
SELECT COUNT(*) FROM employees WHERE salary > 100000;
</FINAL_ANSWER>

====== Example 5 =======
**************************
【Table creation statements】
CREATE TABLE Airlines
(
	FL_DATE TEXT, -- examples: `2018/8/9`| `flight date` description: flight date
	ORIGIN TEXT, -- examples: `SAN`| description: airport of origin
	DEST TEXT, -- examples: `SAN`| `destination` description: Destination airport
	FOREIGN KEY (ORIGIN) REFERENCES Airports(Code),
	FOREIGN KEY (DEST) REFERENCES Airports(Code),
);

CREATE TABLE Airports
(
	Code TEXT primary key,
	Description TEXT,
);


**************************
【Question】
Question: 
How many flights were there from San Diego International airport to Los Angeles International airport in the August of 2018? 

Evidence:
flights from refers to ORIGIN; San Diego International airport refers to Description = 'San Diego, CA: San Diego International'; flights to refers to DEST; Los Angeles International airport refers to Description = 'Los Angeles, CA: Los Angeles International'; in the August of 2018 refers to FL_DATE like '2018/8%';


**************************
【Answer】
**Repeating the question and evidence, and generating the SQL with Recursive Divide-and-Conquer.**

**Question**: How many flights were there from San Diego International airport to Los Angeles International airport in the August of 2018? 
**Evidence**: flights from refers to ORIGIN; San Diego International airport refers to Description = 'San Diego, CA: San Diego International'; flights to refers to DEST; Los Angeles International airport refers to Description = 'Los Angeles, CA: Los Angeles International'; in the August of 2018 refers to FL_DATE like '2018/8%';

**1. Divide and Conquer:**

* **Main Question:** How many flights were there from San Diego International airport to Los Angeles International airport in the August of 2018? 
    * **Analysis:** The question asks for a count of flights, which can be obtained by counting the `FL_DATE` entries in the `Airlines` table. We need to apply three filters: flights originating from San Diego International, flights destined for Los Angeles International, and flights occurring in August 2018.
    * **Pseudo SQL:** SELECT COUNT(`FL_DATE`) FROM `Airlines` WHERE <flights are in August 2018> AND <flights are from San Diego International> AND <flights are to Los Angeles International>

    * **Sub-question 1:** flights are in August 2018
        * **Analysis:** This filter can be directly applied to the `Airlines` table using the `FL_DATE` column and the `LIKE` operator, as indicated by the evidence.
        * **Pseudo SQL:** `FL_DATE` LIKE '2018/8%'

    * **Sub-question 2:** flights are from San Diego International
        * **Analysis:**  We need to find the airport code (`ORIGIN`) corresponding to 'San Diego, CA: San Diego International' from the `Airports` table and use it to filter the `Airlines` table. This requires joining `Airports` and `Airlines` based on `Airports`.`Code` = `Airlines`.`ORIGIN`.
        * **Pseudo SQL:** `ORIGIN` = (SELECT `T2`.`ORIGIN` FROM `Airports` AS `T1` INNER JOIN `Airlines` AS `T2` ON `T1`.`Code` = `T2`.`ORIGIN` WHERE `T1`.`Description` = 'San Diego, CA: San Diego International')

    * **Sub-question 3:** flights are to Los Angeles International
        * **Analysis:** Similar to sub-question 2, we need to find the airport code (`DEST`) for 'Los Angeles, CA: Los Angeles International' from the `Airports` table and use it to filter the `Airlines` table. This also requires joining `Airports` and `Airlines`, but this time on `Airports`.`Code` = `Airlines`.`DEST`.
        * **Pseudo SQL:** `DEST` = (SELECT `T4`.`DEST` FROM `Airports` AS `T3` INNER JOIN `Airlines` AS `T4` ON `T3`.`Code` = `T4`.`DEST` WHERE `T3`.`Description` = 'Los Angeles, CA: Los Angeles International')

**2. Assembling SQL:**

* **Sub-question 1 (flights are in August 2018):**
    * **SQL:** `FL_DATE` LIKE '2018/8%'

* **Sub-question 2 (flights are from San Diego International):**
    * **SQL:** `ORIGIN` = (SELECT `T2`.`ORIGIN` FROM `Airports` AS `T1` INNER JOIN `Airlines` AS `T2` ON `T1`.`Code` = `T2`.`ORIGIN` WHERE `T1`.`Description` = 'San Diego, CA: San Diego International')

* **Sub-question 3 (flights are to Los Angeles International):**
    * **SQL:** `DEST` = (SELECT `T4`.`DEST` FROM `Airports` AS `T3` INNER JOIN `Airlines` AS `T4` ON `T3`.`Code` = `T4`.`DEST` WHERE `T3`.`Description` = 'Los Angeles, CA: Los Angeles International')

* **Main Question (count of flights):**
    * **SQL:** SELECT COUNT(`FL_DATE`) FROM `Airlines` WHERE `FL_DATE` LIKE '2018/8%' AND `ORIGIN` = (SELECT `T2`.`ORIGIN` FROM `Airports` AS `T1` INNER JOIN `Airlines` AS `T2` ON `T1`.`Code` = `T2`.`ORIGIN` WHERE `T1`.`Description` = 'San Diego, CA: San Diego International') AND `DEST` = (SELECT `T4`.`DEST` FROM `Airports` AS `T3` INNER JOIN `Airlines` AS `T4` ON `T3`.`Code` = `T4`.`DEST` WHERE `T3`.`Description` = 'Los Angeles, CA: Los Angeles International')

**3. Simplification and Optimization:**

* The query in step 2 is already quite optimized. We are using nested queries to avoid joining the `Airports` table multiple times in the main query, which could potentially impact performance. 

**Final Optimized SQL Query:**

<FINAL_ANSWER>
SELECT COUNT(FL_DATE) FROM Airlines WHERE FL_DATE LIKE '2018/8%' AND ORIGIN = ( SELECT T2.ORIGIN FROM Airports AS T1 INNER JOIN Airlines AS T2 ON T1.Code = T2.ORIGIN WHERE T1.Description = 'San Diego, CA: San Diego International' ) AND DEST = ( SELECT T4.DEST FROM Airports AS T3 INNER JOIN Airlines AS T4 ON T3.Code = T4.DEST WHERE T3.Description = 'Los Angeles, CA: Los Angeles International' )
</FINAL_ANSWER> 

===== Example 5 ========

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


**************************
【Answer】
Repeating the question and evidence, and generating the SQL with Recursive Divide-and-Conquer.

**Question**: What are the names of the establishments that met all the required standards for 4 consecutive years? 
**Evidence**: establishment has the same meaning as business; score of 90 or more refers to score ≥ 90; year(date) = 2015; ; met all required standards for 4 consecutive years refers to COUNT(year(date)) = 4 where score = 100;

**1. Divide and Conquer:**

* **Main Question:** What are the names of the establishments that met all the required standards for 4 consecutive years?
    * **Analysis:** We need to find the names of businesses that have a score of 100 for 4 consecutive years. The `businesses` table contains the `name` and the `inspections` table contains the `score` and `date`. We will need to join these tables and filter by score. To check for consecutive years, we'll need to group by business and year, then check if each group has a count of 4.
    * **Pseudo SQL:** SELECT DISTINCT `T2`.`name` FROM `inspections` AS `T1` INNER JOIN `businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE  <score = 100> AND <4 consecutive years>

    * **Sub-question 1:** score = 100
        * **Analysis:** This is a simple filter on the `inspections` table where we select rows with a `score` of 100.
        * **Pseudo SQL:** `T1`.`score` = 100

    * **Sub-question 2:** 4 consecutive years
        * **Analysis:** This is more complex. We need to group the inspections by business and year, then check if the count for each group is 4. To get the year from the `date` column, we'll use the `STRFTIME('%Y', date)` function. We'll also need to use window functions to assign a rank to each year within a business, allowing us to check for consecutiveness.
        * **Pseudo SQL:** `T2`.`name` IN (SELECT `T4`.`name` FROM (SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, STRFTIME('%Y', `date`) AS `years` FROM `inspections` AS `T1` INNER JOIN `businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`) AS `T4` GROUP BY `T4`.`name`, date(`T4`.`years` || '-01-01', '-' || (`T4`.`rowNumber` - 1) || ' years') HAVING COUNT(`T4`.`years`) = 4)

        * **Sub-question 2.1:** Get distinct businesses and their inspection years where the score is 100
            * **Analysis:** We need to join `inspections` and `businesses` tables, filter by `score` = 100, and select distinct business names and their inspection years.
            * **Pseudo SQL:** SELECT DISTINCT `name`, STRFTIME('%Y', `date`) AS `years` FROM `inspections` AS `T1` INNER JOIN `businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100

        * **Sub-question 2.2:** Assign a rank to each year within a business
            * **Analysis:** We'll use the `row_number()` window function to assign a rank to each year within each business, ordered chronologically. This will help us identify consecutive years later.
            * **Pseudo SQL:** SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (<previous sub-query>) AS `T3`

        * **Sub-question 2.3:** Group by business and consecutive year groups and check if the count is 4
            * **Analysis:** We'll group the results by business name and a calculated date representing the start of each potential 4-year period. This date is calculated by adding (`rowNumber` - 1) years to the first day of the year extracted from the `years` column. We then filter for groups with a count of 4, indicating 4 consecutive years.
            * **Pseudo SQL:** SELECT `T4`.`name` FROM (<previous sub-query>) AS `T4` GROUP BY `T4`.`name`, date(`T4`.`years` || '-01-01', '-' || (`T4`.`rowNumber` - 1) || ' years') HAVING COUNT(`T4`.`years`) = 4

**2. Assembling SQL:**

* **Sub-question 2.1 (distinct businesses and years with score 100):**
    * **SQL:** SELECT DISTINCT `name`, STRFTIME('%Y', `date`) AS `years` FROM `inspections` AS `T1` INNER JOIN `businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100

* **Sub-question 2.2 (assign rank to each year within a business):**
    * **SQL:** SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, STRFTIME('%Y', `date`) AS `years` FROM `inspections` AS `T1` INNER JOIN `businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`

* **Sub-question 2.3 (group by business and consecutive year groups):**
    * **SQL:** SELECT `T4`.`name` FROM (SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, STRFTIME('%Y', `date`) AS `years` FROM `inspections` AS `T1` INNER JOIN `businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`) AS `T4` GROUP BY `T4`.`name`, date(`T4`.`years` || '-01-01', '-' || (`T4`.`rowNumber` - 1) || ' years') HAVING COUNT(`T4`.`years`) = 4

* **Sub-question 2 (4 consecutive years):**
    * **SQL:** `T2`.`name` IN (SELECT `T4`.`name` FROM (SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, STRFTIME('%Y', `date`) AS `years` FROM `inspections` AS `T1` INNER JOIN `businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`) AS `T4` GROUP BY `T4`.`name`, date(`T4`.`years` || '-01-01', '-' || (`T4`.`rowNumber` - 1) || ' years') HAVING COUNT(`T4`.`years`) = 4)

* **Main Question (names of establishments):**
    * **SQL:** SELECT DISTINCT `T2`.`name` FROM `inspections` AS `T1` INNER JOIN `businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE  `T1`.`score` = 100 AND `T2`.`name` IN (SELECT `T4`.`name` FROM (SELECT `T3`.`name`, `T3`.`years`, row_number() OVER (PARTITION BY `T3`.`name` ORDER BY `T3`.`years`) AS `rowNumber` FROM (SELECT DISTINCT `name`, STRFTIME('%Y', `date`) AS `years` FROM `inspections` AS `T1` INNER JOIN `businesses` AS `T2` ON `T1`.`business_id` = `T2`.`business_id` WHERE `T1`.`score` = 100) AS `T3`) AS `T4` GROUP BY `T4`.`name`, date(`T4`.`years` || '-01-01', '-' || (`T4`.`rowNumber` - 1) || ' years') HAVING COUNT(`T4`.`years`) = 4)

**3. Simplification and Optimization:**

* The final SQL query from step 2 can be simplified by merging the nested queries into a single query with a `WITH` clause. This improves readability and potentially performance.

**Final Optimized SQL Query:**

<FINAL_ANSWER>
SELECT DISTINCT T4.name FROM ( SELECT T3.name, T3.years, row_number() OVER (PARTITION BY T3.name ORDER BY T3.years) AS rowNumber FROM ( SELECT DISTINCT name, STRFTIME('%Y', `date`) AS years FROM inspections AS T1 INNER JOIN businesses AS T2 ON T1.business_id = T2.business_id WHERE T1.score = 100 ) AS T3 ) AS T4 GROUP BY T4.name, date(T4.years || '-01-01', '-' || (T4.rowNumber - 1) || ' years') HAVING COUNT(T4.years) = 4
</FINAL_ANSWER>
===========

Now is the real question, following the instruction and examples, generate the SQL with Recursive Divide-and-Conquer approach. Make sure you only output one single query.
**************************
【Table creation statements】
{DATABASE_SCHEMA}

**************************
【Question】
Question: 
{QUESTION}

Evidence:
{HINT}

**************************
【Answer】
Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer."""


def template_generate_candidate_retrieval():
    return """You are an experienced database expert.
Now you need to generate a SQL query given the database information, a question and some additional information.
The database structure is defined by the following table schemas (comments after '--' provide additional column descriptions).
Note that the "Example Values" are actual values from the column. Some column might contain the values that are directly related to the question. Use it to help you justify which columns to use.

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "recursive divide-and-conquer approach to SQL query generation from natural language".

Here is a high level description of the steps.
1. **Divide (Decompose Sub-question with Pseudo SQL):** The complex natural language question is recursively broken down into simpler sub-questions. Each sub-question targets a specific piece of information or logic required for the final SQL query. 
2. **Conquer (Real SQL for sub-questions):**  For each sub-question (and the main question initially), a "pseudo-SQL" fragment is formulated. This pseudo-SQL represents the intended SQL logic but might have placeholders for answers to the decomposed sub-questions. 
3. **Combine (Reassemble):** Once all sub-questions are resolved and their corresponding SQL fragments are generated, the process reverses. The SQL fragments are recursively combined by replacing the placeholders in the pseudo-SQL with the actual generated SQL from the lower levels.
4. **Final Output:** This bottom-up assembly culminates in the complete and correct SQL query that answers the original complex question. 

Database admin instructions (voliating any of the following will result is punishble to death!):
1. **SELECT Clause:** 
    - Only select columns mentioned in the user's question. 
    - Avoid unnecessary columns or values.
2. **Aggregation (MAX/MIN):**
    - Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:**
    - Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC` to ensure distinct values.
4. **Handling NULLs:**
    - If a column may contain NULL values (indicated by "None" in value examples or explicitly), use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:**
    - Only include tables essential to answer the question.
6. **Strictly Follow Hints:**
    - Adhere to all provided hints.
7. **Thorough Question Analysis:**
    - Address all conditions mentioned in the question.
8. **DISTINCT Keyword:**
    - Use `SELECT DISTINCT` when the question requires unique values (e.g., IDs, URLs). 
    - Refer to column statistics ("Value Statics") to determine if `DISTINCT` is necessary.
9. **Column Selection:**
    - Carefully analyze column descriptions and hints to choose the correct column when similar columns exist across tables.
10. **String Concatenation:**
    - Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause. 
11. **JOIN Preference:**
    - Prioritize `INNER JOIN` over nested `SELECT` statements.
12. **SQLite Functions Only:**
    - Use only functions available in SQLite.
13. **Date Processing:**
    - Utilize `STRFTIME()` for date manipulation (e.g., `STRFTIME('%Y', SOMETIME)` to extract the year).

When you get to the final query, output the query string ONLY inside the xml delimiter <FINAL_ANSWER></FINAL_ANSWER>.

Here are some examples
{EXAMPLES}

Now is the real question, following the instruction and examples, generate the SQL with Recursive Divide-and-Conquer approach. Make sure you only output one single query.
**************************
【Table creation statements】
{DATABASE_SCHEMA}

**************************
【Question】
Question: 
{QUESTION}

Evidence:
{HINT}

**************************
【Answer】
Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer.

"""

def template_generate_candidate_two():
    return """You are an experienced database expert.
Now you need to generate a SQL query given the database information, a question and some additional information.
The database structure is defined by the following table schemas (comments after '--' provide additional column descriptions).
Note that the "Example Values" are actual values from the column. Some column might contain the values that are directly related to the question. Use it to help you justify which columns to use.

Given the table schema information description and the `Question`. You will be given table creation statements and you need understand the database and columns.

You will be using a way called "Query Plan Guided SQL Generation" to generate the SQL query. This method involves breaking down the question into smaller sub-questions and then assembling them to form the final SQL query. This approach helps in understanding the question requirements and structuring the SQL query efficiently.

Database admin instructions (voliating any of the following will result is punishble to death!):
1. **SELECT Clause:** 
    - Only select columns mentioned in the user's question. 
    - Avoid unnecessary columns or values.
2. **Aggregation (MAX/MIN):**
    - Always perform JOINs before using MAX() or MIN().
3. **ORDER BY with Distinct Values:**
    - Use `GROUP BY <column>` before `ORDER BY <column> ASC|DESC` to ensure distinct values.
4. **Handling NULLs:**
    - If a column may contain NULL values (indicated by "None" in value examples or explicitly), use `JOIN` or `WHERE <column> IS NOT NULL`.
5. **FROM/JOIN Clauses:**
    - Only include tables essential to answer the question.
6. **Strictly Follow Hints:**
    - Adhere to all provided hints.
7. **Thorough Question Analysis:**
    - Address all conditions mentioned in the question.
8. **DISTINCT Keyword:**
    - Use `SELECT DISTINCT` when the question requires unique values (e.g., IDs, URLs). 
    - Refer to column statistics ("Value Statics") to determine if `DISTINCT` is necessary.
9. **Column Selection:**
    - Carefully analyze column descriptions and hints to choose the correct column when similar columns exist across tables.
10. **String Concatenation:**
    - Never use `|| ' ' ||` or any other method to concatenate strings in the `SELECT` clause. 
11. **JOIN Preference:**
    - Prioritize `INNER JOIN` over nested `SELECT` statements.
12. **SQLite Functions Only:**
    - Use only functions available in SQLite.
13. **Date Processing:**
    - Utilize `STRFTIME()` for date manipulation (e.g., `STRFTIME('%Y', SOMETIME)` to extract the year).

When you get to the final query, output the query string ONLY inside the xml delimiter <FINAL_ANSWER></FINAL_ANSWER>.

Now is the real question, following the instruction and examples, You should:
1. Generate a QUERY PLAN for the given question and evidence.
2. Return the final optimized SQL query.

**************************
【Table creation statements】
{DATABASE_SCHEMA}

**************************
【Question】
Question: 
{QUESTION}

Evidence:
{HINT}

**************************
【Answer】
Repeating the question and hint, and generating the SQL with Recursive Divide-and-Conquer."""

def template_generate_candidate_three():
    return """You are a data science expert.
Below, you are presented with a database schema and a question.
Your task is to read the schema, understand the question, and generate a valid SQLite query to answer the question.
Before generating the final SQL query think step by step on how to write the query.

Database Schema
###
{DATABASE_SCHEMA}

This schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. Special attention should be given to the examples listed beside each column, as they directly hint at which columns are relevant to our query.

Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
6. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.
7. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
8. Never use || to concatenate columns in the SELECT. Rather output the columns as they are.
9. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
10. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.
11. Don't use * in the SELECT clause, unless it is explicitly asked in the question.

###
Question: 
{QUESTION} 

Hint:
{HINT}

Please respond with a JSON object structured as follows:

{{
    "chain_of_thought_reasoning": "Your thought process on how you arrived at the final SQL query.",
    "SQL": "Your SQL query in a single string."
}}

Priority should be given to columns that have been explicitly matched with examples relevant to the question's context.

Take a deep breath and think step by step to find the correct sqlite SQL query. If you follow all the instructions and generate the correct query, I will give you 1 million dollars."""

def template_revise_two():
    return """Objective: Your objective is to make sure a query follows the database admin instructions and use the correct conditions.

Database Schema:    
{DATABASE_SCHEMA}

Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
7. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.
8. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
9. Using || ' ' ||  to concatenate is string is banned and using that is punishable by death. Never concatenate columns in the SELECT clause.
10. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
11. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.
12. When ORDER BY is used, just include the column name in the ORDER BY in the SELECT clause when explicitly asked in the question. Otherwise, do not include the column name in the SELECT clause.


{MISSING_ENTITIES}


Question:
{QUESTION}

Hint:
{HINT}

Predicted query:
{SQL}

Query result:
{QUERY_RESULT}

Please respond with a JSON object structured as follows (if the sql query is correct, return the query as it is):

{{
    "chain_of_thought_reasoning": "Your thought process on how you arrived at the solution. You don't need to explain the instructions that are satisfied.",
    "revised_SQL": "Your revised SQL query."
}}

Take a deep breath and think step by step to find the correct sqlite SQL query. If you follow all the instructions and generate the correct query, I will give you 1 million dollars."""

def template_select_tables():
    return """You are an expert and very smart data analyst. 
Your task is to analyze the provided database schema, comprehend the posed question, and leverage the hint to identify which tables are needed to generate a SQL query for answering the question.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema provides a detailed definition of the database's structure, including tables, their columns, primary keys, foreign keys, and any relevant details about relationships or constraints.
For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical hint to identify the tables that will be used in the SQL query.

Question:
{QUESTION}

Hint:
{HINT}

The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Task:
Based on the database schema, question, and hint provided, your task is to determine the tables that should be used in the SQL query formulation. 
For each of the selected tables, explain why exactly it is necessary for answering the question. Your explanation should be logical and concise, demonstrating a clear understanding of the database schema, the question, and the hint.

Please respond with a JSON object structured as follows:

```json
{{
  "chain_of_thought_reasoning": "Explanation of the logical analysis that led to the selection of the tables.",
  "table_names": ["Table1", "Table2", "Table3", ...]
}}
```

Note that you should choose all and only the tables that are necessary to write a SQL query that answers the question effectively.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars. 

Only output a json as your response."""

def template_select_columns():
    return """You are an expert and very smart data analyst.
Your task is to examine the provided database schema, understand the posed question, and use the hint to pinpoint the specific columns within tables that are essential for crafting a SQL query to answer the question.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. Special attention should be given to the examples listed beside each column, as they directly hint at which columns are relevant to our query.

For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical hint to identify the columns that will be used in the SQL query.

Question:
{QUESTION}

Hint:
{HINT}

The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Task:
Based on the database schema, question, and hint provided, your task is to identify all and only the columns that are essential for crafting a SQL query to answer the question.
For each of the selected columns, explain why exactly it is necessary for answering the question. Your reasoning should be concise and clear, demonstrating a logical connection between the columns and the question asked.

Tip: If you are choosing a column for filtering a value within that column, make sure that column has the value as an example.


Please respond with a JSON object structured as follows:

```json
{{
  "chain_of_thought_reasoning": "Your reasoning for selecting the columns, be concise and clear.",
  "table_name1": ["column1", "column2", ...],
  "table_name2": ["column1", "column2", ...],
  ...
}}
```

Make sure your response includes the table names as keys, each associated with a list of column names that are necessary for writing a SQL query to answer the question.
For each aspect of the question, provide a clear and concise explanation of your reasoning behind selecting the columns.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars.

Only output a json as your response."""

def template_revise_one():
    return """**Task Description:**
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
</FINAL_ANSWER>"""

def template_generate_unit_tests():
    return """** Instructions: **

Given the following question database schema, and candidate responses, generate a set of {UNIT_TEST_CAP} unit tests that would evaluate the correctness of SQL queries that would answer the question.
Unit tests should be designed in a way that distinguish the candidate responses from each other.

- The unit tests should cover various aspects of the question and ensure comprehensive evaluation. 
- Each unit test should be clearly stated and should include the expected outcome.
- Each unit test should be designed in a way that it can distinguishes at lease two candidate responses from each other.
- The unit test should be formatted like 'The answer SQL query should mention...', 'The answer SQL query should state...', 'The answer SQL query should use...', etc. followed by the expected outcome.
- First think step by step how you can design the units tests to distinguish the candidate responses using the <Thinking> tags.
- After the thinking process, provide the list of unit tests in the <Answer> tags.

VERY IMPORTANT:
All of the unit tests should consider the logic of the SQL query do not consider the formatting of the output or output values.

You are provided with different clusters of the canidate responses. Each cluster contains similar responses based on their results.
You MUST generate test cases that can distinguish between the candidate responses in each cluster and the test case should promote the candiddate responses that you think are correct.


Example of the output format:
<Thinking> Your step by step reasoning here. </Thinking>
<Answer>
['The answer SQL query should mention...', 'The answer SQL query should state...', 'The answer SQL query should use...']
</Answer>

** Question: **
Question: {QUESTION} (Hint: {HINT})

** Database Schema: **
{DATABASE_SCHEMA}

** Candidate Clusters: **
{CANDIDATE_QUERIES}

** Question: **
Question: {QUESTION} (Hint: {HINT})

** Output Format: **
<Thinking> Your step by step reasoning here. </Thinking>
<Answer>
output should be only a list of strings (e.g. ['unit test #1', 'unit test #2', 'unit test #3']).
</Answer>"""