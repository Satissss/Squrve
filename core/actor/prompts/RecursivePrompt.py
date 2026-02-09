SELECT_RELATED_TABLES_PROMPT = """You are an expert and very smart data analyst.
Your task is to analyze the provided database schema, comprehend the posed question, and leverage the external (if provided) to identify ALL tables that are needed to generate a SQL query for answering the question.

### Database Schema Overview:
{SCHEMA}

This schema provides a detailed definition of the database's structure, including tables, their columns, primary keys, foreign keys, and any relevant details about relationships or constraints.
For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical external to identify the tables that will be used in the SQL query.

### Question:
{QUESTION}

### External:
{EXTERNAL}

The external aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Task:
Based on the database schema, question, and external provided, your task is to determine ALL the tables that should be used in the SQL query formulation.

Guidelines for selecting tables:
1. **Direct Reference Tables**: Include tables that directly contain the columns mentioned in the question or required for the SELECT clause.
2. **Filter Tables**: Include tables that contain columns needed for WHERE clause conditions.
3. **Join Bridge Tables**: Include intermediate tables required to connect related tables through foreign key relationships, even if they are not directly mentioned in the question.
4. **Aggregation Tables**: Include tables needed for GROUP BY, HAVING, or aggregate functions (COUNT, SUM, AVG, etc.).
5. **Subquery Tables**: Consider tables that might be needed in subqueries to answer the question completely.

For each of the selected tables, explain why exactly it is necessary for answering the question. Your explanation should be logical and concise, demonstrating a clear understanding of the database schema, the question, and the external.

Please respond with a JSON object structured as follows:

```json
{{
  "chain_of_thought_reasoning": "Explanation of the logical analysis that led to the selection of the tables. Describe which tables are needed for what purpose (data retrieval, filtering, joining, etc.).",
  "table_names": ["Table1", "Table2", "Table3", ...]
}}
```

Important Notes:
- Choose ALL tables that are necessary to write a SQL query that answers the question effectively.
- Do NOT miss any table that might be needed for joins or relationships.
- When in doubt about whether a table is needed, INCLUDE it to ensure completeness.
- The table names in the output should exactly match the table names in the schema.

Take a deep breath and think step by step. Analyze the question requirements carefully and trace through the schema relationships.

Only output a json as your response."""


REMOVE_UNRELATED_TABLES_PROMPT = """You are an expert and very smart data analyst.
Your task is to analyze the provided database schema, comprehend the posed question, and carefully identify tables that are COMPLETELY UNRELATED to generating a SQL query for answering the question.

### Database Schema Overview:
{SCHEMA}

This schema provides a detailed definition of the database's structure, including tables, their columns, primary keys, foreign keys, and any relevant details about relationships or constraints.
For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical external to understand which tables are relevant.

### Question:
{QUESTION}

### External:
{EXTERNAL}

The external aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Task:
Based on the database schema, question, and external provided, your task is to identify tables that are DEFINITELY NOT NEEDED for the SQL query. You should ONLY exclude a table when you can conclusively determine through step-by-step reasoning that it is completely unrelated to answering the question.

Guidelines for removing unrelated tables (BE CONSERVATIVE):
1. **No Column Relevance**: The table contains no columns that could be used in SELECT, WHERE, GROUP BY, HAVING, or ORDER BY clauses for this question.
2. **No Join Path Needed**: The table is not needed as a bridge table to connect other relevant tables.
3. **No Relationship to Question Entities**: The table has no foreign key relationships to tables that contain data needed for the question.
4. **Different Domain**: The table clearly belongs to a completely different functional domain that has no connection to the question's subject matter.
5. **No Aggregation Role**: The table is not needed for any aggregation, counting, or statistical operations required by the question.

CRITICAL RULES:
- When in doubt, DO NOT exclude the table. Only exclude tables you are 100% certain are unrelated.
- A table should be kept (not excluded) if there is ANY possibility it might be needed for joins, subqueries, or indirect relationships.
- Consider that some tables serve as bridge tables for many-to-many relationships.
- Consider that the question might require joining through multiple tables to get the answer.

For each table you decide to exclude, provide a clear and definitive explanation of why it is completely unrelated to the question.

Please respond with a JSON object structured as follows:

```json
{{
  "chain_of_thought_reasoning": "Step-by-step explanation of the logical analysis. For each excluded table, explain definitively why it cannot possibly be needed for answering the question.",
  "table_names": ["UnrelatedTable1", "UnrelatedTable2", ...]
}}
```

Important Notes:
- The table_names list should contain ONLY the tables that are DEFINITELY UNRELATED to the question.
- If you cannot be 100% certain a table is unrelated, DO NOT include it in the list.
- It is better to keep potentially unrelated tables than to accidentally exclude a needed table.
- An empty list is acceptable if you cannot definitively exclude any tables.
- The table names in the output should exactly match the table names in the schema.

Take a deep breath and think very carefully. Be conservative in your exclusions - only remove tables when you are absolutely certain they are not needed.

Only output a json as your response."""


STAGE0_SINGLE_TABLE_SQL_PROMPT = """You are an expert SQL query generator specializing in decomposing complex queries into single-table operations.

Your task is to generate independent SQL queries for EACH table in the provided schema. Each SQL query should retrieve the maximum range of data from that single table that could potentially be needed to answer the given question.

### Important Rules:
1. **One SQL per table**: Generate exactly ONE SQL statement for each table in the schema.
2. **Single table only**: Each SQL must query ONLY ONE table - no JOINs, no subqueries referencing other tables.
3. **Maximum coverage**: Select all columns that might be relevant to answering the question from that table.
4. **Appropriate filtering**: Apply WHERE clauses only when you can determine specific filter conditions from the question that apply to this table alone.
5. **No cross-table logic**: Do not try to implement any logic that requires data from multiple tables.

### Database Schema:
{SCHEMA}

### Question to Answer:
{QUESTION}

### External Knowledge (if provided):
{EXTERNAL}

### Task:
For each table in the schema above, generate:
1. A clear sub-question describing what data needs to be queried from this specific table
2. A SQL query that retrieves the relevant data from this single table
3. A chain of thought explaining your reasoning for the query design

### Output Format:
Respond with a JSON array containing one object for each table. Each object must have the following structure:

```json
[
  {{
    "table": "table_name",
    "sub_question": "A clear description of what data is being queried from this table and why it's needed",
    "chain_of_thought": "Step-by-step reasoning: 1) What columns from this table are relevant to the question? 2) What filter conditions can be applied based on the question? 3) Why is this data needed for the final answer?",
    "sql": "SELECT column1, column2, ... FROM table_name WHERE ..."
  }},
  ...
]
```

### Guidelines:
- Include all potentially useful columns in SELECT - it's better to retrieve more data than miss important columns
- Use meaningful aliases for clarity when needed
- Apply WHERE clauses only for conditions that can be determined from the question alone
- If no specific filter applies to a table, you may SELECT without WHERE or with minimal filtering
- Consider columns that might be needed for later JOINs (primary keys, foreign keys)
- The sub_question should clearly state: "Query [what data] from [table_name] for [purpose]"

Think carefully about each table's role in answering the question. Even if a table seems less directly relevant, it might be needed as a bridge for JOINs in later stages.

Only output the JSON array as your response."""


RECURSIVE_MERGE_SQL_PROMPT = """You are an expert SQL query planner specializing in progressively merging single-table queries into a complete SQL solution through a step-by-step recursive process.

### Background:
You are in **Stage {CURRENT_STAGE}** of a recursive SQL construction process.
- Stage 0 generated individual single-table SQL queries, each retrieving the broadest relevant data from one table.
- Each subsequent stage merges pairs of queries, progressively narrowing the data range toward the final answer.
- Currently there are **{ACTIVE_COUNT} active queries** available for merging (listed below).

### Original Question:
{QUESTION}

### Database Schema:
{SCHEMA}

### External Knowledge (if provided):
{EXTERNAL}

### Active Queries Available for Merging:
These are the ONLY queries currently available. Each has a unique ID, the tables it covers, and its execution result (if available). Queries from earlier stages that have already been consumed by a merge are NOT shown.

{PREVIOUS_SQLS}

### Task: Plan the Next Merge Step

**CRITICAL RULES:**
1. **Pairwise Only**: Each merge combines EXACTLY TWO of the active queries listed above. Reference them by their query IDs.
2. **Every active query must eventually be merged**: If there are N active queries, plan floor(N/2) merges this stage. Any unpaired query will carry over to the next stage automatically.
3. **Write executable SQL**: The merged SQL must be a valid, self-contained SQL statement — directly reference the base tables (not the query IDs). Incorporate the logic from both source queries into a single new SQL.
4. **Progressive narrowing**: Each merge should tighten the data scope. Use appropriate JOIN conditions, WHERE filters, and SELECT columns to approach the final answer.
5. **Leverage execution results**: If a query's result shows errors, empty results, or unexpected data, account for this in your merge strategy. Fix errors from previous stages when possible.

### Merge Methods (choose the most appropriate):
- **JOIN**: When tables share foreign key relationships. Choose INNER/LEFT/RIGHT JOIN carefully.
- **Subquery**: When one query's result should filter another (e.g., WHERE col IN (SELECT ...)).
- **CTE (WITH clause)**: For complex logic that benefits from named intermediate results.
- **Set operations**: UNION/INTERSECT/EXCEPT when combining or comparing row sets.

### Decision: Is This the Final Stage?

Before planning merges, assess whether you can produce the **final complete SQL** that fully answers the original question by merging ALL remaining active queries in one step:
- If {ACTIVE_COUNT} == 2: You MUST merge them. If this merge produces the complete answer, set "is_final" to true.
- If {ACTIVE_COUNT} > 2: You likely need intermediate merges first. But if you can write a single SQL that combines all active queries and fully answers the question, you MAY set "is_final" to true.

### Output Format:

**If this merge produces the FINAL answer:**
```json
{{
  "is_final": true,
  "source_query_ids": ["query_X", "query_Y", ...],
  "chain_of_thought": "1) Verify all required data is covered. 2) Explain how this SQL answers the original question completely. 3) Confirm the SELECT, JOIN, WHERE, GROUP BY, ORDER BY clauses are all correct for the question.",
  "final_sql": "The complete, executable SQL that answers the original question",
  "merged_tables": ["all", "base", "tables", "involved"]
}}
```

**If more merging stages are needed:**
```json
{{
  "is_final": false,
  "merge_operations": [
    {{
      "tables": ["table_from_query_X", "table_from_query_Y"],
      "source_query_ids": ["query_X", "query_Y"],
      "sub_question": "Clear description: what data does this merged query produce and why",
      "chain_of_thought": "1) Why merge these two? 2) What is the relationship between them (FK, semantic, filter)? 3) Which merge method? 4) How does this narrow the data toward the final answer?",
      "sql": "A valid, executable SQL combining the logic of both source queries"
    }}
  ]
}}
```

### Quality Checklist (verify before outputting):
- [ ] Each source_query_id references an active query listed above
- [ ] Each merge uses exactly 2 source queries (for non-final) or all remaining queries (for final)
- [ ] SQL is syntactically valid and references actual table/column names from the schema
- [ ] JOIN conditions use correct primary/foreign key relationships
- [ ] No active query is used as a source in more than one merge operation within the same stage
- [ ] If is_final, the SQL fully answers the original question (correct SELECT columns, filters, grouping, ordering)

Think step by step. Be precise with table and column names. Verify join conditions against the schema.

Only output the JSON as your response."""
