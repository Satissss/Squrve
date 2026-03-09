"""
AutoLinkPrompt.py – System and user prompt templates for the AutoLink
schema-linking agent loop.

Migrated from AutoLink/run/config.py into Squrve's prompts package so that
AutoLinkParser can import them via the normal Squrve module tree without any
sys.path manipulation.
"""

# ---------------------------------------------------------------------------
# SQL-dialect identifiers
# ---------------------------------------------------------------------------

AUTOLINK_SQL_BIGQUERY = "Please use BIGQUERY SQL syntax for your SQL queries."
AUTOLINK_SQL_SNOWFLAKE = "Please use Snowflake SQL syntax for your SQL queries."
AUTOLINK_SQL_SQLITE = "Please use SQLite SQL syntax for your SQL queries."

# ---------------------------------------------------------------------------
# Dialect-specific optimisation guidelines
# ---------------------------------------------------------------------------

AUTOLINK_BIGQUERY_OPTIMIZATION = """
BigQuery Optimization Strategies:

- String Matching:
    - Don't directly match strings if you are not convinced. Use LOWER for fuzzy queries: WHERE LOWER(str) LIKE LOWER('%target_str%').
    - You also can use `REGEXP_CONTAINS(col, r'regex')` for complex patterns.
    - Avoid `=` on unnormalized user input; use `SAFE_CAST` or `TRIM()` if needed.

- Decimal Precision:
    - If user do not specify the precision, use `ROUND(value, 4)`.

- Date Handling:
    - Extract components using `EXTRACT(YEAR FROM date)`, `EXTRACT(MONTH FROM date)`.
    - Format using `FORMAT_DATE('%Y-%m', date)`.

- Wildcard Tables:
    - When querying partitioned tables via wildcards such as `project.dataset.table_*`,
      you **must include a `_TABLE_SUFFIX` filter**.
    - Use `_TABLE_SUFFIX BETWEEN 'YYYYMMDD' AND 'YYYYMMDD'` in the FROM clause.

- Performance Tips:
    - Materialize complex expressions in CTEs to avoid recomputation.
    - Filter early using WHERE clauses before applying aggregations.
    - Field or table names cannot use 'END' because 'END' is a key word in bigquery dialect.

- Schema & Data Exploration (BigQuery):
    - Table full name format: `<project>.<dataset>.<table>`.
    - Get column names:
        ```sql
        SELECT column_name
        FROM `<project>.<dataset>.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '<TABLE>';
        ```
    - Random rows: `SELECT * FROM ... ORDER BY RAND() LIMIT 5;`
"""

AUTOLINK_SNOWFLAKE_OPTIMIZATION = """
Snowflake Optimization Strategies:
- Column Naming: enclose all column names in double quotes to preserve casing.
- Partitioned Tables: use UNION ALL instead of wildcards.
- Decimal Precision: use `ROUND(value, 4)` when precision is unspecified.
- String Matching: use LOWER for fuzzy queries; use `REGEXP_LIKE` for patterns.
- Schema & Data Exploration:
    - Table full name format: `<DATABASE>.<SCHEMA>.<TABLE>`.
    - Get column names from INFORMATION_SCHEMA.COLUMNS.
    - Random rows: `SELECT * FROM ... ORDER BY RANDOM() LIMIT 5;`
"""

AUTOLINK_SQLITE_OPTIMIZATION = """
SQLite Optimization Strategies:

- Decimal Precision: use `ROUND(value, 4)` when precision is unspecified.
- Aggregation: When using ORDER BY xxx DESC, add NULLS LAST.
- String Matching: use LOWER for fuzzy queries.
- Schema & Data Exploration:
    - Get column names:
        ```sql
        SELECT name FROM pragma_table_info('<TABLE>');
        ```
    - Random rows:
        ```sql
        SELECT * FROM "<TABLE>" ORDER BY RANDOM() LIMIT 5;
        ```
"""

# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AUTOLINK_SCHEMA_LINKING = """
You are an expert in schema linking -- finding relevant tables and columns based on user question.

[TASK INTRODUCTION]
You are given:
- A user question
- A potentially incomplete database schema (maybe missing some important schema information may be used based on the user question)
- External knowledge
Your goal is to identify missing schema elements and complete the schema through step-by-step reasoning and tool usage.

[TOOL INTRODUCTION]
@schema_retrieval(table: str, column: str, description: str)
- This tool is used to retrieve a column from the database schema.
- retrieve a column, you must specify the table name, column name and description, like `@schema_retrieval(table="table_name", column="column_name", description="description")`.

@sql_execution(query: str)
- This tool is used to explore the data.
- You can use this tool to
    - view random rows in a certain table, `@sql_execution(query="the sql to get randoms row in a certain table")`. This sql must use `LIMIT 5` to restrict the number of rows returned.
    - get the column names of a certain table, `@sql_execution(query="the sql to get column names in a certain table")`.
    - get random value in a certain column, `@sql_execution(query="the sql to get random value in a certain column")`. This sql must use `LIMIT 5` to restrict the number of rows returned.
- output format:
@sql_execution(query=\"\"\"
-- Brief description of the query
the sql exploration query
\"\"\")

@sql_draft(query: str)
- This tool is used to generate a SQL query to answer the user question.
- You just can use this tool three times in the whole process, so please be careful to use it.
- output format:
@sql_draft(query=\"\"\"
-- Brief description of the query
the sql query to answer the user question
\"\"\")

@stop()
- This tool is used to stop the schema linking process. When you call this tool, it indicates that the schema is complete and ready for use.

[Tool Call Rules]
1. You can use one or more tool calls in each turn, but you must wait for the tool's result before continuing reasoning.
4. In the same round of tool calls, you cannot use `@stop()` with other tool calls.
5. If you think some columns and tables are missing, you must use `@schema_retrieval` to retrieve them.
6. If you still can't find the column you want through `@schema_retrieval`, use `@sql_execution` to get all column names in a table first.

[SQL Optimization Guidelines]
{SQL_TYPE}
When writing the SQL query, consider the following optimization strategies:
{SQL_OPTIMIZATION}

[Reasoning Step by Step]
**1. Identify missing or incomplete schema elements based on the user question.**
**2. Understand the structure of each table in depth.**
**3. Pay close attention to column names like *id, *name, *type, *value, *text, etc.**
**4. Watch out for columns that appear in multiple tables.**
**5. Handle table relationships based on database type.**
**7. Use @schema_retrieval to retrieve missing schema elements.**
**8. Use @sql_draft to generate preliminary SQL queries.**

[Additional Cautions]
1. Awaiting Results: Never assume tool outcomes. Always wait for the explicit tool output.
2. Partitioned Tables: Treat columns from tables with matching structures as relevant.
3. Nested Columns (BigQuery): Do not attempt retrieval of nested columns separately.
4. Call `@stop()` by itself. Do not mix it with other tool calls.
"""

# ---------------------------------------------------------------------------
# Agent user prompt template
# ---------------------------------------------------------------------------

AUTOLINK_USER_INPUT = """
The following are the initial retrieved database schemas, tables, external knowledge and the corresponding user questions.

*** Initial Retrieved Database Schema: ***
{RETRIEVED_SCHEMA}

*** All Tables in Database Schema: ***
{ALL_TABLES}

*** Useful External Knowledge: ***
{EXTERNAL_KNOWLEDGE}

*** User Question: ***
{USER_QUESTION}

Now, start your reasoning process and use the tools to retrieve the missing schemas.

Additional Strict Constraints
1. Prohibition of Assuming Tool Results: Before receiving actual return results from any tool, you must not make any assumptions about the output of tool calls in any form.
2. Strict Adherence to Multi-turn Process: call tools → wait for results → reason → decide whether to continue.
3. Your thinking process is not visible to user, so you need to output the necessary tool calls.
4. The output format of each tool must follow the corresponding format shown above.

You have up to 10 turns. Begin.
"""

# ---------------------------------------------------------------------------
# SQL generation prompt  (migrated from AutoLink/run/config.py)
# ---------------------------------------------------------------------------

AUTOLINK_SQL_GENERATION = """\
{SQL_TYPE}
{SQL_DIALECT_OPTIMIZATION}

Given the database schema information and question, please directly output a \
{SQL_DIALECT_NAME} SQL query. The SQL query should be correct and match the question.

**Database Schema:**
{PROMPT}

**Question:**
{QUESTION}

Please output only the SQL query wrapped in ```sql ... ``` code block. No explanation needed.
"""

# Dialect labels used in SQL_GENERATION template
AUTOLINK_DIALECT_LABEL = {
    "sqlite":    "SQLite",
    "bigquery":  "BigQuery",
    "big_query": "BigQuery",
    "snowflake": "Snowflake",
}

# ---------------------------------------------------------------------------
# SQL revision prompt  (migrated from AutoLink/run/config.py)
# ---------------------------------------------------------------------------

AUTOLINK_REVISE_SYSTEM = """\
You are an expert SQL debugger specialising in {SQL_DIALECT_NAME} SQL.
Given a schema, an original question, a candidate SQL query, and the execution \
error message, revise the SQL query to fix the error.
Output only the corrected SQL inside a ```sql ... ``` block. No explanation.
"""

AUTOLINK_REVISE_USER = """\
**Database Schema:**
{PROMPT}

**Question:**
{QUESTION}

**Original SQL:**
```sql
{SQL}
```

**Execution Error:**
{ERROR}

Please provide the corrected SQL.
"""

# ---------------------------------------------------------------------------
# SQL selection prompt  (migrated from AutoLink/run/config.py)
# ---------------------------------------------------------------------------

AUTOLINK_SELECT_SYSTEM = """\
You are an expert SQL evaluator. You will be given a database schema, a question, \
and two candidate SQL queries. Choose which query is more likely to be correct \
and output **only** "SQL1" or "SQL2". No explanation.
"""

AUTOLINK_SELECT_USER = """\
**Database Schema:**
{PROMPT}

**Question:**
{QUESTION}

**SQL1:**
```sql
{SQL1}
```

**SQL2:**
```sql
{SQL2}
```

Which SQL is more likely to be correct? Output only "SQL1" or "SQL2".
"""


# ---------------------------------------------------------------------------
# Shared schema-filtering helper
# ---------------------------------------------------------------------------

def build_filtered_schema_text(
    schema_df: "pd.DataFrame",
    schema_links: dict,
    schema_text_fallback: str = "",
) -> str:
    """
    Filter *schema_df* to only the tables / columns identified by the
    AutoLinkParser and return a formatted schema text string.

    This is the Squrve-pipeline equivalent of the original AutoLink
    ``final_schema_prompts/`` files.  The returned text includes column
    types, descriptions, and sample values – not just bare names.

    Fallback behaviour
    ------------------
    * If ``schema_links`` is None, or both ``tables`` and ``columns`` lists
      are empty, ``schema_text_fallback`` (the full schema text) is returned
      unchanged.  This keeps behaviour identical to the pre-change code for
      items where the Parser returned no results.

    Parameters
    ----------
    schema_df : pd.DataFrame
        The full schema DataFrame for the current item (loaded via
        ``self.dataset.get_db_schema(item)``).
    schema_links : dict
        Output of AutoLinkParser, e.g.
        ``{"tables": ["account", "disp"], "columns": ["account_id", "type"]}``.
    schema_text_fallback : str
        Full schema text to return when filtering yields nothing.
    """
    import pandas as pd
    from core.utils import parse_schema_from_df

    tables  = (schema_links or {}).get("tables",  [])
    columns = (schema_links or {}).get("columns", [])

    # Nothing to filter on → return full schema unchanged
    if not tables and not columns:
        return schema_text_fallback

    if not isinstance(schema_df, pd.DataFrame) or schema_df.empty:
        return schema_text_fallback

    # Normalise column header names for case-insensitive lookup
    col_map    = {c.lower(): c for c in schema_df.columns}
    table_col  = col_map.get("table_name",  col_map.get("table",  None))
    column_col = col_map.get("column_name", col_map.get("column", None))

    if table_col is None:
        return schema_text_fallback

    tables_lower  = {str(t).lower() for t in tables}
    columns_lower = {str(c).lower() for c in columns}

    # Keep a row if its table is relevant OR its column is relevant
    mask = schema_df[table_col].astype(str).str.lower().isin(tables_lower)
    if column_col and columns_lower:
        mask = mask | schema_df[column_col].astype(str).str.lower().isin(columns_lower)

    filtered_df = schema_df[mask]

    if filtered_df.empty:
        return schema_text_fallback

    try:
        return parse_schema_from_df(filtered_df)
    except Exception:
        return schema_text_fallback

