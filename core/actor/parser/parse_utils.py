import pandas as pd
import math
from typing import Union, List, Dict


def slice_schema_df(schema_df: pd.DataFrame, slice_size: int = 200, slice_num: int = 10):
    if not isinstance(schema_df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    total_rows = len(schema_df)

    if total_rows == 0:
        return [schema_df]

    result = []

    # Priority 1: slice_size
    if isinstance(slice_size, int) and slice_size > 0:
        if slice_size >= total_rows:
            return [schema_df]  # 返回自身
        else:
            for start in range(0, total_rows, slice_size):
                result.append(schema_df.iloc[start:start + slice_size])
            return result

    # Priority 2: slice_num
    if isinstance(slice_num, int) and slice_num > 0:
        if slice_num <= 1:  # 修正：当要求1个或更少分片时返回整个DataFrame
            return [schema_df]

        size = math.ceil(total_rows / slice_num)
        for start in range(0, total_rows, size):
            result.append(schema_df.iloc[start:start + size])
        return result

    return [schema_df]


def normalize_schema_links(
        schema_links: Union[List[str], Dict[str, List[str]]],
        output_type: str = "A"
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Normalize schema_links to a unified format.

    Args:
        schema_links: Input schema links in any format (List or Dict)
        output_type: Desired output type - "A", "B", or "C" (default: "A")
            - "A": List[str] with table.column only
            - "B": Dict with {"tables": [...], "columns": [...]}
            - "C": List[str] with table.column and literal values

    Returns:
        Normalized schema_links in the specified format
    """
    # Step 1: Extract columns and values from input
    columns, values = [], []

    if isinstance(schema_links, dict):
        # Type B input: {"tables": [...], "columns": [...]}
        raw_columns = schema_links.get("columns", [])
        columns = [_clean_column_ref(col) for col in raw_columns]
    elif isinstance(schema_links, list):
        # Type A or C input: List[str]
        for item in schema_links:
            cleaned = _clean_column_ref(item)
            if _is_column_ref(cleaned):
                columns.append(cleaned)
            else:
                values.append(item)
    else:
        raise ValueError(f"Invalid schema_links type: {type(schema_links)}")

    # Step 2: Convert to requested output format
    if output_type == "A":
        return list(dict.fromkeys(columns))  # Deduplicate while preserving order
    elif output_type == "B":
        tables = list(dict.fromkeys(col.split('.')[0] for col in columns))
        return {"tables": tables, "columns": columns}
    elif output_type == "C":
        return list(dict.fromkeys(columns + values))
    else:
        raise ValueError(f"Invalid output_type: {output_type}. Must be 'A', 'B', or 'C'")


def _clean_column_ref(ref: str) -> str:
    """Remove backticks, quotes, and extra whitespace from column reference."""
    return ref.strip().replace('`', '').replace('"', '').replace("'", '')


def _is_column_ref(ref: str) -> bool:
    """Check if a string is a column reference (table.column format)."""
    return '.' in ref and len(ref.split('.')) == 2 and all(ref.split('.'))


def format_schema_links(schema_links: Union[str, List[str], List[List[str]]], output_type: str = "A") -> str:
    if schema_links is None:
        return ""
    if isinstance(schema_links, str):
        return schema_links

    schema_links = normalize_schema_links(schema_links, output_type)

    if isinstance(schema_links, list):
        # 如果是嵌套列表（多个 parser 的结果），合并它们
        return "\n".join([str(link) for link in schema_links])
    elif isinstance(schema_links, dict):
        format_lis = []
        if "tables" in schema_links:
            format_lis.append("Linked Tables: " + str(schema_links["tables"]))
        if "columns" in schema_links:
            format_lis.append("Linked Columns: " + str(schema_links["columns"]))
        schema_links = "\n\n".join(format_lis)

    return schema_links


# ---------------------------------------------------------------------------
# AutoLink agent output parser
# ---------------------------------------------------------------------------

def parse_model_output(output: str):
    """Parse tool-call blocks from a raw AutoLink agent LLM response.

    Recognises @schema_retrieval, @sql_execution, @sql_draft,
    @add_schema, and @stop() blocks, including multi-line triple-quoted
    query arguments.

    Returns
    -------
    full_lines : list[str]
        Raw text of each matched tool-call block.
    tool_calls : list[dict]
        Parsed representation. Each dict has at least a 'tool' key;
        additional keys depend on the tool type:
        - schema_retrieval -> {tool, table, column, description}
        - sql_execution / sql_draft -> {tool, query}
        - add_schema -> {tool, table, column}
        - stop -> {tool}
    """
    import re as _re

    call_types = [
        "@schema_retrieval",
        "@sql_execution",
        "@sql_draft",
        "@sql_exploration",
        "@stop()",
        "@add_schema",
    ]

    lines = [l.strip() for l in output.splitlines()]
    blocks = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if any(line.startswith(ct) for ct in call_types):
            stack = []
            block_lines = [line]
            open_pos = line.find("(")
            if open_pos != -1:
                stack.append("(")
                for c in line[open_pos + 1:]:
                    if c == "(":
                        stack.append("(")
                    elif c == ")":
                        stack.pop()
                        if not stack:
                            break
                j = i + 1
                while stack and j < len(lines):
                    next_line = lines[j].strip()
                    block_lines.append(next_line)
                    for c in next_line:
                        if c == "(":
                            stack.append("(")
                        elif c == ")":
                            if stack:
                                stack.pop()
                            if not stack:
                                break
                    j += 1
                    if not stack:
                        break
                i = j
                blocks.append("\n".join(block_lines))
            else:
                i += 1
        else:
            i += 1

    full_lines = []
    tool_calls = []

    for block in blocks:
        for call_type in call_types:
            if not block.strip().startswith(call_type):
                continue
            full_lines.append(block)

            if call_type == "@schema_retrieval":
                table_m = _re.search(r'table\s*[:=]\s*["\']([^"\']*)["\']', block)
                col_m   = _re.search(r'column\s*[:=]\s*["\']([^"\']*)["\']', block)
                desc_m  = _re.search(r'description\s*[:=]\s*["\']([^"\']*)["\']', block)
                tool_calls.append({
                    "tool":        "schema_retrieval",
                    "table":       table_m.group(1) if table_m else "",
                    "column":      col_m.group(1)   if col_m   else "",
                    "description": desc_m.group(1)  if desc_m  else "",
                })

            elif call_type == "@add_schema":
                table_m = _re.search(r'table\s*[:=]\s*["\']([^"\']*)["\']', block)
                col_m   = _re.search(r'column\s*[:=]\s*["\']([^"\']*)["\']', block)
                tool_calls.append({
                    "tool":   "add_schema",
                    "table":  table_m.group(1) if table_m else "",
                    "column": col_m.group(1)   if col_m   else "",
                })

            elif call_type in ("@sql_execution", "@sql_draft"):
                tool_type = call_type[1:]
                qm = _re.search(r'query\s*[:=]\s*"""(.*?)"""', block, _re.DOTALL)
                if not qm:
                    qm = _re.search(r'query\s*[:=]\s*["\']([^"\']*)["\']', block)
                query = qm.group(1) if qm else ""
                tool_calls.append({"tool": tool_type, "query": query})

            elif call_type == "@stop()":
                tool_calls.append({"tool": "stop"})

            break  # only match first call_type per block

    return full_lines, tool_calls

