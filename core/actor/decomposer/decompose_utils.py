from typing import Union, List, Dict, Tuple

# Supported input: str | List[str] (DINSQL) | List[Tuple[str, str]] (MACSQL) | List[Dict] (Recursive)
SubQuestionsInput = Union[str, List[str], List[Tuple[str, str]], List[Dict]]


def normalize_sub_questions(
        sub_questions: SubQuestionsInput,
        output_type: str = "C"
) -> Union[List[str], List[Tuple[str, str]], List[Dict]]:
    """
    Normalize sub_questions to a unified format.

    Args:
        sub_questions: Input in any decomposer format:
            - str: single sub_question
            - List[str]: DINSQLDecomposer output
            - List[Tuple[str, str]]: MACSQLDecomposer (sub_question, sql)
            - List[Dict]: RecursiveDecomposer sql_containers (sub_question, sql, etc.)
        output_type: Desired output - "A", "B", or "C" (default: "A")
            - "A": List[str] with sub_question only
            - "B": List[Tuple[str, str]] with (sub_question, sql)
            - "C": List[Dict] with sub_question and literal values (sql, chain_of_thought)

    Returns:
        Normalized result according to output_type.
    """
    # Normalize to internal list of dicts (sub_question + optional sql, chain_of_thought)
    output: List[Dict] = []

    if isinstance(sub_questions, str):
        output = [{"sub_question": sub_questions, "sql": ""}]
    elif isinstance(sub_questions, list):
        if not sub_questions:
            output = []
        elif all(isinstance(item, str) for item in sub_questions):
            for item in sub_questions:
                output.append({"sub_question": item, "sql": ""})
        elif all(isinstance(item, tuple) for item in sub_questions):
            for item in sub_questions:
                q, s = (item[0], item[1]) if len(item) >= 2 else (item[0], "")
                output.append({"sub_question": q, "sql": s})
        elif all(isinstance(item, dict) for item in sub_questions):
            for item in sub_questions:
                new_sub: Dict = {
                    "sub_question": item.get("sub_question", ""),
                    "sql": item.get("sql", ""),
                }
                if "chain_of_thought" in item:
                    new_sub["chain_of_thought"] = item["chain_of_thought"]
                output.append(new_sub)

    if output_type == "A":
        return [item["sub_question"] for item in output]
    if output_type == "B":
        return [(item["sub_question"], item.get("sql", "")) for item in output]
    if output_type == "C":
        return output
    raise ValueError(f"Invalid output_type: {output_type}. Use 'A', 'B', or 'C'.")


def format_sub_questions(sub_questions: Union[str, List, List[Tuple[str, str]]], output_type: str = "C", max_num: int = 10) -> str:
    if sub_questions is None:
        return ""

    if isinstance(sub_questions, str):
        return sub_questions

    sub_questions = normalize_sub_questions(sub_questions, output_type=output_type)

    formatted = []
    for sub in sub_questions:
        if output_type == "A":
            formatted.append(f"Sub question: {sub}")
        elif output_type == "B":
            formatted.append(f"Sub question: {sub[0]}\nSQL:\n```sql\n{sub[1]}\n```")
        elif output_type == "C":
            valid_lis = []
            valid_lis.append(f"Sub question: {sub['sub_question']}")
            if "chain_of_thought" in sub:
                valid_lis.append(f"Chain of thought: {sub['chain_of_thought']}")
            valid_lis.append(f"SQL:\n```sql\n{sub['sql']}\n```")
            if "result" in sub:
                valid_lis.append(f"Result: {sub['result']}")
            formatted.append("\n".join(valid_lis))

    formatted.reverse()
    formatted = formatted[:max(max_num, len(formatted))]

    return "\n\n".join(formatted)
