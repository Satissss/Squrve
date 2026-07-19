
from __future__ import annotations
import random
from operator import attrgetter
from typing import Dict, List
import networkx as nx
from .schema_graph import Node, snode


def serialize_schema(
    schema: dict, G: nx.DiGraph, separator: str = "<sep>", shuffle: bool = True
) -> str:
    """
    完全来自原论文：将 schema 序列化为 DFS 前序遍历序列
    输入格式：{"database": "db_id", "metadata": [{"name": "t1", "columns": [...]}, ...]}
    """
    # Check separator is not in labels
    assert separator not in schema["database"]
    assert all(separator not in t["name"] for t in schema["metadata"])

    nodes = {
        snode,
        Node(schema["database"], "source"),
        *[Node(t["name"], schema["database"]) for t in schema["metadata"]],
    }
    stack = [snode]
    visited = []
    while stack:
        node = stack.pop()
        visited.append(node)
        if set(visited) == nodes:
            break

        children = [
            child for child in list(G[node]) if child in nodes and child not in visited
        ]
        if shuffle:
            random.shuffle(children)
        stack.extend(children)
    else:
        print(schema)

    return separator.join(map(attrgetter("name"), visited[1:]))


def deserialize_schema(s: str, separator: str = "<sep>") -> dict:
    """
    完全来自原论文：将字符串反序列化为 schema 字典
    输出：{"database": "db_id", "tables": ["t1", "t2", ...]}
    """
    # Remove space after separator for T5,
    # see https://github.com/huggingface/transformers/issues/24743
    s = s.replace(f"{separator} ", f"{separator}")

    database, *tables = s.split(f"{separator}")
    schema = {"database": database, "tables": tables}

    return schema


def parse_squrve_text(text: str) -> Dict[str, List[str]]:
    """
    保持兼容性：解析 Squrve 格式的文本 schema
    """
    result = {}
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("### Table = "):
            parts = line.split(", columns = ")
            if len(parts) == 2:
                tbl = parts[0].replace("### Table = ", "").strip("`'\"")
                cols_str = parts[1].strip()
                cols = eval(cols_str) if cols_str.startswith("[") else []
                if isinstance(cols, list):
                    result[tbl] = [c.strip(" `'\"") for c in cols]
    return result

