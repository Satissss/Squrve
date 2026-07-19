
from __future__ import annotations
from collections import namedtuple
import itertools
import random
import re
from collections import defaultdict
from typing import Dict, Any, List, Optional
from operator import attrgetter
import networkx as nx


# 完全复用原论文的 Node 和 snode
Node = namedtuple("Node", ["name", "affiliation"])
snode = Node("source", "")


def schema2graph(schemas: dict) -> nx.DiGraph:
    """
    完全来自原论文：将多个数据库的 schema 转换为 networkx.DiGraph
    输入格式：{db_id: [{"name": table_name, "columns": [...], ...}, ...]}
    """
    G = nx.DiGraph()

    for database, tables in schemas.items():
        links = defaultdict(set)
        dnode = Node(database, "source")
        G.add_edge(snode, dnode)

        for table in tables:
            tnode = Node(table["name"], database)
            G.add_edge(dnode, tnode)
            for column in table.get("columns", []):
                foreign_key = column.get("foreign_key", None)
                if foreign_key:
                    key = f'{foreign_key["table"]}.{foreign_key["column"]}'
                    links[key].update([foreign_key["table"], table["name"]])

        for tables in links.values():
            for tbl1, tbl2 in itertools.product(tables, tables):
                if tbl1 != tbl2:
                    tnode1 = Node(tbl1, database)
                    tnode2 = Node(tbl2, database)
                    G.add_edge(tnode1, tnode2)

    return G


def from_multi_db_schema(multi_db_schema: Dict[str, Any]) -> nx.DiGraph:
    """
    适配不同格式的输入，转换为 schema2graph 需要的格式
    支持：
    1. Spider central format（含 table_names_original, column_names_original 等）
    2. List of dicts（[{"table_name": "t1", "columns": [...]}, ...]）
    3. Dict format（{table_name: {"columns": [...]}, ...}）
    """
    normalized = {}

    for db_id, schema_info in multi_db_schema.items():
        tables: List[Dict] = []

        if isinstance(schema_info, list):
            # List format: could be flat column-level entries or table-level entries
            # Check first item to determine format
            first_item = schema_info[0] if schema_info else {}
            first_item_is_col = isinstance(first_item, dict) and "column_name" in first_item

            if first_item_is_col:
                # Flat column-level format (spider single_db column files)
                # Group by table_name
                table_map: Dict[str, Dict] = {}
                for item in schema_info:
                    tbl_name = item.get("table_name", item.get("name", "unknown"))
                    col_name = item.get("column_name", item.get("name", ""))
                    if tbl_name not in table_map:
                        table_map[tbl_name] = {"name": tbl_name, "columns": []}
                    table_map[tbl_name]["columns"].append({"name": col_name})

                    # Parse foreign_key string (format: "[table.column]" or "")
                    fk_str = item.get("foreign_key", "")
                    if fk_str and isinstance(fk_str, str):
                        match = re.match(r"\[(.*?)\.(.*?)\]", fk_str)
                        if match:
                            ref_table, ref_col = match.groups()
                            table_map[tbl_name]["columns"].append({
                                "foreign_key": {"table": ref_table, "column": ref_col}
                            })
                tables = list(table_map.values())
            else:
                # Table-level format: [{"table_name": "t1", "columns": [...]}, ...]
                for item in schema_info:
                    t: Dict = {
                        "name": item.get("table_name", item.get("name", "unknown")),
                        "columns": []
                    }
                    cols = item.get("columns", [])
                    for col in cols:
                        if isinstance(col, str):
                            t["columns"].append({"name": col})
                        else:
                            t["columns"].append({
                                "name": col.get("column_name", col.get("name", ""))
                            })
                    # 处理 foreign keys
                    if "foreign_key" in item:
                        fk = item["foreign_key"]
                        if isinstance(fk, str) and fk:
                            match = re.match(r"\[(.*?)\.(.*?)\]", fk)
                            if match:
                                ref_table, ref_col = match.groups()
                                t["columns"].append({
                                    "foreign_key": {"table": ref_table, "column": ref_col}
                                })
                        elif isinstance(fk, dict):
                            t["columns"].append({"foreign_key": fk})
                    tables.append(t)

        elif isinstance(schema_info, dict):
            if "table_names_original" in schema_info:
                # Spider central format
                table_names = schema_info.get("table_names_original", [])
                column_names = schema_info.get("column_names_original", [])
                column_types = schema_info.get("column_types", [])
                primary_keys = schema_info.get("primary_keys", [])
                foreign_keys = schema_info.get("foreign_keys", [])

                for tbl_idx, tbl_name in enumerate(table_names):
                    t = {"name": tbl_name, "columns": []}
                    for col_idx, (c_tbl_idx, c_name) in enumerate(column_names):
                        if c_tbl_idx == tbl_idx:
                            t["columns"].append({"name": c_name})
                    tables.append(t)

                # 处理 foreign keys
                if foreign_keys:
                    for fk_entry in foreign_keys:
                        if isinstance(fk_entry, list) and len(fk_entry) == 2:
                            fk_col_idx, pk_col_idx = fk_entry
                            fk_tbl_idx, fk_col_name = column_names[fk_col_idx]
                            pk_tbl_idx, pk_col_name = column_names[pk_col_idx]
                            fk_tbl = table_names[fk_tbl_idx]
                            pk_tbl = table_names[pk_tbl_idx]

                            # 找到对应的表添加 foreign_key
                            for tbl in tables:
                                if tbl["name"] == fk_tbl:
                                    for col in tbl["columns"]:
                                        if col["name"] == fk_col_name:
                                            col["foreign_key"] = {
                                                "table": pk_tbl,
                                                "column": pk_col_name
                                            }
                                            break
                                    break
            else:
                # Dict format: {table_name: {"columns": [...]}}
                for tbl_name, tbl_info in schema_info.items():
                    t = {"name": tbl_name, "columns": []}
                    cols = tbl_info.get("columns", []) if isinstance(tbl_info, dict) else []
                    for col in cols:
                        if isinstance(col, str):
                            t["columns"].append({"name": col})
                        else:
                            t["columns"].append({
                                "name": col.get("column_name", col.get("name", ""))
                            })
                    tables.append(t)

        elif isinstance(schema_info, str):
            # String format: "Table: t1\n..."
            from .schema_serializer import parse_squrve_text
            parsed = parse_squrve_text(schema_info)
            for tbl_name, cols in parsed.items():
                tables.append({
                    "name": tbl_name,
                    "columns": [{"name": c} for c in cols]
                })

        normalized[db_id] = tables

    return schema2graph(normalized)


def _extract_table_columns(multi_db_schema: Dict, db_id: str, table_name: str) -> List[str]:
    if multi_db_schema is None:
        return []
    schema_info = multi_db_schema.get(db_id)
    if schema_info is None:
        return []

    if isinstance(schema_info, list):
        first_item = schema_info[0] if schema_info else {}
        if isinstance(first_item, dict) and "column_name" in first_item:
            cols = []
            for item in schema_info:
                if item.get("table_name", item.get("name")) == table_name:
                    cols.append(item.get("column_name", item.get("name", "")))
            return [c for c in cols if c]
        else:
            for item in schema_info:
                tbl_name = item.get("table_name", item.get("name", ""))
                if tbl_name == table_name:
                    cols = item.get("columns", [])
                    return [
                        c if isinstance(c, str) else c.get("column_name", c.get("name", ""))
                        for c in cols
                    ]
            return []

    if isinstance(schema_info, dict):
        if "table_names_original" in schema_info:
            table_names = schema_info.get("table_names_original", [])
            if table_name in table_names:
                tbl_idx = table_names.index(table_name)
                column_names = schema_info.get("column_names_original", [])
                return [cn for cti, cn in column_names if cti == tbl_idx]
            return []
        else:
            tbl_info = schema_info.get(table_name, {})
            if isinstance(tbl_info, dict):
                cols = tbl_info.get("columns", [])
                return [
                    c if isinstance(c, str) else c.get("column_name", c.get("name", ""))
                    for c in cols
                ]
            return []

    if isinstance(schema_info, str):
        from .schema_serializer import parse_squrve_text
        parsed = parse_squrve_text(schema_info)
        return parsed.get(table_name, [])

    return []


def to_summary_text(G: nx.DiGraph, multi_db_schema: Dict = None) -> str:
    """
    将 nx.DiGraph 转换为给 LLM 看的 summary text
    支持可选的 multi_db_schema 参数来包含列信息
    """
    parts = []
    d_nodes = [n for n in G[snode]]
    for d_node in sorted(d_nodes, key=lambda x: x.name):
        db_id = d_node.name
        parts.append(f"Database '{db_id}':")
        t_nodes = [n for n in G[d_node]]
        for t_node in sorted(t_nodes, key=lambda x: x.name):
            if multi_db_schema:
                columns = _extract_table_columns(multi_db_schema, db_id, t_node.name)
                if columns:
                    parts.append(f"  Table '{t_node.name}' (columns: {', '.join(columns)})")
                else:
                    parts.append(f"  Table '{t_node.name}'")
            else:
                parts.append(f"  Table '{t_node.name}'")
        # 获取外键关系
        # 查找该数据库下所有的 table -> table 边
        fk_strs = []
        for u, v, attrs in G.edges(data=True):
            if (
                hasattr(u, "affiliation") and hasattr(v, "affiliation")
                and u.affiliation == db_id
                and v.affiliation == db_id
                and hasattr(u, "name") and hasattr(v, "name")
                and u.name != db_id
                and v.name != db_id
            ):
                fk_strs.append(f"{u.name} → {v.name}")
        if fk_strs:
            parts.append(f"  Foreign keys: {', '.join(fk_strs)}")
        parts.append("")
    return "\n".join(parts)

