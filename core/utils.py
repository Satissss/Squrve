from llama_index.core.schema import NodeWithScore
from typing import List, Dict, Union, Any
from os import PathLike
from pathlib import Path
import pandas as pd
import warnings
import json
import time
import random
import os
import torch
from loguru import logger


def parse_list_from_str(string: str = None) -> List[str]:
    """
    Parse a list from string. Expected format: "['a','b','c']"
    """
    try:
        cleaned = string.translate(str.maketrans('', '', '"\'[] \n`'))
        cleaned = cleaned.replace("python", "")

        return cleaned.split(',') if cleaned else []
    except Exception as e:
        print("Error parsing list from string!")
        raise ValueError("Invalid string format for list parsing") from e


def parse_json_from_str(string: str = None) -> dict:
    try:
        cleaned = string.translate(str.maketrans('', '', '\n`'))
        cleaned = cleaned.replace("json", "") if "json" in cleaned else cleaned
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError("Failed to parse JSON from string") from e


def get_all_files(directory, suffix: str = ".sql"):
    return [f.stem for f in Path(directory).iterdir() if f.is_file() and f.suffix == suffix]


def get_all_directories(directory):
    return [f.name for f in Path(directory).iterdir() if f.is_dir()]


def parse_schemas_from_nodes(
        nodes: List[NodeWithScore],
        schema_source: Union[str, PathLike] = None,
        output_format: str = None,
        multi_database: bool = False,
        db_id: str = None,
        **kwargs
):
    all_schema = []
    for node in nodes:
        file_path = None
        if schema_source:
            schema_source = Path(schema_source)
            if multi_database:
                file_path = schema_source / node.node.metadata["file_name"]
            elif db_id:
                file_path = schema_source / db_id / node.node.metadata["file_name"]
        if not file_path:
            file_path = Path(node.node.metadata["file_path"])
        if not file_path.exists():
            warnings.warn(f"读取文件时，给定路径无效，该文件不存在。文件路径为：{file_path}", category=UserWarning)
            continue
        col_info = load_dataset(file_path)
        if not isinstance(col_info, dict):
            continue
        schema = {
            "db_id": col_info["db_id"],
            "table_name": col_info["table_name"],
            "column_name": col_info["column_name"],
            'column_types': col_info["column_types"],
            'column_descriptions': col_info.get("column_descriptions", None),
            'sample_rows': col_info.get("sample_rows", None),
            'turn_n': node.metadata.get("turn_n", None)
        }
        all_schema.append(schema)

    output_format = "dataframe" if not output_format else output_format
    if output_format == "dataframe":
        all_schema = pd.DataFrame(all_schema)

    return all_schema


def parse_schema_from_df(df: pd.DataFrame) -> str:
    grouped = df.groupby('table_name')
    output_lines = []

    for table_name, group in grouped:
        columns = []
        for _, row in group.iterrows():
            col_type = row["column_types"]
            if isinstance(col_type, str) and len(col_type) > 150:
                col_type = col_type[:150]
            columns.append(f'{row["column_name"]}(Type: {col_type})')

        line = f'### Table {table_name}, columns = [{", ".join(columns)}]'
        output_lines.append(line)

    return "\n".join(output_lines)


def set_node_turn_n(node: NodeWithScore, turn_n: int):
    node.metadata["turn_n"] = turn_n
    return node


def load_dataset(data_source: Union[str, PathLike]):
    """
    Load dataset from a given file path. Supports .json, .txt, .sql, .csv, .xlsx.

    Args:
        data_source (str or PathLike): Path to the data file.

    Returns:
        Loaded dataset (dict, str, or pd.DataFrame), or None if file does not exist.
    """

    data_source = Path(data_source)
    # logger.info(f"load the dataset from the source:{data_source}")
    if not data_source.exists():
        logger.info("Invalid path: the file does not exist.", category=UserWarning)
        return None

    dataset = None
    if data_source.suffix == ".json":
        with open(data_source, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    elif data_source.suffix in (".txt", ".sql", ".md"):
        with open(data_source, "r", encoding="utf-8") as f:
            dataset = f.read().strip()
    elif data_source.suffix == ".csv":
        dataset = pd.read_csv(data_source)
    elif data_source.suffix == ".xlsx":
        dataset = pd.read_excel(data_source)

    return dataset


def save_dataset(
        dataset: Union[str, List, Dict] = None,
        old_data_source: Union[str, PathLike] = None,
        new_data_source: Union[str, PathLike] = None
):
    if old_data_source:
        dataset = load_dataset(old_data_source)
    if dataset is None:
        warnings.warn(f"Unable to save file, file is empty.", category=UserWarning)
        return
    if new_data_source is None:
        warnings.warn(f"Unable to save file, save dir is empty.", category=UserWarning)
        return
    new_data_source = Path(new_data_source) if isinstance(new_data_source, str) else new_data_source
    new_data_source.parent.mkdir(parents=True, exist_ok=True)

    if new_data_source.suffix == ".json":
        with open(new_data_source, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
    elif new_data_source.suffix in ('.txt', '.sql'):
        with open(new_data_source, "w", encoding="utf-8") as f:
            f.write(str(dataset))
    elif new_data_source.suffix == ".csv":
        dataset.to_csv(str(new_data_source), index=False, encoding='utf-8')
    elif new_data_source.suffix == ".xlsx":
        dataset.to_excel(str(new_data_source), index=False)


def parse_schema_link_from_str(string: str) -> List:
    schema_links = string.split("[")[1].split("]")[0].strip()
    cleaned_links = schema_links.split(",")
    remove_chars = str.maketrans('', '', '`"\'')

    return [link.strip().translate(remove_chars) for link in cleaned_links if link.strip()]


def sql_clean(raw_sql: str):
    cleaned_sql = (
        raw_sql
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("```", "")
        .replace("sql", "")
        .strip()
    )
    return cleaned_sql


def throw_hash_id(ind: int):
    """ A simple hash method to map any number to an Int(100-999) """
    h = (ind ^ (ind >> 3)) * 2654435761
    return (h % 900) + 100


def timestamp_hash_key():
    """ Get a unique number related to timestamp. """
    ts = int(time.time())
    rand = random.randint(0, 99)
    return int(f"{ts % 1000000}{rand:02d}")


def get_safe_device():
    """
    Get a safe device configuration to avoid meta tensor issues.
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    try:
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except Exception:
        return "cpu"


def initialize_model_safely(model_class, model_name, **kwargs):
    """
    Safely initialize a model to avoid meta tensor issues.
    
    Args:
        model_class: The model class to instantiate
        model_name: The model name/path
        **kwargs: Additional arguments for model initialization
        
    Returns:
        The initialized model
    """
    device = get_safe_device()

    # Add device to kwargs if not already present
    if 'device' not in kwargs:
        kwargs['device'] = device

    try:
        return model_class(model_name, **kwargs)
    except Exception as e:
        if "meta tensor" in str(e).lower():
            # Try with explicit device mapping
            kwargs['device_map'] = 'auto'
            if 'device' in kwargs:
                del kwargs['device']
            return model_class(model_name, **kwargs)
        else:
            raise e
