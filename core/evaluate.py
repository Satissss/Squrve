import math
import warnings
from os import PathLike
from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut

from core.data_manage import Dataset, load_dataset
from core.db_connect import get_sql_exec_result
from core.utils import parse_schema_link_from_str
from loguru import logger


class Evaluator:
    _eval_type_lis = [
        "reduce_recall", "reduce_rate", "reduce_precision",  # Reduce
        "parse_recall", "parse_precision", "parse_exact_matching",  # Parse
        "execute_accuracy"  # Generate
    ]

    def __init__(
            self,
            dataset: Dataset = None,
            eval_type: Union[str, List] = None,
            db_credential: Dict = None,  # A dict save all `credential` file path.
            db_path: Union[str, PathLike] = None,
    ):
        self.dataset: Dataset = dataset
        self.eval_type: List = self.__init_eval_type__(eval_type)
        self.eval_results: dict = {}
        self.db_credential: Dict = self.dataset.credential if not db_credential else db_credential
        self.db_path: Union[str, PathLike] = self.dataset.db_path if not db_path else db_path

    @classmethod
    def __init_eval_type__(cls, eval_type: Union[str, List] = None):
        if isinstance(eval_type, str):
            return [eval_type]

        elif isinstance(eval_type, list):
            return eval_type

        return []

    @staticmethod
    def _resolve_sql(row: dict, key: str) -> Optional[str]:
        """Resolve SQL from row by key (raw string or file path). Returns None if invalid."""
        raw = row.get(key)
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            sql = load_dataset(raw) if Path(raw).is_file() else raw
        except Exception:
            return None
        return sql if isinstance(sql, str) and sql.strip() else None

    def eval_all(self, verbose: bool = True):
        dataset = self.dataset
        eval_type = self.eval_type
        if not dataset or not eval_type:
            warnings.warn(f"dataset or eval type is not available.", category=UserWarning)
            return {}

        # Evaluation
        eval_results = {}
        for type_ in eval_type:
            if type_ not in self._eval_type_lis:
                warnings.warn(f"The eval_type `{type_}` is incorrect.", category=UserWarning)
                continue

            valid_num, res_lis, acc_res = 0, [], 0
            total_items = len(self.dataset)
            if verbose:
                print(f"Evaluating {type_} for {total_items} items...")

            for ind in range(total_items):
                try:
                    res = func_timeout(60, self.eval, args=(ind, type_))
                    if res is None:
                        print(f"Warning: Evaluation result is None for item {ind} in {type_}")
                        continue
                    res_lis.append([ind, res])
                    acc_res += res
                    valid_num += 1
                except FunctionTimedOut:
                    print(f"Timeout: Skipping item {ind} in {type_} (exceeded 60 seconds)")
                    continue
                except Exception as e:
                    print(f"Error evaluating item {ind} for {type_}: {e}")
                    continue
            if verbose:
                print(f"Completed {type_}: {valid_num}/{total_items} valid results")

            # 防止除零错误，当没有有效结果时设置默认值
            if valid_num == 0:
                eval_results[type_] = {
                    "avg": 0.0,
                    "results": res_lis,
                    "valid_num": valid_num,
                    "total_items": total_items,
                    "warning": f"No valid evaluation results found for {type_}. All {total_items} items failed evaluation."
                }
                warnings.warn(f"Warning: No valid results for {type_}, setting average to 0.0")
            else:
                avg_result = acc_res / valid_num
                eval_results[type_] = {
                    "avg": avg_result,
                    "results": res_lis,
                    "valid_num": valid_num,
                    "total_items": total_items
                }
                if verbose:
                    print(f"Average for {type_}: {avg_result:.4f}")

        self.eval_results.update(eval_results)
        return eval_results

    def eval(self, item, eval_type: str):
        """ 工厂评估方法，用于决定具体的评估方法调用。 """
        if eval_type not in self._eval_type_lis:
            return None

        res = None
        if eval_type == "reduce_recall":
            res = self.eval_reduce_recall(item)
        elif eval_type == "reduce_rate":
            res = self.eval_reduce_rate(item)
        elif eval_type == "reduce_precision":
            res = self.eval_reduce_precision(item)
        elif eval_type == "parse_recall":
            res = self.eval_parse_recall(item)
        elif eval_type == "parse_precision":
            res = self.eval_parse_precision(item)
        elif eval_type == "parse_exact_matching":
            res = self.eval_parse_exact_matching(item)
        elif eval_type == "execute_accuracy":
            res = self.eval_generate_execute_accuracy(item)

        return res

    """ Reduce """

    def eval_reduce_recall(self, item):
        try:
            row = self.dataset[item]
            if not isinstance(row, dict):
                print(f"Warning: Row {item} is not a dictionary")
                return None

            gold_schemas = row.get("gold_schemas", None)
            pred_schemas = load_dataset(row.get("instance_schemas", None))
            res = self.cal_schema_recall(gold_schemas, pred_schemas)

            return res
        except Exception as e:
            print(f"Error in eval_reduce_recall for item {item}: {e}")
            return None

    def eval_reduce_rate(self, item):
        try:
            row = self.dataset[item]
            if not isinstance(row, dict):
                print(f"Warning: Row {item} is not a dictionary")
                return None

            db_size = row.get("db_size", None)
            if db_size is None or db_size == 0:
                print(f"Warning: db_size is None or 0 for item {item}")
                return None

            pred_schemas = load_dataset(row.get("instance_schemas", None))
            pred_schemas = self._normalize_pred_schemas(pred_schemas)

            if pred_schemas is None:
                return None
            reduce_rate = len(pred_schemas) / db_size

            return reduce_rate
        except Exception as e:
            print(f"Error in eval_reduce_rate for item {item}: {e}")
            return None

    def eval_reduce_precision(self, item):
        try:
            row = self.dataset[item]
            if not isinstance(row, dict):
                print(f"Warning: Row {item} is not a dictionary")
                return None

            gold_schemas = row.get("gold_schemas", None)
            pred_schemas = load_dataset(row.get("instance_schemas", None))
            res = self.cal_schema_precision(gold_schemas, pred_schemas)

            return res
        except Exception as e:
            print(f"Error in eval_reduce_precision for item {item}: {e}")
            return None

    """ Parse """

    def eval_parse_recall(self, item):
        try:
            row = self.dataset[item]
            if not isinstance(row, dict):
                print(f"Warning: Row {item} is not a dictionary")
                return None

            gold_schemas = row.get("gold_schemas", None)
            schema_links = row.get("schema_links", None)
            pred_schemas = load_dataset(schema_links) if isinstance(schema_links, str) else schema_links
            res = self.cal_schema_recall(gold_schemas, pred_schemas)

            return res
        except Exception as e:
            print(f"Error in eval_parse_recall for item {item}: {e}")
            return None

    def eval_parse_precision(self, item):
        try:
            row = self.dataset[item]
            if not isinstance(row, dict):
                print(f"Warning: Row {item} is not a dictionary")
                return None

            gold_schemas = row.get("gold_schemas", None)
            schema_links = row.get("schema_links", None)
            pred_schemas = load_dataset(schema_links) if isinstance(schema_links, str) else schema_links
            res = self.cal_schema_precision(gold_schemas, pred_schemas)

            return res
        except Exception as e:
            print(f"Error in eval_parse_precision for item {item}: {e}")
            return None

    def eval_parse_exact_matching(self, item):
        try:
            row = self.dataset[item]
            if not isinstance(row, dict):
                print(f"Warning: Row {item} is not a dictionary")
                return None

            gold_schemas = row.get("gold_schemas", None)
            pred_schemas = load_dataset(row.get("schema_links", None))
            res = self.cal_schema_exact_matching(gold_schemas, pred_schemas)

            return res
        except Exception as e:
            print(f"Error in eval_parse_exact_matching for item {item}: {e}")
            return None

    """ Generate """

    def eval_generate_execute_accuracy(self, item):
        try:
            row = self.dataset[item]
            if not isinstance(row, dict):
                print(f"Warning: Row {item} is not a dictionary")
                return None
            gold_sql = self._resolve_sql(row, "query")
            if gold_sql is None:
                print(f"Warning: The gold sql is not available for item {item}")
                return None

            pred_sql = self._resolve_sql(row, "pred_sql")
            if pred_sql is None:
                print(f"Warning: The pred sql is not available for item {item}")
                return 0

            db_id = row.get("db_id", "")
            db_type = row.get("db_type", "")
            if not db_id or not db_type:
                print(f"Warning: Missing db_id or db_type for item {item}")
                return None
            if not self.db_path:
                print(f"Warning: Missing db_path for item {item}")
                return None

            db_path = Path(self.db_path) / (db_id + ".sqlite")
            base_exec_args = {
                "db_type": db_type,
                "db_path": db_path,
                "db_id": db_id,
                "credential_path": self.db_credential.get(db_type, None)
            }
            pred_args = {"sql_query": pred_sql, **base_exec_args}
            gold_args = {"sql_query": gold_sql, **base_exec_args}
            pred, pred_err = get_sql_exec_result(**pred_args)
            gold, gold_err = get_sql_exec_result(**gold_args)

            if gold is None:
                logger.warning(f"Ground-Truth SQL execution error for item {item}: {gold_err}")
                return None

            if pred is None:
                logger.warning(f"Predicted SQL execution failure for item {item}: {pred_err}")
                return 0
            score = self.compare_pandas_table(pred, gold)

            return score
        except Exception as e:
            print(f"Error in eval_generate_execute_accuracy for item {item}: {e}")
            return None

    @classmethod
    def _normalize_pred_schemas(cls, pred_schemas) -> Union[set, None]:
        """Normalize various input formats into a set of 'table.column' strings."""
        try:
            if isinstance(pred_schemas, pd.DataFrame):
                return {
                    f"{row['table_name']}.{row['column_name']}"
                    for _, row in pred_schemas.iterrows()
                }
            if isinstance(pred_schemas, str):
                pred_schemas = parse_schema_link_from_str(pred_schemas)

            if isinstance(pred_schemas, list):
                if all(isinstance(x, str) for x in pred_schemas):
                    return set(pred_schemas)
                if all(isinstance(x, dict) for x in pred_schemas):
                    return {
                        f"{row['table_name']}.{row['column_name']}"
                        for row in pred_schemas
                    }
                if all(isinstance(x, list) and len(x) == 2 for x in pred_schemas):
                    return {f"{tbl}.{col}" for tbl, col in pred_schemas}

        except Exception as e:
            print(f"[Error] Failed to normalize pred_schemas: {e}")
        return None

    @classmethod
    def cal_schema_recall(
            cls,
            gold_schemas: List,
            pred_schemas: Union[str, List[str], List[List[str]], List[Dict], pd.DataFrame]
    ):
        if not gold_schemas or pred_schemas is None:
            return None

        # Transform the item list into set
        pred_schemas = cls._normalize_pred_schemas(pred_schemas)
        if pred_schemas is None:
            return None

        # 防止除零错误
        if len(gold_schemas) == 0:
            return 0.0

        hit_count = sum(
            any(pred in gold for pred in pred_schemas)
            for gold in gold_schemas
        )

        return hit_count / len(gold_schemas)

    @classmethod
    def cal_schema_precision(
            cls,
            gold_schemas: List,
            pred_schemas: Union[str, List[str], List[List[str]], List[Dict], pd.DataFrame]
    ):
        if not gold_schemas or pred_schemas is None:
            return None

        # Transform the item list into set
        pred_schemas = cls._normalize_pred_schemas(pred_schemas)

        if pred_schemas is None:
            return None
        elif len(pred_schemas) == 0:
            return 0

        hit_count = sum(
            any(pred in gold for gold in gold_schemas)
            for pred in pred_schemas
        )

        return hit_count / len(pred_schemas)

    @classmethod
    def cal_schema_exact_matching(
            cls,
            gold_schemas: List,
            pred_schemas: Union[str, List[str], List[List[str]], List[Dict], pd.DataFrame]
    ):
        if not gold_schemas or pred_schemas is None:
            return None

        recall_ = cls.cal_schema_recall(gold_schemas, pred_schemas)
        precision_ = cls.cal_schema_precision(gold_schemas, pred_schemas)

        if recall_ is None or precision_ is None:
            return None

        return recall_ == precision_

    @classmethod
    def _is_na(cls, x) -> bool:
        """Safe scalar NA check that won't raise on array-like values."""
        if x is None:
            return True
        try:
            return bool(pd.isna(x))
        except (ValueError, TypeError):
            return False

    @classmethod
    def quick_reject(cls, pred: pd.DataFrame, gold: pd.DataFrame, ignore_order: bool) -> bool:
        """Return True if pred and gold are obviously not equivalent."""
        
        def _normalize_and_sort_row(row: Tuple) -> Tuple:
            # 标准化：NA→None, float→round
            normalized = tuple(
                None if cls._is_na(x) 
                else round(x, 2) if isinstance(x, float)
                else x
                for x in row
            )
            # 排序
            return tuple(sorted(normalized, key=lambda x: (x is None, type(x).__name__, str(x))))

        # 快速检查
        if pred.shape != gold.shape:
            return True

        # 转换并比较
        pred_rows = [_normalize_and_sort_row(tuple(row)) for row in pred.values]
        gold_rows = [_normalize_and_sort_row(tuple(row)) for row in gold.values]
        if ignore_order:
            return sorted(pred_rows) != sorted(gold_rows)
        
        return pred_rows != gold_rows


    @classmethod
    def compare_pandas_table(
        cls,
        pred,
        gold,
        condition_cols=None,
        ignore_order=False,
        strict_columns=False,
    ):
        """Compare two DataFrames for equivalence.

        Args:
            pred: Predicted result DataFrame.
            gold: Gold standard result DataFrame.
            condition_cols: Column indices to compare. Default [] means all columns.
            ignore_order: Whether to ignore order when comparing values within columns.
            strict_columns: If True, require pred and gold to have exactly the same columns 
                            (same number and same content). If False (default), allow pred to 
                            have extra columns beyond what's in gold, since SQL queries often 
                            have ambiguity about which columns to return.
        """
        if strict_columns and cls.quick_reject(pred, gold, ignore_order=ignore_order):
            return 0

        condition_cols = condition_cols or []
        tolerance = 1e-2

        def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
            if ignore_order_:
                sort_key = lambda x: (cls._is_na(x), type(x).__name__, str(x))
                v1, v2 = sorted(v1, key=sort_key), sorted(v2, key=sort_key)
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if cls._is_na(a) and cls._is_na(b):
                    continue
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if not math.isclose(float(a), float(b), abs_tol=tol):
                        return False
                elif a != b:
                    return False
            return True

        gold_cols_df = gold.iloc[:, condition_cols] if condition_cols else gold
        t_gold_list = gold_cols_df.transpose().values.tolist()
        t_pred_list = pred.transpose().values.tolist()

        if not t_gold_list:
            return 1

        # Each gold column must match a distinct pred column (bipartite assignment).
        used_pred_indices: set = set()
        for gold_col in t_gold_list:
            matched_idx = next(
                (j for j, pred_col in enumerate(t_pred_list)
                 if j not in used_pred_indices
                 and vectors_match(gold_col, pred_col, ignore_order_=ignore_order)),
                None,
            )
            if matched_idx is None:
                return 0
            used_pred_indices.add(matched_idx)

        return 1
