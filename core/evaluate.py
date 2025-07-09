import math
import warnings
from os import PathLike
from pathlib import Path
from typing import List, Union, Dict

import pandas as pd

from core.data_manage import Dataset, load_dataset
from core.db_connect import get_sql_exec_result
from core.utils import parse_schema_link_from_str


class Evaluator:
    """
        用于对 Text-to-SQL 流程提供专业的评估。
        需要设计的评估方法：
            Reduce:
            - recall: 标准数据库模式的召回率
            - reduce_rate: 减少数据库模式的数量占比
            - precision: 削减后数据库模式的精准率

            Parse:
            - recall: 模式链接的召回率
            - precision: 模式链接的准确率
            - exact matching: 模式链接的精准匹配率

            Generate:
            - execute accuracy: 生成 SQL 的执行准确率
    """

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

    def eval_all(self):
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
            for ind in range(len(self.dataset)):
                try:
                    res = self.eval(ind, type_)
                    if res is None:
                        continue
                    res_lis.append([ind, res])
                    acc_res += res
                    valid_num += 1
                except Exception as e:
                    print(f"There is something error when evaluating the {type_}. Here is the message: {e}")
            eval_results[type_] = {
                "avg": acc_res / valid_num,
                "results": res_lis,
                "valid_num": valid_num
            }
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
        row = self.dataset[item]
        assert isinstance(row, dict)

        gold_schemas = row.get("gold_schemas", None)
        pred_schemas = load_dataset(row.get("instance_schemas", None))
        res = self.cal_schema_recall(gold_schemas, pred_schemas)

        return res

    def eval_reduce_rate(self, item):
        row = self.dataset[item]
        assert isinstance(row, dict)

        db_size = row["db_size"]
        pred_schemas = load_dataset(row.get("instance_schemas", None))
        pred_schemas = self._normalize_pred_schemas(pred_schemas)

        if pred_schemas is None:
            return None
        reduce_rate = len(pred_schemas) / db_size

        return reduce_rate

    def eval_reduce_precision(self, item):
        row = self.dataset[item]
        assert isinstance(row, dict)

        gold_schemas = row.get("gold_schemas", None)
        pred_schemas = load_dataset(row.get("instance_schemas", None))
        res = self.cal_schema_precision(gold_schemas, pred_schemas)

        return res

    """ Parse """

    def eval_parse_recall(self, item):
        row = self.dataset[item]
        assert isinstance(row, dict)

        gold_schemas = row.get("gold_schemas", None)
        pred_schemas = load_dataset(row.get("schema_links", None))
        res = self.cal_schema_recall(gold_schemas, pred_schemas)

        return res

    def eval_parse_precision(self, item):
        row = self.dataset[item]
        assert isinstance(row, dict)

        gold_schemas = row.get("gold_schemas", None)
        pred_schemas = load_dataset(row.get("schema_links", None))
        res = self.cal_schema_precision(gold_schemas, pred_schemas)

        return res

    def eval_parse_exact_matching(self, item):
        row = self.dataset[item]
        assert isinstance(row, dict)

        gold_schemas = row.get("gold_schemas", None)
        pred_schemas = load_dataset(row.get("schema_links", None))
        res = self.cal_schema_exact_matching(gold_schemas, pred_schemas)

        return res

    """ Generate """

    def eval_generate_execute_accuracy(self, item):
        row = self.dataset[item]
        assert isinstance(row, dict)

        pred_sql = load_dataset(row.get("pred_sql", None))
        gold_sql = row.get("query", None)

        if not pred_sql or not gold_sql:
            warnings.warn(f"The pred sql or gold sql is not available.", category=UserWarning)
            return None

        db_path = Path(self.db_path) / (row["db_id"] + ".sqlite")
        base_exec_args = {
            "db_type": row["db_type"],
            "db_path": db_path,
            "db_id": row["db_id"],
            "credential_path": self.db_credential.get(row["db_type"], None)
        }
        pred_args = {"sql_query": pred_sql, **base_exec_args}
        gold_args = {"sql_query": gold_sql, **base_exec_args}
        pred, _ = get_sql_exec_result(**pred_args)
        gold, _ = get_sql_exec_result(**gold_args)

        if pred is None or gold is None:
            warnings.warn(f"SQL Execution Error!", category=UserWarning)
            return None
        score = self.compare_pandas_table(pred, gold)

        return score

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
    def compare_pandas_table(cls, pred, gold, condition_cols=None, ignore_order=False):
        """_summary_

        Args:
            pred (Dataframe): _description_
            gold (Dataframe): _description_
            condition_cols (list, optional): _description_. Defaults to [].
            ignore_order (bool, optional): _description_. Defaults to False.

        """
        if not condition_cols:
            condition_cols = []
        # print('condition_cols', condition_cols)

        tolerance = 1e-2

        def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
            if ignore_order_:
                v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                          sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if pd.isna(a) and pd.isna(b):
                    continue
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if not math.isclose(float(a), float(b), abs_tol=tol):
                        return False
                elif a != b:
                    return False
            return True

        if condition_cols:
            gold_cols = gold.iloc[:, condition_cols]
        else:
            gold_cols = gold
        pred_cols = pred

        t_gold_list = gold_cols.transpose().values.tolist()
        t_pred_list = pred_cols.transpose().values.tolist()
        score = 1
        for _, gold in enumerate(t_gold_list):
            if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
                score = 0
            else:
                for j, pred in enumerate(t_pred_list):
                    if vectors_match(gold, pred, ignore_order_=ignore_order):
                        break

        return score
