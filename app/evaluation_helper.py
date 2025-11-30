"""
独立的SQL评估模块，用于run_batch接口
参考 core.evaluate.py 实现，但独立于core模块
"""
import math
from pathlib import Path
from typing import Union, Dict, Tuple
import pandas as pd
from loguru import logger

from core.data_manage import load_dataset
from core.db_connect import get_sql_exec_result


class SQLEvaluationResult:
    """SQL评估结果"""
    
    def __init__(self):
        self.can_execute = False  # SQL是否可以执行（无语法错误）
        self.execution_error = None  # 执行错误信息
        self.is_correct = False  # 结果是否正确
        self.score = 0.0  # 评分
    
    def __repr__(self):
        return (f"SQLEvaluationResult(can_execute={self.can_execute}, "
                f"is_correct={self.is_correct}, score={self.score})")


def evaluate_sql_execution(
        pred_sql: str,
        gold_sql: str,
        db_id: str,
        db_type: str,
        db_path: Union[str, Path],
        db_credential: Dict = None
) -> SQLEvaluationResult:
    """
    评估预测的SQL执行情况
    
    Args:
        pred_sql: 预测的SQL语句
        gold_sql: 标准SQL语句
        db_id: 数据库ID
        db_type: 数据库类型 (sqlite, big_query, snowflake)
        db_path: 数据库路径
        db_credential: 数据库凭证
    
    Returns:
        SQLEvaluationResult: 评估结果
    """
    result = SQLEvaluationResult()
    
    try:
        # 处理pred_sql，如果是文件路径则加载
        if isinstance(pred_sql, str) and Path(pred_sql).is_file():
            pred_sql = load_dataset(pred_sql)
        
        if not pred_sql or not gold_sql:
            logger.warning("pred_sql or gold_sql is empty")
            result.execution_error = "Empty SQL query"
            return result
        
        # 构建数据库连接参数
        if db_type == "sqlite":
            db_path = Path(db_path) / (db_id + ".sqlite")
        
        base_exec_args = {
            "db_type": db_type,
            "db_path": db_path,
            "db_id": db_id,
            "credential_path": db_credential.get(db_type, None) if db_credential else None
        }
        
        # 执行标准SQL
        gold_args = {"sql_query": gold_sql, **base_exec_args}
        gold_result, gold_err = get_sql_exec_result(**gold_args)
        
        if gold_result is None:
            logger.error(f"Gold SQL execution failed: {gold_err}")
            result.execution_error = f"Gold SQL error: {gold_err}"
            return result
        
        # 执行预测SQL
        pred_args = {"sql_query": pred_sql, **base_exec_args}
        pred_result, pred_err = get_sql_exec_result(**pred_args)
        
        # 判断预测SQL是否可以执行
        if pred_result is None:
            logger.warning(f"Predicted SQL execution failed: {pred_err}")
            result.can_execute = False
            result.execution_error = pred_err
            return result
        
        # SQL可以执行，无语法错误
        result.can_execute = True
        
        # 比较执行结果
        try:
            result.is_correct = compare_sql_results(pred_result, gold_result)
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            result.is_correct = False
        
        return result
        
    except Exception as e:
        logger.exception(f"Evaluation error: {e}")
        result.execution_error = str(e)
        return result


def compare_sql_results(
        pred: pd.DataFrame,
        gold: pd.DataFrame,
        tolerance: float = 1e-2,
        ignore_order: bool = False
) -> bool:
    """
    比较两个SQL查询结果是否相同
    参考 core.evaluate.Evaluator.compare_pandas_table
    
    Args:
        pred: 预测结果DataFrame
        gold: 标准结果DataFrame
        tolerance: 数值比较容差
        ignore_order: 是否忽略顺序
    
    Returns:
        bool: 结果是否匹配
    """
    
    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        """比较两个向量是否匹配"""
        if ignore_order_:
            v1, v2 = (
                sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
            )
        
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
    
    try:
        # 转置并转换为列表
        t_gold_list = gold.transpose().values.tolist()
        t_pred_list = pred.transpose().values.tolist()
        
        # 检查每一列是否匹配
        for gold_col in t_gold_list:
            if not any(vectors_match(gold_col, pred_col, ignore_order_=ignore_order) 
                      for pred_col in t_pred_list):
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in compare_sql_results: {e}")
        return False


def calculate_score_from_evaluation(eval_result: SQLEvaluationResult) -> Tuple[float, Dict]:
    """
    根据评估结果计算分数
    
    评分规则：
    - SQL可执行（无语法错误）: +1
    - SQL不可执行（有语法错误）: -1
    - SQL执行结果正确: +1.5
    - SQL执行结果错误: -1.5
    
    Args:
        eval_result: SQL评估结果
    
    Returns:
        Tuple[float, Dict]: (分数, 详细信息)
    """
    score = 0.0
    details = {
        "can_execute": eval_result.can_execute,
        "is_correct": eval_result.is_correct,
        "execution_error": eval_result.execution_error
    }
    
    # SQL执行判断
    if eval_result.can_execute:
        score += 1.0
        details["execution_score"] = 1.0
    else:
        score -= 1.0
        details["execution_score"] = -1.0
    
    # SQL结果正确性判断（只有可执行时才判断）
    if eval_result.can_execute:
        if eval_result.is_correct:
            score += 1.5
            details["correctness_score"] = 1.5
        else:
            score -= 1.5
            details["correctness_score"] = -1.5
    else:
        details["correctness_score"] = 0.0
    
    return score, details

