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


USE_LLM_EVALUATION = True

if USE_LLM_EVALUATION:
    from core.llm.QwenModel import QwenModel
    from core.utils import parse_json_from_str
    import json

    llm = QwenModel(model_name="qwen-plus", api_key="...")
    with open("data/ground_truth_res.json", "r", encoding="utf-8") as f:
        ground_truth_res = json.load(f)
else:
    llm = None
    ground_truth_res = None


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



EVALUATION_CRITERION = """1. The Principle of Search Space Decoupling
### Abstract Description: The pipeline must strictly separate the identification of schema elements (Parsing) from the synthesis of logic (Generation).

### Why it Improves Success: Attempting to generate SQL directly from a massive, raw schema leads to "hallucinated" columns. Decoupling ensures that the Generator only operates on a "high-confidence" subset of the database, minimizing logical noise.

### Evaluation Guidance: Does the pipeline always execute a Parse actor (or a parallel set of them) before any Generate or Scale actors are called?

2. The Principle of Methodological Consensus (Diversity)
### Abstract Description: When facing high complexity or ambiguity, the pipeline should deploy a "committee" of parallel actors with distinct internal logic (e.g., CoT-based vs. multi-agent).

### Why it Improves Success: No single LLM methodology is universal; what one model misses in a multi-table join, another may capture through iterative refinement. Parallelism maximizes the "Recall" of the correct SQL candidate.

### Evaluation Guidance: For "Complex" or "Vague" tasks, does the pipeline utilize a nested list of three or more diverse Generate and Scale actors?

3. The Principle of Sequential Refinement (The Optimizer Chain)
### Abstract Description: Optimization should not be a single event but a cumulative process where different "Optimizers" tackle specific error types (syntax vs. logical grounding) in sequence.

### Why it Improves Success: Chaining optimizers (e.g., RSLSQLOptimizer followed by CHESSOptimizer) allows the system to fix syntax errors first and then use the successful execution to verify more nuanced domain logic.

### Evaluation Guidance: Does the pipeline contain a sequence of multiple Optimize actors after the initial generation phase?

4. The Principle of Empirical Selection (Precision Shift)
### Abstract Description: After maximizing recall through parallel generation, the pipeline must transition to a high-precision state using an execution-based filter.

### Why it Improves Success: A pipeline is only as good as its final choice. Using a Select actor that evaluates runtime performance (e.g., execution time or result consistency) ensures that the system outputs the most "provably correct" candidate.

### Evaluation Guidance: Does the pipeline terminate with a Select actor (like FastExecSelector) to prune failed or suboptimal candidates from the final output?

5. The Principle of Structural Elasticity
### Abstract Description: The "depth" (number of sequential steps) and "width" (number of parallel actors) must scale linearly with the complexity and colloquialism of the input.

### Why it Improves Success: Simple queries are prone to over-engineering errors, while complex queries fail in "shallow" pipelines. Optimal success requires a longer chain (Parse → Generate → Optimize → Scale → Select) for complex/multi-turn tasks.

### Evaluation Guidance: Is the pipeline's length and parallelism proportional to the "Complexity Level" and "Question Style" provided in the task?

6. The Principle of Architectural Integrity (Type Compatibility)
### Abstract Description: Success is mathematically impossible if the "Informational Flow" is broken; every actor's output must perfectly satisfy the next actor's input requirements.

### Why it Improves Success: Even the most advanced LLM fails if it receives mismatched context (e.g., sending raw schema to an optimizer that expects a predicted SQL string). Format and type alignment are the absolute constraints for execution.

### Evaluation Guidance: Can you trace a continuous flow of data types (e.g., schema → schema_links → pred_sql) through the entire proposed list?
"""

def load_prompt_args(instance_id: str):
    ground_truth = ground_truth_res[instance_id]
    input_prompt = ground_truth['prompt']
    baseline_actor = ground_truth['parsed_seq']
    return input_prompt, baseline_actor


def evaluate_sql_by_llm(
    instance_id: str,
    pred_actor_seq: list
):
    available_ins_lis = list(ground_truth_res.keys())
    if instance_id not in available_ins_lis:
        return False, 0.0
    
    prompt_template = """# Role: Expert SQL Pipeline Auditor
You are an expert system architect specializing in Text-to-SQL Actor pipelines. Your task is to evaluate a **Predicted Actor Sequence** against a **Baseline Actor Sequence** for a specific SQL generation task.

# Evaluation Criteria(Success Principles):
You must judge the sequences based on the following 6 principles:
CRITERION

# Input Prompt:
INPUT_PROMPT

# Baseline Actor Sequence:
BASELINE_ACTOR_SEQUENCE

# Predicted Actor Sequence:
PREDICTED_ACTOR_SEQUENCE

# Comparison Task & Decision Logic
1. Analyze the **Input Prompt** for complexity and schema size.
2. Compare the **Predicted Sequence** to the **Baseline** using the Success Principles. 
3. Determine the result based on these specific rules:
   - **BETTER**: The Predicted Sequence more effectively adheres to the Principles (e.g., better schema pruning, higher parallel diversity, or improved error-correction).
   - **BETTER (Uncertainty/Tie-Breaker)**: If the sequences are identical, or if you are not significantly certain that the Predicted Sequence is worse, you **must** output `BETTER`.
   - **NOT_BETTER (Clear Inferiority)**: The Predicted Sequence introduces type errors, removes essential parsing for large schemas, or lacks a required selector for parallel branches.
   - **NOT_BETTER (Efficiency Violation)**: The Predicted Sequence is **NOT_BETTER** if it introduces excessive or unnecessary Actors that do not contribute to the success probability of the specific task (Over-engineering). Quality must be balanced with structural efficiency.

# Confidence Score Logic
- Assign a score from **0.0 to 1.0**.
- **Constraint**: If your judgment is `NOT_BETTER`, your confidence score must be **≥ 0.3**. If your confidence is lower than 0.3, it indicates high uncertainty; in such cases, you must default the judgment to `BETTER` (and adjust the score accordingly to reflect that new judgment).

# Output Format
Provide the final evaluation in a valid JSON object strictly following this structure:
{
  "reasoning": "A brief explanation focusing on why the predicted sequence is better (or worse) based on principles and efficiency.",
  "judgment": "BETTER" or "NOT_BETTER",
  "confidence_score": float
}
    """
    try:
        input_prompt, baseline_actor = load_prompt_args(instance_id)
        prompt = prompt_template.replace("CRITERION", EVALUATION_CRITERION)
        prompt = prompt.replace("INPUT_PROMPT", input_prompt)
        prompt = prompt.replace("BASELINE_ACTOR_SEQUENCE", str(baseline_actor))
        prompt = prompt.replace("PREDICTED_ACTOR_SEQUENCE", str(pred_actor_seq))

        response = llm.complete(prompt).text.strip()
        response = parse_json_from_str(response)
        score = float(response['confidence_score'])
        judgment = response['judgment']
        
        if score is None or judgment not in ['BETTER', 'NOT_BETTER']:
            return False, -0.5

        if judgment == "BETTER":
            final_score = 3 + 0.5 * score
            return True, final_score
        else:
            final_score = -0.5 * score
            return True, final_score

    except Exception as e:
        logger.error(f"Error in evaluate_sql_by_llm: {e}")
        return False, -0.5