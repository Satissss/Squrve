from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from core.actor.selector.BaseSelect import BaseSelector
from core.data_manage import Dataset
from core.utils import load_dataset, save_dataset, sql_clean


class SQLPool:
    """A simple shared SQL pool for debate agents."""

    def __init__(self, sql_list: Optional[List[str]] = None):
        self._sql_pool: List[str] = []
        self.is_add: bool = False
        if sql_list:
            for s in sql_list:
                self.add_sql(s)

    def add_sql(self, sql: Union[str, List[str]]):
        if isinstance(sql, str):
            cleaned = sql_clean(sql)
            if cleaned and cleaned not in self._sql_pool:
                self._sql_pool.append(cleaned)
                self.is_add = True
            return

        if isinstance(sql, list):
            for s in sql:
                self.add_sql(s)
            return

        raise TypeError(f"sql must be str or List[str], got {type(sql)}")

    def get_sql(self) -> List[str]:
        return list(self._sql_pool)


@dataclass
class ProposerOutput:
    sql: str
    reason: str
    from_pool: bool


@dataclass
class ExpertOutput:
    agree: bool
    reason: str


class AgentDebateSelector(BaseSelector):
    """
    Multi-round debate selector:
    - Maintain a shared SQL pool (initialized from pred_sql).
    - Proposer agent selects best SQL from pool or proposes a new one.
    - Expert agent validates and either agrees (stop) or disagrees (continue).
    - Persist final SQL via BaseSelector.save_result; persist chat_history as JSON.
    """

    NAME = "AgentDebateSelector"

    PROPOSER_ROLE_NAME = "Data Analyst"
    PROPOSER_ROLE_DESCRIPTION = (
        "You are a senior data analyst specialized in Text-to-SQL. "
        "Given (question, schema, external knowledge, chat history) and a SQL pool, "
        "you must either (a) pick the single best SQL from the pool that perfectly matches the query, "
        "or (b) propose a new corrected SQL. Ground every decision in the provided schema/external knowledge. "
        "Be precise: do not invent tables/columns not present in schema. Prefer minimal, correct fixes. "
        "Output MUST be valid JSON."
    )

    EXPERT_ROLE_NAME = "Database Scientist"
    EXPERT_ROLE_DESCRIPTION = (
        "You are a strict database scientist and SQL auditor. "
        "You must carefully judge whether the candidate SQL perfectly matches the user's question "
        "and aligns with schema/external knowledge. If any mismatch, ambiguity, missing constraint, "
        "or likely schema hallucination exists, you MUST disagree. Be strict but fair. "
        "Output MUST be valid JSON."
    )

    PROPOSER_PROMPT = """### System
{role_description}

### Task
You are debating to find the best final SQL for a Text-to-SQL task.
Make a single decision: select one SQL from the pool OR propose a corrected SQL.

### Decision Policy
- Prefer an existing SQL if it already fully answers the question.
- If all candidates have issues, propose a corrected SQL with minimal necessary changes.
- Do not guess tables/columns; only use schema-provided items.
- If the question is ambiguous, choose the safest interpretation and state the assumption in reason.

### SQL Quality Checklist
- Correct tables/joins aligned with schema links and keys
- Correct filters/conditions (including dates, ranges, nulls, units)
- Correct aggregations/grouping/distinct
- Correct projection/order/limit as asked
- No schema hallucinations

### Context
[Question]
{question}

[DB Type]
{db_type}

[Schema]
{schema_text}

[Schema Links (Key tables/columns identified for this question)]
{schema_links_text}

[External Knowledge]
{external_text}

[Current SQL Pool]
{sql_pool_text}

[Chat History]
{chat_history_text}

### Output (STRICT JSON only)
Return a single JSON object with these keys:
- "sql": string, the chosen/proposed SQL
- "from_pool": boolean, true if you selected an existing SQL from the pool, else false
- "reason": string, a concise but complete justification
"""

    EXPERT_PROMPT = """### System
{role_description}

### Task
Judge whether the candidate SQL is correct and complete.
If correct, set agree=true and explain briefly.
If not correct, set agree=false and provide concrete reasons (missing filters/joins/aggregations, wrong columns, etc.).
You can compare with other SQL candidates in the pool to make a better judgment.

### Review Checklist
- Schema grounding: all tables/columns exist and joins are valid
- Question coverage: every constraint/metric is captured
- Aggregation correctness: group by, distinct, having, count/sum/avg
- Filtering details: time ranges, units, null handling, inclusions/exclusions
- Output shape: select list, order by, limit/top-k
- No unnecessary tables or filters

### Agreement Rule
Agree ONLY if every checklist item is satisfied with no ambiguity.

### Context
[Question]
{question}

[DB Type]
{db_type}

[Schema]
{schema_text}

[Schema Links (Key tables/columns identified for this question)]
{schema_links_text}

[External Knowledge]
{external_text}

[Candidate SQL]
{candidate_sql}

[Proposer Reason]
{proposer_reason}

[Other SQL Candidates in Pool (for reference)]
{other_sqls_text}

[Chat History]
{chat_history_text}

### Output (STRICT JSON only)
Return a single JSON object with:
- "agree": boolean
- "reason": string
"""

    def __init__(
        self,
        dataset: Dataset = None,
        llm: Any = None,
        is_save: bool = True,
        save_dir: Union[str, Path] = "../files/pred_sql/spider2",
        max_rounds: int = 4,
        max_schema_chars: int = 500000,
        max_history_chars: int = 500000,
        use_external: bool = True,
        **kwargs,
    ):
        super().__init__(dataset, llm, is_save, save_dir, **kwargs)
        self.max_rounds = max_rounds
        self.max_schema_chars = max_schema_chars
        self.max_history_chars = max_history_chars
        self.use_external = use_external

    @staticmethod
    def _get_llm(llm: Any) -> Any:
        if isinstance(llm, list):
            return llm[0] if llm else None
        return llm

    @staticmethod
    def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        # Try raw JSON first.
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try extracting the largest {...} block.
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(text[l : r + 1])
            except Exception:
                return None
        return None

    @staticmethod
    def _format_sql_pool(sqls: List[str], max_items: int = 20, exclude_sql: str = None) -> str:
        """Format SQL pool for display, optionally excluding a specific SQL."""
        if not sqls:
            return "(empty)"
        # Filter out the excluded SQL if provided
        filtered_sqls = [s for s in sqls if s != exclude_sql] if exclude_sql else sqls
        if not filtered_sqls:
            return "(no other candidates)"
        lines = []
        for i, s in enumerate(filtered_sqls[:max_items], start=1):
            lines.append(f"[{i}]\n{s}\n")
        if len(filtered_sqls) > max_items:
            lines.append(f"...({len(filtered_sqls) - max_items} more)")
        return "\n".join(lines).strip()

    @staticmethod
    def _schema_to_text(schema: Any) -> str:
        if schema is None:
            return ""
        if isinstance(schema, str):
            return schema
        if isinstance(schema, dict) or isinstance(schema, list):
            return json.dumps(schema, ensure_ascii=False, indent=2)
        if isinstance(schema, pd.DataFrame):
            # Keep it readable; markdown is usually easier for LLMs.
            try:
                return schema.to_markdown(index=False)
            except Exception:
                return schema.to_csv(index=False)
        return str(schema)

    def _load_schema_text(self, item: int, schema: Any) -> str:
        row = self.dataset[item] if self.dataset else {}
        if isinstance(schema, (str, PathLike)) and schema and Path(schema).exists():
            schema = load_dataset(schema)
        if schema is None:
            instance_schema_path = row.get("instance_schemas")
            if instance_schema_path:
                schema = load_dataset(instance_schema_path)
        if schema is None and self.dataset:
            try:
                schema = self.dataset.get_db_schema(item)
            except Exception:
                schema = None
        schema_text = self._schema_to_text(schema).strip()
        if self.max_schema_chars and len(schema_text) > self.max_schema_chars:
            schema_text = schema_text[: self.max_schema_chars] + "\n...(truncated)"
        return schema_text

    @staticmethod
    def _load_external_text(row: Dict[str, Any]) -> str:
        external_path = row.get("external")
        if not external_path:
            return ""
        try:
            external = load_dataset(external_path)
        except Exception:
            return ""
        if not external:
            return ""
        external_str = str(external)
        if len(external_str) > 20000:
            external_str = external_str[:20000] + "\n...(truncated)"
        return external_str

    def _load_schema_links_text(self, schema_links: Union[str, List[str], None], item: int) -> str:
        """Load schema links information."""
        if schema_links is None:
            # Try to load from dataset
            row = self.dataset[item] if self.dataset else {}
            schema_links = row.get("schema_links")
        
        if schema_links is None:
            return "(not provided)"
        
        if isinstance(schema_links, str):
            if Path(schema_links).exists():
                try:
                    schema_links = load_dataset(schema_links)
                except Exception:
                    pass
            return str(schema_links)
        
        if isinstance(schema_links, list):
            return "\n".join(str(s) for s in schema_links)
        
        return str(schema_links)

    def _chat_history_to_text(self, chat_history: Dict[str, Any]) -> str:
        text = json.dumps(chat_history, ensure_ascii=False, indent=2)
        if self.max_history_chars and len(text) > self.max_history_chars:
            return text[-self.max_history_chars :]
        return text

    def _call_llm(self, prompt: str) -> str:
        llm = self._get_llm(self.llm)
        if not llm:
            return ""
        try:
            return llm.complete(prompt).text
        except Exception as e:
            logger.warning(f"{self.NAME} | LLM call failed: {e}")
            return ""

    def _parse_proposer(self, raw: str, fallback_sql: str) -> ProposerOutput:
        payload = self._safe_json_loads(raw) or {}
        sql = payload.get("sql") if isinstance(payload, dict) else None
        reason = payload.get("reason") if isinstance(payload, dict) else None
        from_pool = payload.get("from_pool") if isinstance(payload, dict) else None

        sql = sql_clean(sql) if isinstance(sql, str) and sql.strip() else sql_clean(fallback_sql)
        reason = str(reason).strip() if reason is not None else ""
        from_pool = bool(from_pool) if isinstance(from_pool, (bool, int)) else False

        return ProposerOutput(sql=sql, reason=reason, from_pool=from_pool)

    def _parse_expert(self, raw: str) -> ExpertOutput:
        payload = self._safe_json_loads(raw) or {}
        agree = payload.get("agree") if isinstance(payload, dict) else None
        reason = payload.get("reason") if isinstance(payload, dict) else None

        agree_bool = bool(agree) if isinstance(agree, (bool, int)) else False
        reason_str = str(reason).strip() if reason is not None else raw.strip()
        return ExpertOutput(agree=agree_bool, reason=reason_str)

    def _save_chat_history(self, item: int, chat_history: Dict[str, Any], instance_id: Any) -> Optional[str]:
        if not self.is_save:
            return None
        save_path = Path(self.save_dir)
        if self.dataset and self.dataset.dataset_index:
            save_path = save_path / str(self.dataset.dataset_index)
        save_path = save_path / f"{self.NAME}_{instance_id}.chat_history.json"
        save_dataset(chat_history, new_data_source=save_path)
        if self.dataset:
            self.dataset.setitem(item, "chat_history", str(save_path))
        return str(save_path)

    def act(
        self,
        item,
        schema: Union[str, PathLike, Dict, List] = None,
        schema_links: Union[str, List[str]] = None,  # reserved for future use
        pred_sql: Union[str, PathLike, List[str], List[PathLike]] = None,
        data_logger=None,
        **kwargs,
    ):
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")

        row = self.dataset[item] if self.dataset else {}
        question = row.get("question", "")
        db_type = row.get("db_type", "sqlite")
        instance_id = row.get("instance_id", item)

        max_rounds = int(kwargs.get("max_rounds", self.max_rounds))
        max_rounds = max(1, max_rounds)

        pred_sql_list = self.load_pred_sql(pred_sql, item)
        if not pred_sql_list:
            if data_logger:
                data_logger.info(f"{self.NAME}.no_candidates | item={item}")
            return ""

        sql_pool = SQLPool(pred_sql_list)
        schema_text = self._load_schema_text(item, schema)
        schema_links_text = self._load_schema_links_text(schema_links, item)
        # Only load external knowledge if use_external is enabled
        external_text = self._load_external_text(row) if self.use_external else ""

        chat_history: Dict[str, Any] = {
            "agents": {
                "proposer": {"name": self.PROPOSER_ROLE_NAME, "description": self.PROPOSER_ROLE_DESCRIPTION},
                "expert": {"name": self.EXPERT_ROLE_NAME, "description": self.EXPERT_ROLE_DESCRIPTION},
            },
            "rounds": [],
        }

        # If no LLM, fallback to first SQL.
        if not self._get_llm(self.llm):
            best_sql = sql_clean(sql_pool.get_sql()[0])
            best_sql = self.save_result(best_sql, item, instance_id)
            self._save_chat_history(item, {**chat_history, "final_sql": best_sql, "stop_reason": "no_llm"}, instance_id)
            if data_logger:
                data_logger.info(f"{self.NAME}.selected_sql | sql={best_sql}")
                data_logger.info(f"{self.NAME}.act end | item={item}")
            return best_sql

        last_sql = sql_clean(sql_pool.get_sql()[0])
        stop_reason = "max_rounds"

        for r in range(1, max_rounds + 1):
            chat_history_text = self._chat_history_to_text(chat_history)
            sql_pool_text = self._format_sql_pool(sql_pool.get_sql())

            proposer_prompt = self.PROPOSER_PROMPT.format(
                role_description=self.PROPOSER_ROLE_DESCRIPTION,
                question=question,
                db_type=db_type,
                schema_text=schema_text,
                schema_links_text=schema_links_text,
                external_text=external_text,
                sql_pool_text=sql_pool_text,
                chat_history_text=chat_history_text,
            )
            proposer_raw = self._call_llm(proposer_prompt)
            proposer_out = self._parse_proposer(proposer_raw, fallback_sql=last_sql)

            if proposer_out.sql and proposer_out.sql not in sql_pool.get_sql():
                sql_pool.add_sql(proposer_out.sql)

            round_record: Dict[str, Any] = {
                "round": r,
                "proposer": {
                    "name": self.PROPOSER_ROLE_NAME,
                    "sql": proposer_out.sql,
                    "from_pool": proposer_out.from_pool,
                    "reason": proposer_out.reason,
                    "raw": proposer_raw,
                },
            }

            # Format other SQLs in pool (excluding the candidate) for expert reference
            other_sqls_text = self._format_sql_pool(sql_pool.get_sql(), exclude_sql=proposer_out.sql)
            
            expert_prompt = self.EXPERT_PROMPT.format(
                role_description=self.EXPERT_ROLE_DESCRIPTION,
                question=question,
                db_type=db_type,
                schema_text=schema_text,
                schema_links_text=schema_links_text,
                external_text=external_text,
                candidate_sql=proposer_out.sql,
                proposer_reason=proposer_out.reason,
                other_sqls_text=other_sqls_text,
                chat_history_text=self._chat_history_to_text({**chat_history, "pending_round": round_record}),
            )
            expert_raw = self._call_llm(expert_prompt)
            expert_out = self._parse_expert(expert_raw)

            round_record["expert"] = {
                "name": self.EXPERT_ROLE_NAME,
                "agree": expert_out.agree,
                "reason": expert_out.reason,
                "raw": expert_raw,
            }
            chat_history["rounds"].append(round_record)

            last_sql = proposer_out.sql or last_sql
            if data_logger:
                data_logger.info(
                    f"{self.NAME}.round_end | round={r} | agree={expert_out.agree} | sql={last_sql}"
                )

            if expert_out.agree:
                stop_reason = "expert_agree"
                break

        final_sql = sql_clean(last_sql)
        final_sql = self.save_result(final_sql, item, instance_id)

        chat_history_out = {**chat_history, "final_sql": final_sql, "stop_reason": stop_reason, "sql_pool": sql_pool.get_sql()}
        self._save_chat_history(item, chat_history_out, instance_id)

        if data_logger:
            data_logger.info(f"{self.NAME}.selected_sql | sql={final_sql}")
            data_logger.info(f"{self.NAME}.act end | item={item}")

        return final_sql