import re
from typing import Union, List, Optional, Dict
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.scaler.BaseScale import BaseScaler
from core.data_manage import Dataset
from core.utils import parse_schema_from_df
from llama_index.core.llms.llm import LLM


class MACSQLScaler(BaseScaler):
    """Scaler implementation based on MAC-SQL's SQL generation strategy for producing multiple SQL candidates."""

    NAME = "MACSQLScaler"

    # MAC-SQL SQL generation template with multi-agent collaboration approach
    SQL_GENERATION_TEMPLATE = '''You are part of a multi-agent SQL generation team. Each agent has specific expertise:

Agent 1 (Schema Analyst): Analyzes database structure and relationships
Agent 2 (Question Interpreter): Understands natural language requirements  
Agent 3 (Query Constructor): Builds the final SQL query

Database Schema:
{schema}

Question: {question}

Evidence: {evidence}

Schema Links (Critical Tables & Columns): {schema_links}

Sub-questions: {sub_questions}

Multi-Agent Collaboration Process:
1. **Schema Analysis**: Identify relevant tables, columns, and relationships
2. **Question Decomposition**: Break down the question using sub-questions and evidence
3. **Query Construction**: Build SQL using schema links and decomposed requirements

Instructions:
- Use the schema links to focus on relevant tables and columns
- Consider the evidence for additional context and constraints
- Apply sub-questions to decompose complex requirements
- Generate syntactically correct SQL that addresses all aspects

Please generate a SQL query that answers the question. Return only the SQL statement:'''

    FALLBACK_TEMPLATE = '''You are an expert SQL developer. Please generate a SQL query for the following question.

Question: {question}

Database Schema:
{schema}

Please generate a valid SQL query. Only return the SQL statement:'''

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Union[LLM, List[LLM]] = None,
            generate_num: int = 8,
            temperature: float = 0.6,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            open_parallel: bool = True,
            max_workers: int = None,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.generate_num = generate_num
        self.temperature = temperature

    def _generate_single_sql(self, llm_: LLM, question: str, schema: str, evidence: str = "", 
                           schema_links: str = "", sub_questions: str = "", use_fallback: bool = False) -> Optional[str]:
        """Generate a single SQL candidate using MAC-SQL approach"""
        try:
            if use_fallback:
                prompt = self.FALLBACK_TEMPLATE.format(
                    question=question,
                    schema=schema
                )
            else:
                prompt = self.SQL_GENERATION_TEMPLATE.format(
                    schema=schema,
                    question=question,
                    evidence=evidence,
                    schema_links=schema_links,
                    sub_questions=sub_questions
                )
                
            response = llm_.complete(prompt, temperature=self.temperature).text
            
            # Clean up SQL output
            sql = response.strip()
            
            # Remove code block markers
            if sql.startswith('```sql'):
                sql = sql[6:]
            elif sql.startswith('```'):
                sql = sql[3:]
            if sql.endswith('```'):
                sql = sql[:-3]
            
            sql = sql.strip()
            
            # Look for SELECT statements
            if not sql.upper().startswith('SELECT'):
                lines = sql.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.upper().startswith('SELECT'):
                        sql = line
                        break
                else:
                    return None
            
            return sql if sql else None
            
        except Exception as e:
            logger.warning(f"Failed to generate SQL candidate: {e}")
            return None

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            sub_questions: Union[str, List[str]] = None,
            **kwargs
    ) -> List[str]:
        row = self.dataset[item]
        question = row['question']
        evidence = row.get('evidence', '') or kwargs.get('evidence', '') or ''
        
        # Load and process schema using base class method
        schema = self.process_schema(schema, item)
        
        # Process schema_links
        if isinstance(schema_links, list):
            schema_links_str = ', '.join(schema_links)
        elif schema_links is None:
            schema_links_str = ""
        else:
            schema_links_str = str(schema_links)

        # Process sub_questions
        if isinstance(sub_questions, list):
            sub_questions_str = '\n'.join([f"- {q}" for q in sub_questions])
        elif sub_questions is None:
            sub_questions_str = ""
        else:
            sub_questions_str = str(sub_questions)

        # Get LLM
        if isinstance(self.llm, list) and self.llm:
            llm = self.llm[0]
        else:
            llm = self.llm

        if llm is None:
            logger.warning("No LLM available for SQL generation")
            return []

        # Generate multiple SQL candidates
        pred_sqls = []
        
        # Generate candidates with different temperature variations
        temperatures = [self.temperature * 0.8, self.temperature, self.temperature * 1.2]
        samples_per_temp = max(1, self.generate_num // len(temperatures))
        
        for temp in temperatures:
            original_temp = self.temperature
            self.temperature = temp
            
            for i in range(samples_per_temp):
                # Use fallback template for some variations
                use_fallback = i % 3 == 2  # Every third attempt uses fallback
                
                sql = self._generate_single_sql(
                    llm, question, schema, evidence, 
                    schema_links_str, sub_questions_str, use_fallback
                )
                if sql:
                    pred_sqls.append(sql)
            
            self.temperature = original_temp
        
        # Fill remaining slots if needed
        while len(pred_sqls) < self.generate_num:
            sql = self._generate_single_sql(
                llm, question, schema, evidence, 
                schema_links_str, sub_questions_str, False
            )
            if sql:
                pred_sqls.append(sql)
            else:
                break

        # Deduplicate
        pred_sqls = list(dict.fromkeys(pred_sqls))

        # Ensure at least one SQL result
        if not pred_sqls:
            logger.warning(f"No SQL candidates generated for item {item}, creating default SQL")
            pred_sqls = ["SELECT * FROM table LIMIT 1"]

        logger.info(f"MACSQLScaler: Generated {len(pred_sqls)} SQL candidates for item {item}")

        # Save results using base class method
        self.save_results(pred_sqls, item, row.get("instance_id"))

        return pred_sqls
