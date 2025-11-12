import re
from typing import Union, List, Optional, Dict
from pathlib import Path
from loguru import logger
import pandas as pd

from core.actor.scaler.BaseScale import BaseScaler
from core.data_manage import Dataset
from core.utils import parse_schema_from_df, load_dataset
from llama_index.core.llms.llm import LLM


class RSLSQLScaler(BaseScaler):
    """Scaler implementation based on RSL-SQL's SQL generation strategy for producing multiple SQL candidates."""

    NAME = "RSLSQLScaler"

    # RSL-SQL SQL generation template with information augmentation
    FINAL_SQL_TEMPLATE = '''You are a SQL expert following the RSL-SQL methodology. Generate a precise SQL query using enhanced information.

Database Schema:
{schema}

Question: {question}

Evidence: {evidence}

Schema Links (Relevant Tables & Columns): {schema_links}

Sub-questions: {sub_questions}

Information Augmentation Context:
- **Schema Links**: Focus on identified critical tables and columns
- **Evidence Analysis**: Use provided evidence for context and constraints  
- **Sub-question Decomposition**: Break down complex requirements using sub-questions
- **Bidirectional Linking**: Consider both question-to-schema and schema-to-question relationships

RSL-SQL Process:
1. **Information Retrieval**: Use schema links to identify relevant database elements
2. **Information Augmentation**: Enhance understanding with evidence and sub-questions
3. **Query Construction**: Build SQL using augmented information
4. **Bidirectional Validation**: Ensure question requirements match schema capabilities

Instructions:
1. Analyze the question and identify required output columns
2. Use schema links to focus on relevant tables and columns
3. Apply evidence for filtering conditions and constraints
4. Use sub-questions to decompose complex requirements
5. Generate syntactically correct SQL with proper JOINs and conditions
6. Ensure the query addresses all aspects of the question

Generate a SQL query that answers the question. Return the SQL in the following JSON format:
{{"sql": "YOUR_SQL_QUERY_HERE"}}'''

    SIMPLE_TEMPLATE = '''Based on the database schema and question, generate a SQL query.

Database Schema:
{schema}

Question: {question}

Generate a SQL query. Return only the SQL statement:'''

    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Union[LLM, List[LLM]] = None,
            generate_num: int = 8,
            temperature: float = 0.5,
            is_save: bool = True,
            save_dir: Union[str, Path] = "../files/pred_sql",
            open_parallel: bool = True,
            max_workers: int = None,
            use_external: bool = True,
            **kwargs
    ):
        super().__init__(dataset, llm, is_save, save_dir, open_parallel, max_workers, **kwargs)
        self.generate_num = generate_num
        self.temperature = temperature
        self.use_external = use_external

    def _augment_information(self, question: str, schema_links: str, evidence: str, sub_questions: str) -> str:
        """Simulate RSL-SQL's information augmentation process"""
        augmentation_parts = []
        
        if schema_links:
            augmentation_parts.append(f"Schema Links Analysis: {schema_links}")
        
        if evidence:
            augmentation_parts.append(f"Evidence Integration: {evidence}")
            
        if sub_questions:
            augmentation_parts.append(f"Sub-question Decomposition: {sub_questions}")
            
        if augmentation_parts:
            return " | ".join(augmentation_parts)
        
        return ""

    def _extract_sql_from_json(self, response: str) -> Optional[str]:
        """Extract SQL from JSON response"""
        try:
            import json
            # Try to find JSON in response
            json_match = re.search(r'\{.*"sql".*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return data.get('sql', '').strip()
        except:
            pass
        
        # Fallback: look for SQL field
        sql_match = re.search(r'"sql"\s*:\s*"([^"]*)"', response)
        if sql_match:
            return sql_match.group(1).strip()
        
        return None

    def _generate_single_sql(self, llm_: LLM, question: str, schema: str, evidence: str = "", 
                           schema_links: str = "", sub_questions: str = "", use_simple: bool = False) -> Optional[str]:
        """Generate a single SQL candidate using RSL-SQL approach"""
        try:
            if use_simple:
                prompt = self.SIMPLE_TEMPLATE.format(
                    schema=schema,
                    question=question
                )
            else:
                prompt = self.FINAL_SQL_TEMPLATE.format(
                    schema=schema,
                    question=question,
                    evidence=evidence,
                    schema_links=schema_links,
                    sub_questions=sub_questions
                )
                
            response = llm_.complete(prompt, temperature=self.temperature).text
            
            # Try to extract SQL from JSON format first
            sql = self._extract_sql_from_json(response)
            
            if not sql:
                # Fallback: extract any SQL-like content
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
            
            return sql if sql and sql.upper().startswith('SELECT') else None
            
        except Exception as e:
            logger.warning(f"Failed to generate SQL candidate: {e}")
            return None

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        external = load_dataset(external)
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def act(
            self,
            item,
            schema: Union[str, Path, Dict, List] = None,
            schema_links: Union[str, List[str]] = None,
            sub_questions: Union[str, List[str]] = None,
            data_logger=None,
            **kwargs
    ) -> List[str]:
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item}")
        row = self.dataset[item]
        question = row['question']
        evidence = row.get('evidence', '') or kwargs.get('evidence', '') or ''
        if self.use_external:
            external_knowledge = self.load_external_knowledge(row.get("external", None))
            if external_knowledge:
                question += "\n" + external_knowledge
                logger.debug("已加载外部知识")
        
        # Load and process schema using base class method
        schema = self.process_schema(schema, item)
        
        # Process schema_links
        if isinstance(schema_links, list):
            schema_links_str = ', '.join(schema_links)
        elif isinstance(schema_links, dict):
            # Handle dict format from RSL-SQL
            tables = schema_links.get('tables', [])
            columns = schema_links.get('columns', [])
            schema_links_str = f"Tables: {', '.join(tables)}, Columns: {', '.join(columns)}"
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

        # Apply RSL-SQL information augmentation
        augmented_info = self._augment_information(question, schema_links_str, evidence, sub_questions_str)
        
        # Generate multiple SQL candidates
        pred_sqls = []
        
        # Use different generation strategies with information augmentation
        strategies = [
            (False, self.temperature * 0.8),  # Full template with augmentation, lower temp
            (False, self.temperature),        # Full template with augmentation, normal temp
            (False, self.temperature * 1.2),  # Full template with augmentation, higher temp
            (True, self.temperature),         # Simple template
        ]
        
        samples_per_strategy = max(1, self.generate_num // len(strategies))
        
        for use_simple, temp in strategies:
            original_temp = self.temperature
            self.temperature = temp
            
            for _ in range(samples_per_strategy):
                sql = self._generate_single_sql(
                    llm, question, schema, evidence, 
                    schema_links_str, sub_questions_str, use_simple
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

        logger.info(f"RSLSQLScaler: Generated {len(pred_sqls)} SQL candidates for item {item}")
        if data_logger:
            data_logger.info(f"{self.NAME}.candidates | count={len(pred_sqls)}")

        # Save results using base class method
        self.save_results(pred_sqls, item, row.get("instance_id"))

        if data_logger:
            data_logger.info(f"{self.NAME}.pred_sqls | output={pred_sqls}")
            data_logger.info(f"{self.NAME}.act end | item={item}")
        return pred_sqls
