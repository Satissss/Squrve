import ast
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

from llama_index.core.llms import LLM
import re
from core.actor.agent.BaseAgent import BaseAgent
from core.actor.base import ActorPool, MergeStrategy
from core.data_manage import Dataset
from core.actor.agent.WorkflowAgent import MultiWorkflowAgent
from core.actor.selector.BaseSelect import BaseSelector
from loguru import logger
from core.utils import load_dataset


class ForkGatherAgent(BaseAgent):
    """Fork-Gather agent: reasons over candidate workflows, then selects best SQL via selector."""
    NAME = "ForkGatherAgent"

    def __init__(
            self,
            dataset: Dataset = None,
            llm: LLM = None,
            select_type: str = "FastExecSelector",
            use_external: bool = True,
            max_n: Optional[int] = None,
            open_parallel: bool = True,
            rollout_llm_args: Dict[str, Any] = None,
            **kwargs
    ):
        """Initialize ForkGatherAgent.

        Args:
            dataset: Dataset for evaluation.
            llm: LLM for workflow rollout.
            select_type: Selector actor name (e.g. FastExecSelector) to choose final SQL.
            use_external: Whether to load external knowledge.
            max_n: Number of candidate workflows to rollout per sample.
            open_parallel: If True, run workflows in parallel; else sequential.
        """
        super().__init__(dataset, llm, **kwargs)
        self.select_type = select_type
        self.max_n = max_n
        self.use_external = use_external
        self.open_parallel = open_parallel
        self.rollout_llm_args = rollout_llm_args
        self.check: bool = self._check_select_type()

    @classmethod
    def load_external_knowledge(cls, external: Union[str, Path] = None):
        if not external:
            return None
        try:
            external = load_dataset(external)
        except FileNotFoundError:
            logger.debug("External file not found, treat it as content.")
        if external and len(external) > 50:
            external = "####[External Prior Knowledge]:\n" + external
            return external
        return None

    def _check_select_type(self, select_type: str = None) -> bool:
        """Verify that select_type is a valid registered selector."""
        if not select_type:
            select_type = self.select_type

        if BaseSelector.syntax_check(select_type):
            for selector in BaseSelector.get_all_actors():
                if selector.NAME == select_type:
                    return True

        return False

    @classmethod
    def get_candidate_template(cls):
        template_lis = [
            "[generator]",
            "[generator, optimizer]",
            "[[generator, generator, generator], selector]",
            "[generator, [optimizer, optimizer], selector]",
            "[parser, generator]",
            "[parser, [scaler, scaler], optimizer, selector]",
            "[[scaler, scaler, scaler, scaler], selector]",
            "[[generator, generator], [scaler, scaler], selector]",
            "[parser, generator, [scaler, scaler, scaler], optimizer, selector]",
            "[parser, [generator, scaler], [optimizer, optimizer], selector, optimizer]"
        ]
        template_str = ""
        for n, t in enumerate(template_lis):
            template_str += f"# Template {chr(ord('A') + n)}:\n{t}\n\n"

        return template_str

    @classmethod
    def get_available_actor(cls):
        skills = ActorPool.gather_skills()
        if not skills:
            raise Exception("No skills available")

        skills_text = "\n\n---\n\n".join(
            f"## Actor: {skill_str}"
            for skill_name, skill_str in skills.items()
        )

        return skills_text

    @classmethod
    def validate_response_str(cls, response_str: str) -> Tuple[bool, List]:
        """Parse the correct actor list from the LLM rollout response string."""
        try:
            # Step 1: Extract content from <answer>...</answer> tags
            answer_pattern = r'<answer>(.*?)</answer>'
            answer_matches = re.findall(answer_pattern, response_str, re.DOTALL)

            if not answer_matches:
                print("[Error] No <answer> tags found in response")
                return False, []

            # Step 2: Extract ```list...``` format list from answer content
            list_pattern = r'```list\s*(.*?)\s*```'
            list_matches = re.findall(list_pattern, response_str, re.DOTALL)

            if not list_matches:
                print("[Error] No ```list...``` format found in answer")
                return False, []

            # Take the last matched list content
            list_str = list_matches[-1].strip()

            # Step 3: Use ast.literal_eval to safely parse string to Python list
            try:
                parsed_list = ast.literal_eval(list_str)
            except (ValueError, SyntaxError) as e:
                print(f"[Error] Failed to parse list string: {e}")
                return False, []

            # Ensure parsed result is a list
            if not isinstance(parsed_list, list):
                print("[Error] Parsed result is not a list")
                return False, []

            return True, parsed_list

        except Exception as e:
            print(f"[Error] Unexpected error during validation: {e}")
            return False, []

    @classmethod
    def _generate_prompt(cls, question="", external="", schema="", size=0):
        templates = cls.get_candidate_template()
        actors = cls.get_available_actor()
        prompt_template = f"""<|im_start|>system
You are a strategic SQL Planning Agent. Your task is to analyze natural language queries and design an optimal Actor pipeline that produces correct SQL statements.

### Available Actors:
Below is the candidate `Actor` Pool available for this round. Do not select any `Actors` outside this list.

{actors}

### Candidate Templates:
Below are the candidate templates, which serve as slots to be filled with the selected `Actors`.
You are encouraged to select templates from the candidate set; however, you may also use alternatives if they better suit task needs.

{templates}

### Analysis Workflow
1. Template Selection:  
    - Analyze the natural language query and the database schema (complexity, table relationships, question type).  
    - Select the template(s) that have the highest likelihood of success for this **query type**, prioritizing **robustness** and **generalization** capability over simplicity.
    - When evaluating templates, consider:
        - Can this template handle queries with missing information or implicit requirements?
        - Does this template accommodate queries with multiple interpretations or ambiguous intent?
        - Will this template structure support queries with additional constraints or complexity beyond the current example?
        - For this query category (e.g., aggregation, multi-table join, filtering), which template has proven more reliable across diverse instances?
    - Choose templates with explicit disambiguation, validation, or schema analysis steps when the query type is prone to ambiguity.
    - Avoid selecting the simplest template solely to reduce redundancy if a more comprehensive template better handles query variations.

2. Actor Selection:  
    - Based on the selected template and the query characteristics, choose the Actors from the available pool that are best suited to handle each step of the task.  
    - Prioritize Actors that demonstrate strong generalization ability across this category of queries, even if they introduce additional processing steps.
    - Select Actors that are most likely to succeed on edge cases and challenging variations of the query, rather than only the simplest Actors that reduce redundancy for straightforward cases.
    - Consider the specific roles and capabilities of each Actor relative to the query.

3. Pipeline Composition:  
    - Fill the selected template with the chosen Actors, arranging them sequentially or in parallel as required.  
    - Ensure that the final Actor is `pred_sql` to produce the SQL output.


### Output Requirements
1. Reasoning and Format:
    - First, reason step by step to determine the final Actor list.
    - Provide your reasoning within `<think>...</think>`.
    - Provide the final result strictly within `<answer>...</answer>`.
    - The final answer must be a **Python JSON Object**, enclosed exactly as ```list[...]``` inside `<answer>`, such as ```list["LinkAlignGenerator"]```.

2. Actor Legality:
    - Only use Actors from the `Available Actors`; any unlisted Actor is invalid.
    - The final pipeline must output `pred_sql` as the last Actor.

<|im_end|>

<|im_start|>user
# Question:
{question}

# Database Schema (Column Number={size}):
{schema}

# External Knowledge:
{external}

# Output
<think>...</think>
<answer>```list[...]```</answer>
<|im_end|>

<|im_start|>assistant
        """
        prompt = prompt_template

        return prompt

    def _init_rollout_llm(self):
        if not self.rollout_llm_args:
            return self.llm

        from core.llm.OpenaiModel import OpenaiModel
        llm = OpenaiModel(**self.rollout_llm_args)

        return llm

    def _fork(self, item, data_logger=None):
        """Rollout multiple candidate workflows via LLM; each workflow is an actor pipeline."""
        row = self.dataset[item]
        question = row['question']
        db_size = row.get("db_size", "")
        external = ""
        instance_schema_path = row.get("instance_schemas")
        if instance_schema_path:
            schema = load_dataset(instance_schema_path)
            logger.debug(f"Loaded schema from: {instance_schema_path}")
        else:
            logger.debug("Fetching schema from dataset")
            schema = self.dataset.get_db_schema(item)

        if schema is None:
            raise ValueError("Failed to load a valid database schema for the sample!")

        if self.use_external:
            external = self.load_external_knowledge(row.get("external", None))

        prompt = self._generate_prompt(question, external, schema, db_size)
        can_workflows = {}
        rollout_llm = self._init_rollout_llm()
        for ind in range(self.max_n):
            try:
                res = rollout_llm.complete(prompt).text
                flag, actor_lis = self.validate_response_str(res)
                if flag:
                    can_workflows[f"W{ind}"] = {"actor_lis": actor_lis}
                else:
                    logger.info("")
            except Exception as e:
                logger.error(f"Failed to complete query: {e}")

        if data_logger:
            data_logger.info(f"{self.NAME}.can_workflows | count={len(can_workflows)} | workflows={can_workflows}")
        return can_workflows

    def _gather(self, can_workflows: Dict, data_logger=None):
        """Assemble MultiWorkflowAgent with candidate workflows and selector; run and collect final SQL."""
        if self.open_parallel:
            actor_lis = [list(can_workflows.keys()), "select"]
        else:
            actor_lis = list(can_workflows.keys()) + ["select"]

        if data_logger:
            data_logger.info(f"{self.NAME}.select_type | select_type={self.select_type} | actor_lis={actor_lis}")

        actor = MultiWorkflowAgent(
            dataset=self.dataset,
            llm=self.llm,
            workflows=can_workflows,
            actors={"select": {"actor_name": self.select_type}},
            actor_lis=actor_lis,
        )

        return actor

    def act(self, item, data_logger=None, **kwargs):
        """Reason over candidate workflows, execute each, then select best SQL via selector.

        Based on all available actors' skill information (as tools), the base model reasons
        and rollouts multiple candidate workflows. Each workflow is executed; the selector
        actor gathers and selects the final SQL.
        """
        if data_logger:
            data_logger.info(f"{self.NAME}.act start | item={item} | select_type={self.select_type}")

        if not self.check:
            if data_logger:
                data_logger.info(f"{self.NAME}.act end | item={item} | skipped=check_failed")
            return ""

        can_workflows = self._fork(item, data_logger=data_logger)
        if not can_workflows:
            if data_logger:
                data_logger.info(f"{self.NAME}.no_workflows | item={item}")
                data_logger.info(f"{self.NAME}.act end | item={item} | skipped=no_workflows")
            return ""

        agent = self._gather(can_workflows, data_logger=data_logger)
        try:
            res = agent.act(item, data_logger=data_logger, **kwargs)
            if data_logger:
                data_logger.info(f"{self.NAME}.final_sql | sql={res}")
            return res
        except Exception as e:
            logger.exception("MultiWorkflowAgent failed to initialize or execute actors: %s", e)
            self.OUTPUT_NAME = "None"
            if data_logger:
                data_logger.info(f"{self.NAME}.act error | item={item} | error={e}")
            return None
        finally:
            if data_logger:
                data_logger.info(f"{self.NAME}.act end | item={item}")
