from typing import Optional, Union, List, Dict, Any

from llama_index.core.llms import LLM

from core.actor.agent.BaseAgent import BaseAgent
from core.actor.base import ActorPool
from core.data_manage import Dataset
from core.actor.agent.WorkflowAgent import MultiWorkflowAgent
from core.actor.selector.BaseSelect import BaseSelector
from loguru import logger


class ForkGatherAgent(BaseAgent):
    def __init__(
            self,
            dataset: Optional[Dataset] = None,
            llm: Optional[LLM] = None,
            select_type: str = "FastExecSelector",
            rnd_n: Optional[int] = None,  # Randomly select the N skills for each actor type.
            **kwargs
    ):
        super().__init__(dataset, llm, **kwargs)
        self.select_type = select_type
        self.rnd_n = rnd_n
        self.check: bool = self._check_select_type()

    def _check_select_type(self, select_type: str = None) -> bool:
        if not select_type:
            select_type = self.select_type

        if BaseSelector.syntax_check(select_type):
            for selector in BaseSelector.get_all_actors():
                if selector.name == select_type:
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
    def _generate_prompt(cls, question="", external="", schema="", size=0, actors=None):
        templates = cls.get_candidate_template()

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
    - The final answer must be a **Python list string**, enclosed exactly as ```list[...]``` inside `<answer>`.

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

    def _fork(self):
        pass

    def _gather(self):
        pass

    def act(self, item, **kwargs):
        # Base on all available actors' skill information, which serve as tools,
        # the base model reasons and rollouts multiple candidate workflows.
        # Executing each workflow and finally gather the final SQL by `selector` actor.
        pass
