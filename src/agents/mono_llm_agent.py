from src.agents.agent import Agent
from src.utils.hf_llm import HuggingFaceLLM
from src.utils.hf_llm_thinking_budget import HFLLM_Thinking_Budget
import logging

from typing import Dict, Any
import re

'''
This file defines a human agent
'''
@Agent._register("mono_llm")
class MonoLLMAgent(Agent):

    def __init__(self, config):
        if config.get('agent', None) and config.get('agent', {}).get('thinking_budget', None):
            self.llm = HFLLM_Thinking_Budget(config)
        else:
            self.llm = HuggingFaceLLM(config)
        self.logger = logging.getLogger(__name__)

    def get_action(self, prompt):
        llm_outputs = self.llm.generate(prompt)
        
        ## Here, we add parsing to handle <action> action_name </action> tags
        content = llm_outputs['content']
        
        # extract first match - if I end up wanting all matches, use findall()
        m = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        action = m.group(1).strip() if m else None
        
        return {"action": action, "content": content}