from src.agents.agent import Agent
from src.utils.hf_llm import HuggingFaceLLM

from typing import Dict, Any
import re

'''
This file defines a human agent
'''
@Agent._register("mono_llm")
class MonoLLMAgent(Agent):

    def __init__(self, config):
        self.llm = HuggingFaceLLM(config)

    def get_action(self, prompt):
        llm_outputs = self.llm.generate(prompt)
        
        ## Here, we add parsing to handle <action> action_name </action> tags
        content = llm_outputs['content']
        
        # extract first match - if I end up wanting all matches, use findall()
        m = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        action = m.group(1).strip() if m else None
        
        return action