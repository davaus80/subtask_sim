from src.agents.agent import Agent
from src.utils.hf_llm import HuggingFaceLLM

from typing import Dict, Any

'''
This file defines a human agent
'''
@Agent._register("mono_llm")
class MonoLLMAgent(Agent):

    def __init__(self, config):
        self.llm = HuggingFaceLLM(config)

    def get_action(self, prompt):
        llm_outputs = self.llm.generate(prompt)
        
        return llm_outputs['content']