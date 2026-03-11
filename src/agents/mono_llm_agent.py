from src.agents.agent import Agent
from src.utils.hf_llm import HuggingFaceLLM
from src.utils.hf_llm_thinking_budget import HFLLM_Thinking_Budget
from src.utils.hf_llm_cot import HFLLM_COT
from src.utils.gemini_llm import GeminiLLM
import logging

from typing import Dict, Any
import re



'''
This file defines an LLM agent
'''
@Agent._register("mono_llm")
class MonoLLMAgent(Agent):

    def __init__(self, config):
        model_name = config.get("agent", {}).get("model_name", "")
        if "gemini" in model_name.lower():
            self.llm = GeminiLLM(config)
        elif config.get('agent', None) and config.get('agent', {}).get('thinking_budget', None):
            # For Qwen models, use thinking mode. For Llama, use CoT
            if "qwen3" in model_name.lower():
                self.llm = HFLLM_Thinking_Budget(config)
            elif "llama" in model_name.lower():
                self.llm = HFLLM_COT(config)
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