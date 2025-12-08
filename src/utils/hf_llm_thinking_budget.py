from typing import Any, Dict, Optional
import logging
from src.utils.hf_llm import HuggingFaceLLM

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class HFLLM_Thinking_Budget(HuggingFaceLLM):
    """
    Simple wrapper to load a HuggingFace causal LLM from a config dict and
    generate completions for a given prompt.

    Config must contain: config['model']['model_name']
    """

    def __init__(self, config: Dict[str, Any]):
        # prefer config['agent']['model_name'] per project convention
        super().__init__(config)
        if config.get("agent", None) and config['agent'].get("thinking_budget", None):
            self.thinking_budget = config['agent']['thinking_budget']
        else:
            self.thinking_budget = 100.0
        assert self.max_new_tokens > self.thinking_budget, f"thinking budget must be smaller than maximum new tokens. Given {self.max_new_tokens=} and {self.thinking_budget=}"



    def generate(self, prompt: str, **kwargs) -> str:
        messages = [
            {"role": 'system', 'content': 'You are a helpful assistant'},
            {"role": "user", "content": prompt}
        ]

        # First get thinking content
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.thinking_budget
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            if 151667 in output_ids: # Thinking was opened but not closed
                index = len(output_ids)
            else:
                index = 0
            
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        ## If we generated under the budget, then continue as normal
        if len(output_ids) < self.thinking_budget:
            return_dict = {
                "rendered_prompt": text,
                'thinking_content': thinking_content,
                'content': content
            }
            return return_dict

        # If we hit the budget, then we need to generate a response given the prior thinking content
        
        # reasoning content is too long
        prior_content = (
                f"{thinking_content}\n</think>"
                "\n\nThe thinking budget is exhausted, and I must give the solution based on the thinking directly now."
        )
        prior_tokens_len = len(self.tokenizer.encode(prior_content, add_special_tokens=False))
        remaining_tokens = self.max_new_tokens - prior_tokens_len
        assert remaining_tokens > 0, f"remaining tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or lower the thinking_budget."
        
        # 2. append reasoning content to messages and call completion
        messages.append({"role": "assistant", "content": f"{prior_content} <action>"})
        # messages.append({"role": "system", "content": f"/no_think"})
        # messages.append({"role": "user", "content": f"Which action do you select?"})
        # messages.append({"role": "assistant", "content": f"<action>"})
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            enable_thinking=False # I think we disable thinking here since we hit our budget
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=remaining_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        response_data = {
            "thinking_content": prior_content,
            "content": f"<action> {content}",
            'rendered_prompt': text
        }
        return response_data