from typing import Any, Dict, Optional
import logging
from src.utils.hf_llm import HuggingFaceLLM

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

'''
Models I may use:
https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
https://huggingface.co/meta-llama/Llama-2-13b-hf
https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
https://huggingface.co/meta-llama/Llama-3.1-8B
'''

class HFLLM_COT(HuggingFaceLLM):
    """
    Simple wrapper to load a HuggingFace causal LLM from a config dict and
    generate completions for a given prompt. This subclass is designed for 
    Llama 2 and 3. These models don't have "thinking" modes, so we will
    implement our own constrained CoT approach following the same logic.

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

        # Detect whether we should use explicit CoT append (Llama) or rely on tokenizer's thinking mode.
        use_llama_cot = 'llama' in self.model_name.lower()

        if use_llama_cot:
            # Append explicit Chain-of-Thought instructions to the user prompt.
            cot_instructions = (
                "\n\nPlease think step-by-step and show your chain-of-thought wrapped between <think> and </think>."
                " After your reasoning, output ONLY the chosen action wrapped in <action>...</action>."
            )
            user_prompt = prompt + cot_instructions

            text = self.tokenizer.apply_chat_template(
                [messages[0], {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # First pass: produce constrained 'thinking' generation
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=int(self.thinking_budget)
            )
            output = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True).strip()

            # If model already produced an <action>, handle three cases:
            # 1) both <action> and </action> present -> return normally
            # 2) <action> present but </action> missing -> continue generation to complete it
            # 3) no <action> present -> fallthrough to follow-up below
            if "<action>" in output:
                # Case 1: complete action already present
                if "</action>" in output:
                    thinking_part = output.split("<action>", 1)[0]
                    action_inner = output.split("<action>", 1)[1].split("</action>", 1)[0]
                    thinking_content = thinking_part.strip()
                    content = "<action>" + action_inner.strip() + "</action>"
                    return {
                        "rendered_prompt": text,
                        "thinking_content": thinking_content,
                        "content": content
                    }

                # Case 2: <action> started but not closed -> continue generation to finish it
                partial_output = output
                # Instruct the model to finish the open action tag only
                completion_instruction = (
                    partial_output
                    + "\n\nThe previous response started an <action> tag but was cut off."
                    " Please finish the action so it is properly closed with </action> and output only the remaining text needed to complete the </action> tag."
                )

                prior_tokens_len = len(self.tokenizer.encode(completion_instruction, add_special_tokens=False))
                remaining_tokens = int(self.max_new_tokens) - prior_tokens_len
                assert remaining_tokens > 0, (
                    f"remaining tokens must be positive. Given {remaining_tokens=}. "
                    "Increase the max_new_tokens or lower the thinking_budget."
                )

                follow_messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "assistant", "content": completion_instruction}
                ]
                text2 = self.tokenizer.apply_chat_template(
                    follow_messages,
                    tokenize=False,
                    continue_final_message=True,
                    enable_thinking=False
                )
                model_inputs2 = self.tokenizer([text2], return_tensors="pt").to(self.model.device)
                generated_ids2 = self.model.generate(
                    **model_inputs2,
                    max_new_tokens=remaining_tokens
                )
                output2 = self.tokenizer.decode(generated_ids2[0][len(model_inputs2.input_ids[0]):], skip_special_tokens=True).strip()

                combined = partial_output + output2
                if "</action>" in combined:
                    thinking_part = combined.split("<action>", 1)[0]
                    action_inner = combined.split("<action>", 1)[1].split("</action>", 1)[0]
                    thinking_content = thinking_part.strip()
                    content = "<action>" + action_inner.strip() + "</action>"
                else:
                    # Still incomplete: return what we have but ensure we include the <action> prefix
                    thinking_content = combined.split("<action>", 1)[0].strip()
                    # include remainder after <action> as best-effort
                    suffix = combined.split("<action>", 1)[1] if "<action>" in combined else combined
                    content = "<action> " + suffix.strip()

                return {
                    "rendered_prompt": text2,
                    "thinking_content": thinking_content,
                    "content": content.strip()
                }

            # No action found within thinking budget -> force a short follow-up to produce the action
            # Close any open think tag and ask for the action only.
            thinking_content = output
            if "</think>" not in thinking_content:
                thinking_content = thinking_content + "\n</think>"

            prior_content = (
                f"{thinking_content}\n\nThe thinking budget is exhausted. Based on the reasoning above, now provide ONLY the chosen action,"
                " enclosed in <action> tags. Do not include additional explanation."
            )

            prior_tokens_len = len(self.tokenizer.encode(prior_content, add_special_tokens=False))
            remaining_tokens = int(self.max_new_tokens) - prior_tokens_len
            assert remaining_tokens > 0, (
                f"remaining tokens must be positive. Given {remaining_tokens=}. "
                "Increase the max_new_tokens or lower the thinking_budget."
            )

            # Ask the model to output the action now.
            follow_messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "assistant", "content": prior_content + "\n\n<action>"}
            ]
            text2 = self.tokenizer.apply_chat_template(
                follow_messages,
                tokenize=False,
                continue_final_message=True,
                enable_thinking=False
            )
            model_inputs2 = self.tokenizer([text2], return_tensors="pt").to(self.model.device)
            generated_ids2 = self.model.generate(
                **model_inputs2,
                max_new_tokens=remaining_tokens
            )
            output2 = self.tokenizer.decode(generated_ids2[0][len(model_inputs2.input_ids[0]):], skip_special_tokens=True).strip()

            # Ensure the response contains <action>
            if "<action>" not in output2:
                content = "<action> " + output2
            else:
                # Keep everything from the first <action>
                content = output2.split("<action>", 1)[1]
                content = "<action>" + content

            return {
                "rendered_prompt": text2,
                "thinking_content": prior_content,
                "content": content.strip()
            }

        # Fallback: rely on tokenizer's thinking mode (if any)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )
        output = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # Try to split thinking and content heuristically using tags
        thinking_content = ""
        content = output
        if "</think>" in output:
            parts = output.split("</think>", 1)
            thinking_content = parts[0]
            content = parts[1]

        return {
            "rendered_prompt": text,
            "thinking_content": thinking_content.strip(),
            "content": content.strip()
        }