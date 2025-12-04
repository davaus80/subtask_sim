from typing import Any, Dict, Optional
import logging

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """
    Simple wrapper to load a HuggingFace causal LLM from a config dict and
    generate completions for a given prompt.

    Config must contain: config['model']['model_name']
    """

    def __init__(self, config: Dict[str, Any]):
        # prefer config['agent']['model_name'] per project convention
        model_name = config["agent"]["model_name"]
        self.model_name = model_name

        ########## Detect available device: prefer CUDA, then MPS (Apple), else CPU ###########
        cuda_available = torch.cuda.is_available()
        mps_available = False
        try:
            mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        except Exception:
            mps_available = False

        if cuda_available:
            self.torch_device = torch.device("cuda")
            # pipeline expects an int index for CUDA devices
            self.pipeline_device = 0
            logger.info("Using CUDA device for model (device=0)")
        elif mps_available:
            self.torch_device = torch.device("mps")
            # pipeline doesn't accept an MPS integer device; we'll pass model already moved to MPS
            self.pipeline_device = -1
            logger.info("Using Apple MPS device for model")
        else:
            self.torch_device = torch.device("cpu")
            self.pipeline_device = -1
            logger.info("Using CPU for model")
            
        ########################## END OF DEVICE CHECKING #############################

        # Load tokenizer and model to ensure pad/eos tokens are set consistently
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        if config.get("agent", None) and config['agent'].get("max_new_tokens", None):
            self.max_new_tokens = config['agent']['max_new_tokens']
        else:
            self.max_new_tokens = 32768


    def generate(self, prompt: str, **kwargs) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]

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
            max_new_tokens=self.max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
            
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return_dict = {
            "rendered_prompt": text,
            'thinking_content': thinking_content,
            'content': content
        }
        
        return return_dict