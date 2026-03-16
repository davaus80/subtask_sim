import os
import logging
from typing import Any, Dict

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiLLM:
    """
    Wrapper for the Gemini API. Reads GEMINI_API_KEY from the environment.

    Config must contain: config['agent']['model_name']
    Optional: config['agent']['thinking_budget'] (token count passed to ThinkingConfig)
    """

    def __init__(self, config: Dict[str, Any]):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = config["agent"]["model_name"]
        self.thinking_budget = config.get("agent", {}).get("thinking_budget", None)
        self.max_output_tokens = config.get("agent", {}).get("max_new_tokens", None)

        logger.info(f"GeminiLLM initialized with model={self.model_name}, thinking_budget={self.thinking_budget}, max_output_tokens={self.max_output_tokens}")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        thinking_config = None

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=thinking_config,
                max_output_tokens=self.max_output_tokens,
            ),
        )

        thinking_content = ""
        content = ""
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                thinking_content += part.text
            else:
                content += part.text

        return {
            "rendered_prompt": prompt,
            "thinking_content": thinking_content.strip(),
            "content": content.strip(),
        }
