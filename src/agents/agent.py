from abc import ABC
from typing import Dict, Type, Any

class Agent(ABC):
    # Registry maintains the subclasses of Task
    _registry: Dict[str, Type["Agent"]] = {}

    @classmethod
    def _register(cls, key: str):
        def _decorator(subclass: Type["Agent"]):
            cls._registry[key] = subclass
            subclass._task_type = key
            return subclass
        return _decorator
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Agent":
        """
        Default constructor: expects config to have 'params' dict.
        Subclasses can override to implement custom parsing/validation.
        """
        params = config.get("params", {})
        return cls(**params)  
    
    @classmethod
    def get_class(cls, key: str) -> Type["Agent"]:
        return cls._registry[key]