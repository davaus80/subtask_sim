from abc import ABC
from typing import Dict, Type, Any

class Task(ABC):
    # Registry maintains the subclasses of Task
    _registry: Dict[str, Type["Task"]] = {}

    @classmethod
    def _register(cls, key: str):
        def _decorator(subclass: Type["Task"]):
            cls._registry[key] = subclass
            subclass._task_type = key
            return subclass
        return _decorator
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Task":
        """
        Default constructor: expects config to have 'params' dict.
        Subclasses can override to implement custom parsing/validation.
        """
        params = config.get("params", {})
        return cls(**params)  
    
    @classmethod
    def get_class(cls, key: str) -> Type["Task"]:
        return cls._registry[key]

    # Instance API (example)
    def __init__(self, *args, **kwargs):
        pass

    # initial_state provides the relevant variables for the task. Use to compose
    # the shared state in World
    def initial_state(self) -> Dict[str, Any]:
        return {}

    '''
    take_action executes the specified action in the environment, or raises an ArgumentError exception
    if the action is invalid.
    Arguments:
    - action_spec: str
        This should be a valid Pythonic function call with arguments from the list of possible actions
    - state_dict: dict
        The world state
    '''
    def take_action(self, action_spec: str): 
        pass
    
    def get_optimal_action(self):
        pass