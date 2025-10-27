from src.tasks.task import Task

import random
import numpy as np

@Task.register("bandit")
class BanditTask(Task):
    '''
    We initialize the bandit task assuming the following types
    - actions: List[Dict] 
        Each entry in the list corresponds to an arm. We expect the keys:
            - "description": a str providing context for the arm
            - "mean": float
            - "variance": float
            We assume normal distributions
    - task_prompt: str
        A string which describes the overall task. NOTE: I think this will have to be moved to the outside once the task structure is hidden
    '''
    def __init__(self, actions, task_prompt):
        self.actions = {arm['name']: {k: v for k, v in arm.items() if k != 'name'} for arm in actions}
        self.task_prompt = task_prompt



    @classmethod
    def from_config(cls, config):
        # custom parsing/validation for this subclass
        # We will only get passed the section of the config relative to this subtask
        params = config.get("params", {})
        prompt = params['prompt']
        actions = params['actions']
        
        return cls(actions=actions, task_prompt=prompt)
    
    # There is no initial state in bandits unless it's a contextual bandit
    def initial_state(self):
        return {}
    
    # Take action receives the state and the action, returns reward and new state
    def take_action(self, state, action):
        arm_info = self.actions.get(action, None)

        # Take a random action if arm is invalid
        if not arm_info:
            action = random.choice(list(self.actions.keys()))
            arm_info = self.actions[action]

        reward = np.random.normal(arm_info['mean'], arm_info['std_dev'])

        # State is not modified in standard bandit
        return state, reward
