from src.tasks.task import Task

import random
import numpy as np

@Task._register("bandit")
class BanditTask(Task):
    '''
    We initialize the bandit task assuming the following types
    - actions: List[Dict] 
        Each entry in the list corresponds to an arm. We expect the keys:
            - "description": a str providing context for the arm
            - "mean": float
            - "variance": float
            We assume normal distributions
    '''
    def __init__(self, actions):
        self.actions = {arm['id']: {k: v for k, v in arm.items() if k != 'id'} for arm in actions}



    @classmethod
    def from_config(cls, config):
        # custom parsing/validation for this subclass
        # We will only get passed the section of the config relative to this subtask
        params = config.get("params", {})
        actions = params['actions']
        
        return cls(actions=actions)
    
    # There is no initial state in bandits unless it's a contextual bandit
    def initial_state(self):
        return {}
    
    # Take action receives the state and the action, returns reward and new state
    # We assume that actions MUST be valid, based on how World is set up
    def take_action(self, state, action):
        arm_info = self.actions.get(action, None)

        # Take a random action if arm is invalid
        if not arm_info:
            action = random.choice(list(self.actions.keys()))
            arm_info = self.actions[action]

        reward = np.random.normal(arm_info['mean'], arm_info['std_dev'])

        # State is not modified in standard bandit
        return state, reward

@Task._register("contextual_bandit")
class ContextualBanditTask(Task):
    '''
    We initialize the bandit task assuming the following types
    - state_vars List[Dict]
        Each entry is a related state variable. 
        - name: the name of the state variable
        - values: the list of possible values of that variable
    - actions: List[Dict] 
        Each entry in the list corresponds to an arm. We expect the keys:
            - "name": the name to give to the arm in the prompt
            - "description": a str providing context for the arm
            - "distn": a dictionary mapping from state_var values (We assume only one state var for now) to mean and std 
                - "mean": float
                - "variance": float
            We assume normal distributions
    '''
    def __init__(self, actions, state_vars):
        self.actions = {arm['id']: {k: v for k, v in arm.items() if k != 'id'} for arm in actions}
        self.state_vars = None # Fix this later
        # This is in progress


    @classmethod
    def from_config(cls, config):
        # custom parsing/validation for this subclass
        # We will only get passed the section of the config relative to this subtask
        params = config.get("params", {})
        actions = params['actions']
        state_vars = params['state_vars']
        
        return cls(actions=actions, state_vars=state_vars)
    
    # There is no initial state in bandits unless it's a contextual bandit
    def initial_state(self):
        return {}
    
    # Take action receives the state and the action, returns reward and new state
    # We assume that actions MUST be valid, based on how World is set up
    def take_action(self, state, action):
        arm_info = self.actions.get(action, None)

        # Take a random action if arm is invalid
        if not arm_info:
            action = random.choice(list(self.actions.keys()))
            arm_info = self.actions[action]

        reward = np.random.normal(arm_info['mean'], arm_info['std_dev'])

        # State is not modified in standard bandit
        return state, reward
