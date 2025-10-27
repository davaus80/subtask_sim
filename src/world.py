from tasks.task import Task
from jinja2 import Template

import random

'''
world.py is the overall environment manager for this simulation. It keeps track of the subtasks and the state.
'''

class World():
    '''
    World stores:
    - subtasks: list[Task]
    - state: dict[float or int]
    '''

    '''
    config should be a Dict containing experiment details
    '''
    def __init__(self, config):
        # Store the config as an instance variable
        self.config = config

        # Populate state dict and load the subtasks w correct vars

        # State_dict stores the state
        self.state_dict = {}
        # Subtasks maps frrom task name (str) to Task instance
        self.subtasks = {}
        # actions maps from action name (str) to the names of the related tasks
        self.actions = {}

        self.time_horizon = config.get("experiments", {}).get("rounds", 0)

        for subtask_cfg in config.get("subtasks", []):
            # expected subtask_cfg format: {"type": "xyz", "params": {...}}
            task_type = subtask_cfg.get("type")
            if task_type is None:
                raise KeyError("subtask entry missing 'type'")
            
            subtask_name = subtask_cfg['name']
            
            TaskClass = Task.get_class(task_type)
            task = TaskClass.from_config(subtask_cfg)
            self.subtasks[subtask_name] = task

            # merge task initial state into world state
            self.state_dict.update(task.initial_state())

            # Update map from action name to subtasks.
            actions = subtask_cfg['actions'] # Expect a list of dicts, each containing "name"
            for action in actions:
                # If action is already in dict, then append this subtask name to it's list
                if self.actions.get(action, None): 
                    self.actions[action].append(subtask_name)
                else:
                    self.actions[action] = [subtask_name]

        # Load the Jinja2 template specified in the config
        template_path = config.get("llm", {}).get("prompt_template_path")
        if template_path:
            with open(template_path, "r") as file:
                self.prompt_template = Template(file.read())
        else:
            self.prompt_template = None

    '''
    Take an action in the environment - if invalid, select a random action
    Returns:
    - action_result: str
        which action was actually performed (i.e. if random)
    - new_state: dict
    - reward: int
    '''
    def take_action(self, action):
        # Get the tasks affected by this action
        relevant_tasks = self.actions.get(action, None)

        # Take a random action if action is invalid
        if not relevant_tasks:
            action = random.choice(list(self.actions.keys()))
            relevant_tasks = self.actions[action]

        total_reward = 0.0

        for task in relevant_tasks:
            new_state, reward = task.take_action(self.state_dict, action)
            self.state_dict = new_state
            total_reward += reward

        return action, self.state_dict, total_reward

    # Should give relevant variables and actions, plus reward history - switch to kwargs eventually
    def get_prompt(self, history):
        if not self.prompt_template:
            raise ValueError("Prompt template is not loaded.")

        # Render the template with the current state, actions, and history
        return self.prompt_template.render(
            task_prompt=self.config.get("llm", {}).get("task_prompt", ""),
            state=self.state_dict,
            actions=list(self.actions.keys()),
            history=history
        )

    def get_state(self):
        return self.state_dict
