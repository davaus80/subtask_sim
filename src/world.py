from src.tasks.task import Task
from jinja2 import Template

import random

'''
world.py is the overall environment manager for this simulation. It keeps track of the subtasks and the state.
'''

class World():
    '''
    World stores:
    - subtasks: dict maps from subtask_id to Task
    - actions: dict maps from action_id to {'name': str, 'subtasks': List of subtask IDs}
    - action_name_to_id_map: Dict maps from str action_names to action_ids
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
        # actions maps from action ID to the name of the task and a list of subtask IDs
        self.actions = {}

        action_list = config['actions']
        for action in action_list:
            self.actions[action['id']] = {
                'name': action['name'],
                'subtasks': [] # This is for the list of relevant subtasks
            }

        # We create a reverse map from action_names to action_IDs
        self.action_name_to_id_map = {}
        for action_id, action in self.actions.items():
            self.action_name_to_id_map[action['name']] = action_id
        

        self.time_horizon = config.get("experiment", {}).get("time_horizon", 0)

        for subtask_cfg in config.get("subtasks", []):
            # expected subtask_cfg format: {"type": "xyz", "params": {...}}
            task_type = subtask_cfg.get("type")
            if task_type is None:
                raise KeyError("subtask entry missing 'type'")
            
            subtask_id = subtask_cfg['id']
            
            TaskClass = Task.get_class(task_type)
            task = TaskClass.from_config(subtask_cfg)
            self.subtasks[subtask_id] = task

            # merge task initial state into world state
            self.state_dict.update(task.initial_state())

            # Update map from action name to subtasks.
            # import pdb; pdb.set_trace()
            subtask_actions = subtask_cfg.get('params', {}).get('actions',{}) # Expect a list of dicts, each containing "name"
            for action in subtask_actions:
                # If action isn't listed then there's an error in the config
                if not self.actions.get(action['id'], None): 
                    raise ValueError(f"Missing action with ID {action['id']}. Check that actions and subtask actions have same IDs in config.")
                
                # Add subtask_id to the subtasks for this action
                if subtask_id not in self.actions[action['id']]['subtasks']:
                    self.actions[action['id']]['subtasks'].append(subtask_id)

        # Load the Jinja2 template specified in the config
        template_path = config.get("world", {}).get("prompt_template_path")
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
    def take_action(self, action_name):
        # Get the tasks affected by this action
        action_id = self.action_name_to_id_map.get(action_name, None)
        relevant_task_ids = self.actions.get(action_id, {}).get('subtasks', None) # This is a list of strings corresponding to the names of tasks affected by this action

        # Take a random action if action is invalid
        if not relevant_task_ids:
            action_id = random.choice(list(self.actions.keys()))
            action_name = self.actions[action_id]['name']
            relevant_task_ids = self.actions[action_id]['subtasks']

        total_reward = 0.0

        for task_id in relevant_task_ids:
            new_state, reward = self.subtasks[task_id].take_action(self.state_dict, action_id)
            self.state_dict = new_state
            total_reward += reward

        return action_name, action_id, self.state_dict, total_reward

    # Should give relevant variables and actions, plus reward history - switch to kwargs eventually
    def get_prompt(self, history):
        if not self.prompt_template:
            raise ValueError("Prompt template is not loaded.")

        # Render the template with the current state, actions, and history
        return self.prompt_template.render(
            task_prompt=self.config.get("world", {}).get("task_prompt", ""),
            state=self.state_dict,
            actions=list(self.action_name_to_id_map.keys()),
            history=history,
            num_turns=self.time_horizon,
            current_turn=len(history)
        )

    def get_state(self):
        return self.state_dict
