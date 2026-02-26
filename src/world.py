from src.tasks.task import Task
from jinja2 import Template

import random
import json

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

        # Load in the list of actions and randomly shuffle the order 
        action_list = config['actions']
        random.shuffle(action_list)

        for action in action_list:

            # If the action has a names_path field, then load it in and randomly sample a name until you get one that doesn't
            # match one of the existing actions. 
            names_path = action.get("names_path", None)
            if names_path:
                # Load the list of strings from a JSON file
                with open(names_path, 'r') as f: # TODO: Add exception handling for bad path? I think better to raise the error than fail silently for now
                    string_list = json.load(f)

                while len(string_list) > 0:
                    sampled_name = random.choice(string_list)
                    # Check if sampled_name matches any existing action name
                    if sampled_name not in [a['name'] for a in self.actions.values()]:
                        # Found a unique name
                        action['name'] = sampled_name
                        break
                    else:
                        # Remove the sampled name to avoid re-sampling it
                        string_list.remove(sampled_name)
                else:
                    # If we exit the loop without breaking, no unique name was found
                    raise ValueError("No unique action name available in names_path.")
            

            self.actions[action['id']] = {
                'name': action['name'],
                'subtasks': [] # This is for the list of relevant subtasks
            }

        # Check all names at the end and raise an exception if there are multiple with the same name.
        action_names = [action['name'] for action in self.actions.values()]
        duplicates = set([name for name in action_names if action_names.count(name) > 1])
        if duplicates:
            raise ValueError(f"Duplicate action names found: {duplicates}")

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
        
        # Check if action name is valid
        action_id = self.action_name_to_id_map.get(action_name, None)
        
        # As a fallback, check if the action name CONTAINS any valid action names (e.g. action_name + action_suffix)
        if not action_id:
            matches = [key for key in self.action_name_to_id_map if key.lower() in action_name.lower()]
            if matches:
                # Return the longest matching key (most specific)
                action_name = max(matches, key=len)
                action_id = self.action_name_to_id_map.get(action_name, None)
            
        
        # Get the tasks affected by this action
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
        
        # Process the action history as well for easier history passing
        # Map from action name to list of observed rewards
        per_action_history = {action_name: [] for action_name in self.action_name_to_id_map}
        for hist_entry in history:
            if not hist_entry['action_taken'] in per_action_history:
                per_action_history[hist_entry['action_taken']] = []
            per_action_history[hist_entry['action_taken']].append(hist_entry['reward'])
            
        per_action_summarized_history = {}
        for action_name, action_history in per_action_history.items():
            count = len(action_history)
            obs_mean = sum(action_history) / count if count > 0 else "Unknown"
            summarized_history = {
                'obs_mean': obs_mean,
                'count': count
            }
            
            per_action_summarized_history[action_name] = summarized_history
            

        # Render the template with the current state, actions, and history
        return self.prompt_template.render(
            task_prompt=self.config.get("world", {}).get("task_prompt", ""),
            state=self.state_dict,
            actions=list(self.action_name_to_id_map.keys()),
            action_suffix=self.config.get("world", {}).get("action_suffix", ""),
            history=history,
            num_turns=self.time_horizon,
            current_turn=len(history),
            thinking_budget=self.config.get("agent", {}).get("thinking_budget", "Not specified"),
            new_token_budget=self.config.get("agent", {}).get("max_new_tokens", "Not specified"),
            per_action_history=per_action_history,
            per_action_summarized_history=per_action_summarized_history
        )

    def get_state(self):
        return self.state_dict

    '''
        Return the set of actions as a dictionary indexed by ID with a dictionary as values
        {
            0: {'name': xyz, 'mean': 100, 'std': 10}, 
        }
    '''

    def get_action_statistics(self):
        """
        Copies the list of actions and calculates the mean reward and std for each action
        based on the relevant subtasks.
        Returns:
            action_stats: dict mapping action_id to {'name': str, 'mean': float, 'std': float}
        """
        # Copy the list of actions
        action_stats = {}

        for action_id, action in self.actions.items():
            # Get the relevant subtasks for this action
            relevant_task_ids = action['subtasks']

            # Initialize variables for mean and std calculation
            total_mean = 0.0
            total_std = 0.0
            task_count = 0

            # Iterate over relevant subtasks
            for task_id in relevant_task_ids:
                task = self.subtasks.get(task_id)
                if task:
                    # Assume the task has 'mean' and 'std' attributes
                    task_mean = task.get('mean', 0.0)
                    task_std = task.get('std', 0.0)

                    total_mean += task_mean
                    total_std += task_std
                    task_count += 1

            # Calculate the average mean and std for the action
            if task_count > 0:
                avg_mean = total_mean / task_count
                avg_std = total_std / task_count
            else:
                avg_mean = 0.0
                avg_std = 0.0

            # Store the statistics for this action
            action_stats[action_id] = {
                'name': action['name'],
                'mean': avg_mean,
                'std': avg_std
            }

        return action_stats
