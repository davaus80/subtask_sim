from tasks.task import Task


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
        # Populate state dict and load the subtasks w correct vars
        self.state_dict = {}
        self.subtasks = []
        self.actions = {}
        self.time_horizon = config.get("experiments", {}).get("rounds", 0)

        for subtask_cfg in config.get("subtasks", []):
            # expected subtask_cfg format: {"type": "xyz", "params": {...}}
            task_type = subtask_cfg.get("type")
            if task_type is None:
                raise KeyError("subtask entry missing 'type'")
            
            TaskClass = Task.get_class(task_type)
            task = TaskClass.from_config(subtask_cfg)
            self.subtasks.append(task)

            # merge task initial state into world state
            self.state_dict.update(task.initial_state())

    '''
    Take an action in the environment - if invalid, select a random action
    Returns:
    - action_result: str
        which action was actually performed (i.e. if random)
    - new_state: dict
    - reward: int
    '''
    def take_action(self, action):
        

    # Should give relevant variables and actions, plus reward history
    def get_prompt(self, history):
        pass

    def get_state(self):
        pass
