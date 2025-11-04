# Custom libraries
from src.agents.agent import Agent
from src.agents.human_agent import HumanAgent
from src.world import World

# System related imports
import logging
import os
from datetime import datetime
import json
import yaml
import jsonlines

'''
Driver runs the experiments. It contains the World and the Agent
'''

class GameDriver:
    # Initialize agent and world
    @staticmethod
    def create_world_from_config(config):
        """
        Create a World instance from the configuration.
        """
        return World(config)

    @staticmethod
    def create_agent_from_config(config):
        """
        Create an Agent instance from the configuration using the class method.
        """
        agent_config = config.get("agent", {})
        agent_type = agent_config.get("type", "human")

        # Dynamically get the Agent class and call its from_config method
        from src.agents.agent import Agent
        AgentClass = Agent.get_class(agent_type)  # Assuming Agent has a registry
        return AgentClass.from_config(config)
    
    '''
    This is basically the constructor, but we separate it out since we use it for reset too.
    '''
    def _setup_from_config_path(self, config_path, run_name=None):
        # Load the YAML config file into a dictionary
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        
        self.world = self.create_world_from_config(config)
        self.agent = self.create_agent_from_config(config)
        self.config = config
        self.experiment_dir = os.path.dirname(config_path) # This is where we'll log everything
        
        ############ LOGGING SETUP #############

        # Configure logger to log to a unique file for each run
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        logs_dir = self.experiment_dir
        os.makedirs(logs_dir, exist_ok=True)

        self.run_name = run_name if run_name else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Make a sub folder for this experiment
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Generate a unique log file name using timestamp
        log_filename = os.path.join(self.experiment_dir, f"{self.run_name}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)

        # Add formatter to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(file_handler)
        
        # Generate a unique JSONL file for logging outputs
        jsonl_log_filename = os.path.join(self.experiment_dir, f"{self.run_name}.jsonl")
        self.json_logger = jsonlines.open(jsonl_log_filename, mode='w')

        ####### END OF LOGGING SETUP ########

    def __init__(self, config_path, run_name=None):
        # Create the world and agent from the config
        self._setup_from_config_path(config_path, run_name)


    def play(self):
        # This will be a list of dicts. Each list element corresponds to a time step.
        # Each dict will contain "state", "action", "reward"
        history = []
        total_reward = 0.0

        # Main game loop
        for turn_num in range (0, self.world.time_horizon):
            # Get state info from world - this is for the recording (agent doesn't see it)
            state = self.world.get_state()

            # Get prompt from world
            prompt = self.world.get_prompt(history)
            
            # Get optimal action and reward from the world (this is for calculating expected regret)
            # We assume for now that we can easily calculate optimal action and reward
            # optimal_action, optimal_reward = self.world.get_optimal()

            # Get action from agent (until it matches a valid choice)
            # For now, let's just have the world check validity and choose a random action
            # if its invalid. Eventually, it'd be nice to give a few tries.
            action = self.agent.get_action(prompt)

            # Pass action to world (receive reward)
            # For now action is a single string. Once we get to subtasks, it should be a tuple or list of strings
            action_taken, new_state, reward = self.world.take_action(action)
            
            history_dict = {
                "state": new_state,
                # "action_selected": action, # Let's omit the action selected for now, since it may confuse the model
                "action_taken": action_taken,
                "reward": reward
            }

            # Log relevant info (state, action, reward)
            logging_dict = {
                "state": new_state,
                "action_selected": action,
                "action_taken": action_taken,
                "reward": reward
            }
            self.json_logger.write(logging_dict)
            history.append(logging_dict)

            total_reward += reward

        print("Final Reward:", total_reward)
        
        
    '''
    NOTE: Reset creates an entirely new world BUT just resets the agent. This is to save loading overhead for the LLM. We may need to change this later,
    but for now, we only have one agent anyway. This is kinda a useless method since it just calls a single other method lol 
    TODO: Merge this with setupfromconfig
    '''
    def reset(self, config_path):
        self._setup_from_config_path(config_path, None)







