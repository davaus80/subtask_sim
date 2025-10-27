# Custom libraries
from src.agents.agent import Agent
from src.agents.human_agent import HumanAgent
from src.world import World

# System related imports
import logging
import os
from datetime import datetime
import json
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
        return AgentClass.from_config(agent_config)

    def __init__(self, config, run_name=None):
        # Create the world and agent from the config
        self.world = self.create_world_from_config(config)
        self.agent = self.create_agent_from_config(config)
        self.config = config


        ############ LOGGING SETUP #############

        # Configure logger to log to a unique file for each run
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        logs_dir = "logs/experiments"
        os.makedirs(logs_dir, exist_ok=True)

        self.run_name = run_name if run_name else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Make a sub folder for this experiment
        self.experiment_logs_dir = f"logs/experiments/{self.run_name}"
        os.makedirs(self.experiment_logs_dir, exist_ok=True)

        # Generate a unique log file name using timestamp
        log_filename = os.path.join(self.experiment_logs_dir, f"{self.run_name}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)

        # Add formatter to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(file_handler)
        
        # Generate a unique JSONL file for logging outputs
        jsonl_log_filename = os.path.join(self.experiment_logs_dir, f"{self.run_name}.jsonl")
        self.json_logger = jsonlines.open(jsonl_log_filename, mode='w')

        ####### END OF LOGGING SETUP ########


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

            # Get action from agent (until it matches a valid choice)
            # For now, let's just have the world check validity and choose a random action
            # if its invalid. Eventually, it'd be nice to give a few tries.
            action = self.agent.get_action(prompt)


            # Pass action to world (receive reward)
            # For now action is a single string. Once we get to subtasks, it should be a tuple or list of strings
            action_taken, new_state, reward = self.world.take_action(action)

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








