# Custom libraries
from src.agents.agent import Agent
from src.world import World

# System related imports
import logging
import os
import datetime

'''
Driver runs the experiments. It contains the World and the Agent
'''

class GameDriver:
    # Initialize agent and world
    def __init__(self, agent, world, config, run_name=None):
        self.agent = agent
        self.world = world
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

        ####### END OF LOGGING SETUP ########


    def play(self):
        # This will be a list of dicts. Each list element corresponds to a time step.
        # Each dict will contain "state", "action", "reward"
        logging_structure = []

        # Main game loop
        for turn_num in range (0, self.world.time_horizon):
            # Get state info from world - this is for the recording (agent doesn't see it)
            state = self.world.get_state()

            # Get prompt from world
            prompt = self.world.get_prompt()

            # Get action from agent (until it matches a valid choice)
            # For now, let's just have the world check validity and choose a random action
            # if its invalid. Eventually, it'd be nice to give a few tries.
            action = self.agent.get_action(prompt)


            # Pass action to world (receive reward)
            # For now action is a single string. Once we get to subtasks, it should be a tuple or list of strings
            reward = self.world.take_action(action)

            # Log relevant info (state, action, reward)
            logging_dict = {
                "state": state,
                "action": action,
                "reward": reward
            }




