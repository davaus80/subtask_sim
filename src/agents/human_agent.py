from src.agents.agent import Agent

'''
This file defines a human agent
'''
class HumanAgent(Agent):

    def __init__(self):
        pass

    def get_action(self, prompt):
        action = input(prompt)
        return action