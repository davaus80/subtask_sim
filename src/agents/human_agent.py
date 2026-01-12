from src.agents.agent import Agent

'''
This file defines a human agent
'''
@Agent._register("human")
class HumanAgent(Agent):

    def __init__(self, config):
        pass

    def get_action(self, prompt):
        print(prompt)
        action = input("Enter your action:")
        return {"action": action, "content": action}