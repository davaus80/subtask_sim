# Subtask_Sim
This repository is for my Subtask-based Multi-Agent Systems project.

## Environment Description
We 



## Layout
We have three classes which interact. 

### Simulator
The simulator is the backbone. It is the interface between the agent(s) and the world. It manages experimental details such as logging, results tracking, etc.

## Development Plan
1. Set up initial experiment with one subtask (contextual bandit?) and one monolithic agent
2. Add additional independent subtasks, each with one variable
   - Single agent
   - One agent per task/variable (same in this case)
3. Introduce multiple variables per subtask
   - Single Agent
   - One agent per variable
   - One agent per subtask (oracle)
4. Introduce shared variables between tasks
   - Single Agent
   - One agent per variable
   - One agent per subtask (oracle)
5. Does oracle outperform single agent and one per variable? I would expect so. 
   - The one agent per variable has insufficient information so should fail
   - Monolithic agent should fail to do excess information (is this true?) and too difficult task space
     - Or would it fail due to only one action per agent per turn?
     - Have we answered the question of how actions are performed 