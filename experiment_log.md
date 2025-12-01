# Experiment_log.md

This my digital experiment log for the Skyfall bandit project. I am also keeping notes in my physical notebook, but this is more for organization of the rrepo etc.

## 2025

### November

#### November 29
Today I narrowed down my first set of experiments to run. I am going to start by running on the farm environment, with alphanumeric and domain-relevant arm names, 3 reward scales, and 3 model sccales.
I will start with thinking models and then move to non-thinking later. While these experiments run over the next few days, I will write my eval code. I will also need to figure out how I'll do a defensible domain-irrelevant sentiment lexicon.

### December 1
I found one thinking trace which was like 140kb. I should be more explicit about how much the model can reason because that's just way too much. Anecdotally, it seems to only happen when the model observes negative reward first. I should also check what temperature I'm using. Maybe as an ablation. Is "changing the temperature" sufficient?