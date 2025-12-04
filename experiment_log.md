# Experiment_log.md

This my digital experiment log for the Skyfall bandit project. I am also keeping notes in my physical notebook, but this is more for organization of the rrepo etc.

## 2025

### November

#### November 29
Today I narrowed down my first set of experiments to run. I am going to start by running on the farm environment, with alphanumeric and domain-relevant arm names, 3 reward scales, and 3 model sccales.
I will start with thinking models and then move to non-thinking later. While these experiments run over the next few days, I will write my eval code. I will also need to figure out how I'll do a defensible domain-irrelevant sentiment lexicon.

### December 1
I found one thinking trace which was like 140kb. I should be more explicit about how much the model can reason because that's just way too much. Anecdotally, it seems to only happen when the model observes negative reward first. I should also check what temperature I'm using. Maybe as an ablation. Is "changing the temperature" sufficient?

Current plan: 
- run the rest of the runs on 8B, 14B etc. P
- Prepare to test with constrained thinking (maybe just 8B to start to see if it's different)
- Prepare to test with passing CoT in history as well
- These are ABLATIONS - they help me to justify the design decisions and experimental choices.

I am now running the 8B and 14B versions. The focus now is on developing the evaluation code so that I can easily get metrics from the runs instead of having to manually copy everything all the time. I will also set up the ablations so I can justify the design of the larger runs. For the report, I can start to write from the conclusions of these results, but keep running the larger experiments in the background.

### December 4
I spent a while this morning trying to figure out how to get CoT working best for Qwen3. I think what I want to do is this:
- Generate a response with thinking. Max character limit is the thinking budget (say 100 tokens)
- If generation hits the character limit, then append the "out of thinking tokens, must generate an answer" think with thinking off(?)
- Return the final generation - the thinking content should also be returned.