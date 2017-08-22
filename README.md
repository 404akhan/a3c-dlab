# Learn to skip frames in RL (WIP)

### Description
- Idea of Nako Sung, add auxiliary actions to skip N states, during which perform (N+1) no_op (no operation) actions or random actions or repeat last non random action.  <br />
- In some environments eg Pong, Breakout when the ball is not near agent, agent can afford perform random actions.  <br />
- This auxilliary actions hopefully can speed-up training by faster reward propogation due to skiping unnecessary states. <br />
- Agent is incentivised to skip frames when possible due to less discount on immediate state right after skipped N. <br />

### Results
Experiments on BreakoutDeterministic-v4 noskip, skip234, and skip248. skip234 corresponds to perform no_op 2, 3, 4 times as a single aux action. skip248 for no_op 2, 4, 8 times. <br /> <br />

![plot 1](plots/Figure_1-2.png "noskip-skip234")   |  ![plot 2](plots/Figure_1-3.png "noskip-skip234")
:-------------------------------------------------:|:-------------------------------------------------:
![plot 3](plots/Figure_1.png "noskip-skip234")     |  ![plot 4](plots/Figure_1-1.png "noskip-skip234")
![plot 5](plots/Figure_1-4.png "noskip-skip234")   |  ![plot 6](plots/Figure_1-5.png "noskip-skip234")
![plot 7](plots/Figure_1-6.png "noskip-skip234")   |  ![plot 8](plots/Figure_1-7.png "noskip-skip234")

### Dependencies
Important to have latest gym version (0.9.2) that has Deterministic-v4 environments (v3 has bug with image states)

### Notes
Deterministic-v4 environments skip exactly 4 frames and doesn't do random action repeat with prob 0.25. The main reason why many people cannot duplicate RL results from papers on Gym is - default gym environments sample number of skip frames from {2, 3, 4} and perform random action repeat.

### Run
To test trained a3c+aux action model for Breakout (reward around 400) or Pong (reward around 20)
```sh
$ python3.5 main.py --env-name "BreakoutDeterministic-v4" --testing True --num-skips 3 --load-dir model-a3c-aux/breakout-deadTerm-skip234-nolstm-seedChange.pth
$ python3.5 main.py --env-name "PongDeterministic-v4" --testing True --num-skips 3 --load-dir model-a3c-aux/pong-deadTerm-skip234-nolstm-seedChange.pth
```
To train a new model
```sh
$ python3.5 main.py --env-name "PongDeterministic-v4" --num-processes 16 --model-name pong-model-name --num-skips 3 
```

### Thoughts
- Introducing new actions influence speed of learning by enlarging action space
- In Pong performance is comparatively same for no skip and skip 1, 2, 3 state skipping as auxiliary actions
- For Breakout when added 1, 2, 3 state skipping as auxiliary actions agent reached scores 200, 300, 400 faster, despite that it was only one run nothing can be concluded
- In logs directory for skip234 Pong and Breakout, statistics of performed actions show that around half or more times agent performs no_op action and skip 3 states, which suggests there is a place for state skipping in RL

### Reference
- Base a3c implemenation, https://github.com/ikostrikov/pytorch-a3c

