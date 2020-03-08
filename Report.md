# Learning Algorithm
The algorithm to train the agent is a deep Q-learning algorithm as described in this [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)



The hyperparameters used are
*BUFFER_SIZE = int(1e5) # replay buffer size
*BATCH_SIZE = 64 # minibatch size
*GAMMA = 0.99 # discount factor
*TAU = 1e-3 # for soft update of target parameters
*LR = 5e-4 # learning rate
*UPDATE_EVERY = 4 # how often to update the network



The Q-Network does consists of 3 fully connected layers
'''
QNetwork(
  (fc1): Linear(in_features=37, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=4, bias=True)
)
'''


# Plot of Rewards
![Plot of scores vs. episodes](https://github.com/baerwind/p1-Navigation/blob/master/rewards.png)

# Ideas for Future Work
choose a different network architecture
    deeper network
    use dropout
use different learning algorithm
    prioritized experience replay
    Dueling DQN
    Rainbow
