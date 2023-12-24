!pip install gym[classic_control]
!apt update
!apt install xvfb
!pip install pyvirtualdisplay
!pip install gym-notebook-wrapper
!nvidia-smi
!pip install pyglet
import torch
print(torch.cuda.is_available())


# Import required packages
import gnwrapper
import numpy as np
import pandas as pd
import gym
import random
import imageio
import numpy as np
import pyglet

# Torch related packages
import torch
import torch.nn as nn
from torch.distributions import Categorical


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

import torch
import torch.nn as nn



class QNetwork_func(nn.Module):
    def __init__(self, input_dim, output_dim = 1, hidden_layer_size = [128], initialization = 'random'):
      super().__init__()

      layers = []

      prev_layer_size = input_dim

      for i in range(len(hidden_layer_size)):
        next_layer_size = hidden_layer_size[i]
        layers.append(torch.nn.Linear(prev_layer_size, next_layer_size))

        # By default its random initialization
        if initialization == 'HE':
          print("################# He initialization")
          nn.init.kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity='relu')
        elif initialization == 'zero':
          layers[-1].weight.data.zero_()
          layers[-1].bias.data.zero_()

        # ReLU - activation
        layers.append(torch.nn.ReLU())

        # update the prev_layer_size with next_layer_size
        prev_layer_size = next_layer_size

      # Final Output Layer
      output_layer = torch.nn.Linear(prev_layer_size, output_dim)

      if initialization == 'HE':
        nn.init.normal_(output_layer.weight, mean=0.0, std=0.01)

      layers.append(output_layer)

      # Final Deep Learning Model
      self.model = nn.Sequential(*layers)

    def forward(self, x):
      return self.model(x)


def SGNS(n, epsilon, decay_rate, alpha, alpha_decay, layer_list, num_episodes, discount=0.99, output=True, start = 'random'):

  # Define the network
  qNet = QNetwork_func(input_dim = num_states, output_dim = num_actions, hidden_layer_size = layer_list, initialization=start)
  qOptim = torch.optim.Adam(qNet.parameters(), lr=alpha)
  qNet.train()

  # Run for 1 iteration
  episode_rewards = []
  for epi in range(num_episodes):

    if (epi > 0) and (epi%alpha_decay == 0):
      for param_group in qOptim.param_groups:
          param_group['lr'] = max(0.00005, param_group['lr']/2.0)

      print("Reduced Alpha from to {}".format(param_group['lr']))

    if output and epi%50 == 0:
      print("Episode : ", epi)
      print(sum(episode_rewards[-10:])/10.0)

    epsilon = epsilon*0.99

    # initialization
    state = env.reset()
    action = random.choice([0, 1]) #get_action(qNet, state, epsilon) # e-greedy is implemented

    all_states = [state]
    all_actions = [action]
    all_rewards = [1.0]

    t = -1
    T = float('inf')
    curr_reward = 0
    while True:
      t += 1

      if t < T:
        state, reward, done, _ = env.step(action)
        curr_reward += reward
        all_rewards.append(reward)
        all_states.append(state)

        if done:
          T = t + 1
        else:

          # epsilpon greedy
          random_number = np.random.rand()
          if random_number <= epsilon:
            action = random.choice([0, 1])
          else:
            action = torch.argmax(qNet(torch.from_numpy(state).float().to(device))).item()

          all_actions.append(action)

      tau = t - n + 1

      if tau >= 0:
        G = 0
        for i in range(tau+1, min(tau + n, T)+1):
          G += (discount**(i-tau-1))*all_rewards[i]

        if tau + n < T:
          with torch.no_grad():
            G += (discount**n)*qNet(torch.from_numpy(all_states[tau+n]).float().to(device))[all_actions[tau+n]].item()


        qOptim.zero_grad()
        curr_q_value = qNet(torch.from_numpy(all_states[tau]).float().to(device))[all_actions[tau]]
        qLoss = nn.functional.mse_loss(torch.tensor(G).float().to(device), curr_q_value)
        qLoss.backward()
        qOptim.step()

      if tau == T - 1:
        episode_rewards.append(curr_reward)
        break

  return qNet, episode_rewards


game_name = 'CartPole-v1'
env = gym.make(game_name)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
num_states, num_actions

all_trained_networks = []
iter_rewards = []

for i in range(20):
  print("Iteration : {}".format(i))

  trained_network, rewards = SGNS(n=10, epsilon=1, decay_rate=0.99, alpha=0.0005, alpha_decay = 100, num_episodes=1000,
                                                     layer_list=[128,128], output=False, start = 'HE')
  all_trained_networks.append(trained_network)
  iter_rewards.append(rewards)

df = pd.DataFrame(iter_rewards)

#Score
plt.figure(figsize = (12,6))
plt.plot(np.mean(df.values, axis=0), label ='Model CartPole Semi Gradient n-step SARSA')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()


game_name = 'CartPole-v1'  # Update with your specific game
env = gym.make(game_name)
state = env.reset()

frames = []  # List to store frames

reward = 0
while True:
    q_values = all_trained_networks[0](torch.from_numpy(state).float().to(device))
    action = torch.argmax(q_values).item()

    env.render(mode='rgb_array')  # Set mode to 'rgb_array' for frame capture

    # Get the rendered frame from the environment
    frame = env.render(mode='rgb_array')
    frames.append(frame)  # Append rendered frame to the frames list

    state, rew, done, _ = env.step(action)
    reward += rew
    if done:
        print(reward)
        break

env.close()

# Saving frames as a GIF using imageio
imageio.mimsave('CartPole_SGNS.gif', frames, duration=0.1)  # Adjust duration as needed
