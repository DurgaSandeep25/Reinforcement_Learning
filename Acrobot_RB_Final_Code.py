##libraries and setup
#gym setup
!pip install gym[classic_control]
!apt update
!apt install xvfb
!pip install pyvirtualdisplay
!pip install gym-notebook-wrapper
!nvidia-smi
!pip install pyglet
import torch
print("GPU Available : ", torch.cuda.is_available())

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import random
import imageio
import warnings
warnings.filterwarnings('ignore')


# Torch related packages
import torch
import torch.nn as nn
from torch.distributions import Categorical
import pyglet
import gnwrapper

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#Environment and Variables
game_name = 'Acrobot-v1'
env = gym.make(game_name)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

#hyperparams
layers_list = [256] # hidden layer size
num_episodes = 1500
max_iters = 20
discount = 0.99 # value of gamma
p_lr = 0.001 # Policy Learning Rate
v_lr = 0.1 # Value Learning Rate




class PolicyAndValueNetwork(nn.Module):
    def __init__(self, input_dim, output_dim = 1, hidden_layer_size = [128], network_type = 'policy', initialization = 'random', activation = 'relu'):
      super().__init__()

      networkLayers = []

      prev_layer_size = input_dim

      for i in range(len(hidden_layer_size)):
        next_layer_size = hidden_layer_size[i]
        networkLayers.append(torch.nn.Linear(prev_layer_size, next_layer_size))

        # By default its random initialization
        if initialization == 'HE':
          print("################# He initialization")
          nn.init.kaiming_normal_(networkLayers[-1].weight, mode='fan_in', nonlinearity='relu')
        elif initialization == 'zero':
          print("################# Zero initialization")
          networkLayers[-1].weight.data.zero_()
          networkLayers[-1].bias.data.zero_()
        else:
          print("################# Random initialization")

        # ReLU - activation
        if activation == 'leaky_relu':
          networkLayers.append(torch.nn.LeakyReLU(negative_slope=0.1))
        else:
          networkLayers.append(torch.nn.ReLU())

        # update the prev_layer_size with next_layer_size
        prev_layer_size = next_layer_size

      # Final Output Layer
      output_layer = torch.nn.Linear(prev_layer_size, output_dim)

      if initialization == 'HE':
        nn.init.normal_(output_layer.weight, mean=0.0, std=0.01)

      networkLayers.append(output_layer)

      # Softmax only For Policy network, not for Value network
      if network_type == 'policy':
        networkLayers.append(torch.nn.Softmax(dim=0))

      # Final Deep Learning Model
      self.model = nn.Sequential(*networkLayers)

    def forward(self, x):
      return self.model(x)

# Deep learning Model training
all_iterations_rewards = []



for i in range(max_iters):
  print("Iteration : ", i)

  # Deep learning network instances
  policyNet =  PolicyAndValueNetwork(input_dim = num_states, output_dim = num_actions, hidden_layer_size = layers_list,
                                     initialization = 'HE', network_type = 'policy', activation = 'leaky_relu').to(device)
  valueNet = PolicyAndValueNetwork(input_dim = num_states, output_dim = 1, hidden_layer_size = layers_list,
                                   initialization = 'HE', network_type = 'value', activation = 'leaky_relu').to(device)

  # set .eval() during inference
  policyNet.train()
  valueNet.train()

  # optimizers
  policyOptim = torch.optim.Adam(policyNet.parameters(), lr=p_lr)
  valueOptim = torch.optim.Adam(valueNet.parameters(), lr=v_lr)

  policy_scheduler = torch.optim.lr_scheduler.StepLR(policyOptim, step_size=25, gamma=0.9)  # Scheduler for Policy Network
  value_scheduler = torch.optim.lr_scheduler.StepLR(valueOptim, step_size=25, gamma=0.9)  # Scheduler for Value Network

  episode_rewards = []
  for i in range(num_episodes):
    if i%50 == 0:
      print("Episode : ", i)

    total_reward = 0
    curr_episode = []
    state = env.reset()
    while True:

      # Selecting action based on probabilities
      probabilities = policyNet(torch.from_numpy(state).float().to(device))
      action = torch.multinomial(probabilities, 1)

      # Baseline
      state_value = valueNet(torch.from_numpy(state).float().to(device))

      # Take the action
      state, reward, done, _ = env.step(action.item())

      # required during the backprop - updating weights based on the cumulative rewards
      curr_episode.append((reward, state_value, torch.log(probabilities[action])))

      total_reward += reward
      if done:
        break

    # save the rewards for training plots
    episode_rewards.append(total_reward)

    # optimize weights - backprop

    # clear the gradients
    policyOptim.zero_grad()
    valueOptim.zero_grad()

    discounted_gt = 0
    count = len(curr_episode)-1
    for ele in curr_episode[::-1]:
      reward = ele[0]
      value = ele[1]
      log_prob = ele[2]

      # Discounted returns
      discounted_gt = (discount*discounted_gt) + reward
      delta = discounted_gt - value

      # cumulative gradients at every time step - will be collected in this loop
      valueNet_loss = -1*delta.item()*value
      valueNet_loss.backward()

      policyNet_loss = -1*delta.item()*(discount**count)*log_prob
      policyNet_loss.backward()

      count -= 1

    policyOptim.step()
    valueOptim.step()

    policy_scheduler.step()  # Step the scheduler for Policy Network
    value_scheduler.step()  # Step the scheduler for Value Network

  all_iterations_rewards.append(episode_rewards)



# Plot the episode rewards
plt.figure(figsize = (12,6))
plt.plot(np.mean(np.array(all_iterations_rewards), axis=0), label ='Model Acrobot Reinforce with Baseline')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()



# Animation - reference from Chatgpt
state = env.reset()

frames = []  # List to store frames

reward = 0
while True:

    probabilities = policyNet(torch.from_numpy(state).float().to(device))
    action = torch.multinomial(probabilities, 1)

    env.render(mode='rgb_array')  # Set mode to 'rgb_array' for frame capture

    # Get the rendered frame from the environment
    frame = env.render(mode='rgb_array')
    frames.append(frame)  # Append rendered frame to the frames list

    state, rew, done, _ = env.step(action.item())
    reward += rew
    if done:
        print(reward)
        break

env.close()

# Saving frames as a GIF using imageio
imageio.mimsave('Acrobot_RB_Output.gif', frames, duration=0.1)  # Adjust duration as needed
