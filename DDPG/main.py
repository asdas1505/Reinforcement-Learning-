import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable

from utils import *
from model import *
from DDPGAgent import DDPGAgent



env = NormalizedEnv(gym.make('Pendulum-v0'))
# env.render()

noise = OUNoise(env.action_space)

agent = DDPGAgent(env, hidden_units=256, critic_lr=1e-3, actor_lr=1e-4, gamma=0.99, tau=1e-2, memory_size=50000)

training_episodes = 70
total_rewards = []

batch_size = 256

for episode in range(training_episodes):
    
    print('Training Episode: {} starts'.format(episode+1))
    
    state = env.reset()
    noise.reset()
    episodic_reward = 0
    
    for step in range(500):
        env.render()
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        next_state, reward, done , _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        
        if agent.memory.__len__() > batch_size:
            agent.update(batch_size)
        
        state = next_state
        episodic_reward = episodic_reward + reward
        
        if done:
            break
        
    total_rewards.append(episodic_reward)
    print('Training Episode: {} ends'.format(episode+1))


plt.plot(total_rewards)
plt.xlabel('Training Episodes')
plt.ylabel('Reward')
plt.show()