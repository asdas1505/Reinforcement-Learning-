import torch
import torch.autograd
import torch.optim as optim
from torch.autograd import Variable
from model import *
from utils import *



class DDPGAgent:

    """
    DDPG: Deep Deterministic Policy Gradient

    Argparse:   

            env: Enviornment Obejct 
            hiddent_units: Number of hidden layers in the actor and critic neural network
            critic_lr: Learning rate of critic optimizer
            actor_lrr: Learning rate of actor optimizer
            gamma: discount factor
            tau: forgetting factor, i.e. by what factor we should forget weights of target networks
            memory_size: max size of memory replay 

    Functions: 

            get_action

    """
    
    def __init__(self, env, hidden_units, critic_lr, actor_lr, gamma, tau, memory_size):
        
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.hidden_units = hidden_units 
        self.gamma = gamma
        self.tau = tau        
        
        self.critic = Critic(self.num_states + self.num_actions, self.hidden_units, self.num_actions)
        self.target_critic = Critic(self.num_states + self.num_actions, self.hidden_units, self.num_actions)
        self.actor = Actor(self.num_states, self.hidden_units, self.num_actions)
        self.target_policy = Actor(self.num_states, self.hidden_units, self.num_actions)
        
        for params, target_params in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_params.data.copy_(params)
        for params, target_params in zip(self.actor.parameters(), self.target_policy.parameters()):
            target_params.data.copy_(params)
        
        self.memory = Memory(memory_size)
        self.criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        

    def get_action(self, state):
        
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        return action
        
        
    def update(self, batch_size):
        
        states, actions, rewards, next_states, done = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        Q_vals = self.critic.forward(states, actions)
        y = rewards + self.gamma*self.target_critic(next_states, self.target_policy.forward(next_states))
        critic_loss = self.criterion(Q_vals, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for params, target_params in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_params.data.copy_(self.tau*params.data + (1-self.tau)*target_params.data)

        for params, target_params in zip(self.actor.parameters(), self.target_policy.parameters()):
            target_params.data.copy_(self.tau*params.data + (1-self.tau)*target_params.data)