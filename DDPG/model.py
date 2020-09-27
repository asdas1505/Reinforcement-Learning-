import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim



class Critic(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(input_layer, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer)
        self.linear3 = nn.Linear(hidden_layer, output_layer)
        
    def forward(self, state, action):
        """
        Input: State, Action      
        Return: Q value corresponding to the (state, action) pair. 
        """
        
        x = torch.cat([state,action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        x = self.linear3(x)
        
        return x

    
class Actor(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(input_layer, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer)
        self.linear3 = nn.Linear(hidden_layer, output_layer)
    
    def forward(self, state):
        """
        Input: State
        Return: Action 
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        
        return x