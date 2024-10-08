import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, observations_dim, actions_dim):
        super(DQN, self).__init__()
        self.fc_layer1 = nn.Linear(observations_dim, 128)
        self.fc_layer2 = nn.Linear(128, 128)
        self.fc_layer3 = nn.Linear(128, actions_dim)

    def forward(self, observations):
        observations = F.relu_(self.fc_layer1(observations))
        observations = F.relu_(self.fc_layer2(observations))
        return self.fc_layer3(observations)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
