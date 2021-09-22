import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import numpy as np
import random
from collections import namedtuple, deque



"""RL agents Deep Q-network
Implementation adpated from Udacity
"""


class Agent_DQN:
    def __init__(self, state_size = 80, action_size = 5,
                 layer_size_1 = 150, layer_size_2 = 100,
                 model_weights = None):
        self.state_size = state_size
        self.action_size = action_size
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(state_size, layer_size_1),
                        torch.nn.ReLU(),
                        torch.nn.Linear(layer_size_1, layer_size_2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(layer_size_2, action_size)
                        )

        if model_weights is not None:
            self.model.load_state_dict(torch.load(model_weights))

    def q_values(self, state):
        state = torch.from_numpy(state).float()
        return self.model(state).data.numpy()

    def act(self, state):
        q_val = self.q_values(state)
        return np.argmax(q_val)
