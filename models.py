import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """
    Description
    -------------
    According to https://arxiv.org/abs/1509.02971
    Initialize the hidden layers (except the output layer) with Uniform(-1/sqrt(fan_in),
    1/sqrt(fan_in)) Where fan_in is the number of units in the layer

    Parameters
    -------------
    layer : torch.nn.modules.linear.Linear object, the fully connected layer to
            initialize.

    Returns
    -------------
    -1/sqrt(fan_in), 1/sqrt(fan_in)
    """

    fan_in = layer.weight.data.size()[1]  # Not sure if it should be [1] instead.
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size=1, seed=0, fc1_units=16, fc2_units=8):
        """
        Description
        -------------
        Actor constructor.

        Parameters
        -------------
        state_size  : Int, Dimension of each state
        action_size : Int, Dimension of each action
        seed        : Int, Random seed
        fc1_units   : Int, Number of nodes in first hidden layer
        fc2_units   : Int, Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -------------
        Reinitialize the layers
        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.fill_(0)

    def forward(self, state):
        """
        Description
        -------------
        Apply a forward pass on a state with Actor network.
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Critic(nn.Module):
    def __init__(self, state_size, action_size=1, seed=0, fcs1_units=64, fc2_units=32):
        """
        Description
        -------------
        Critic constructor

        Parameters
        -------------
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fcs1_units (int): Number of nodes in the first hidden layer
        fc2_units (int): Number of nodes in the second hidden layer
        """

        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_size + action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -------------
        Reinitialize the layers
        """

        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fcs1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.fill_(0)

    def forward(self, state, action):
        """
        Description
        -------------
        Apply a forward pass on a state with Critic network.
        """

        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fcs1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
