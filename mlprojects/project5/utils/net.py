# Copyright © 2023 "Bronte" Sihan Li

import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from dataclasses import dataclass


@dataclass
class NNConfig:
    """
    Defines the configuration of the neural network.
    """

    def __init__(
        self,
        num_conv_layers: int = 2,  # number of convolutional layers
        size_conv_filters: int = 5,  # size of convolutional filters
        activation: str = Literal['relu', 'sigmoid', 'tanh'],  # activation function
        dropout: float = 0.5,  # dropout rate
    ):
        self.num_conv_layers = num_conv_layers
        self.size_conv_filters = size_conv_filters
        self.dropout = dropout  # dropout rate
        self.size_pooling = 2  # size of pooling layer filers
        self.activation = activation


class Net(nn.Module):
    """
    Defines the neural network architecture.
    """

    ACTIVATION_FUNCTIONS = {
        'relu': F.relu,
        'sigmoid': F.sigmoid,
        'tanh': F.tanh,
    }

    # Define the different configurations of the neural network with
    # varying convolutional filter sizes
    # and number of convolutional layers
    DEFINED_CONFIGS = {
        'mnist': NNConfig(num_conv_layers=2, size_conv_filters=5, activation='relu'),
        'c1': NNConfig(num_conv_layers=2, size_conv_filters=2, activation='relu'),
        'c2': NNConfig(num_conv_layers=2, size_conv_filters=3, activation='relu'),
        'c3': NNConfig(num_conv_layers=2, size_conv_filters=4, activation='relu'),
        'c4': NNConfig(num_conv_layers=2, size_conv_filters=5, activation='relu'),
        'c5': NNConfig(num_conv_layers=3, size_conv_filters=2, activation='relu'),
        'c6': NNConfig(num_conv_layers=3, size_conv_filters=3, activation='relu'),
        'c7': NNConfig(num_conv_layers=3, size_conv_filters=4, activation='relu'),
        'c8': NNConfig(num_conv_layers=3, size_conv_filters=5, activation='relu'),
    }

    def __init__(
        self,
        config: str = Literal['mnist', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
        l1: int = 320,  # number of nodes in the first fully connected layer
        l2: int = 50,  # number of nodes in the second fully connected layer
    ):
        super(Net, self).__init__()
        self.config = config
        self.nn_config = self.DEFINED_CONFIGS[config]
        self.l1 = l1
        self.l2 = l2
        kernel_size = self.nn_config.size_conv_filters
        if self.config == 'mnist':
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        elif self.nn_config.num_conv_layers == 2:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
            self.conv2_drop = nn.Dropout2d(p=self.nn_config.dropout)
            self.fc1 = nn.LazyLinear(out_features=self.l1)
            self.fc2 = nn.Linear(self.l1, self.l2)
            self.fc3 = nn.Linear(self.l2, 10)

        elif self.nn_config.num_conv_layers == 3:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
            self.conv3 = nn.Conv2d(20, 40, kernel_size=kernel_size)
            self.conv3_drop = nn.Dropout2d(p=self.nn_config.dropout)
            self.fc1 = nn.LazyLinear(out_features=self.l1)
            self.fc2 = nn.Linear(self.l1, self.l2)
            self.fc3 = nn.Linear(self.l2, 10)

    def forward(self, x):
        """
        Defines the forward pass functions of the neural network.
        """
        activation = self.ACTIVATION_FUNCTIONS[self.nn_config.activation]
        if self.config == 'mnist':
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

        elif self.nn_config.num_conv_layers == 2:
            x = activation(F.max_pool2d(self.conv1(x), 2))
            x = activation(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        elif self.nn_config.num_conv_layers == 3:
            x = activation(F.max_pool2d(self.conv1(x), 2))
            x = activation(self.conv2(x))
            x = activation(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))

        x = x.view(x.size(0), -1)
        x = activation(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.nn_config.dropout)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LinearNN(nn.Module):
    """
    Defines the neural network architecture for a linear model.
    """

    def __init__(
        self,
        l1: int = 12,  # number of nodes in the first fully connected layer
        l2: int = 30,  # number of nodes in the second fully connected layer
        l3: int = None,  # number of nodes in the third fully connected layer
    ):
        super(LinearNN, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.fc1 = nn.Linear(self.l1, self.l2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(self.l2, 2)
        if l3 is not None:
            self.l3 = l3
            self.fc2 = nn.Linear(self.l2, self.l3)
            self.fc3 = nn.Linear(self.l3, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        Defines the forward pass functions of the neural network.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        if hasattr(self, 'l3'):
            x = self.activation(x)
            x = self.fc3(x)
        x = self.softmax(x)
        return x


class StockCNN(nn.Module):
    def __init__(self, l1, l2, kernel_size=5) -> None:
        super().__init__()
        self.l1 = l1
        self.l2 = l2

        self.kernel_size = kernel_size

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=self.kernel_size, padding='same')
        self.conv2 = nn.Conv2d(10, 20, kernel_size=self.kernel_size)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=self.kernel_size)
        self.conv3_drop = nn.Dropout2d(p=0.3)

        self.fc1 = nn.LazyLinear(out_features=self.l1)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(self.l1, self.l2)
        # Because we have 5 labels, we need 5 output nodes.
        self.fc3 = nn.Linear(self.l2, 5)
        # Since we are doing multi-label classification, we use sigmoid as the
        # activation function for the last layer.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        activation = nn.ReLU()

        x = activation(F.max_pool2d(self.conv1(x), 2))
        x = activation(self.conv2(x))
        x = activation(self.conv3_drop(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = activation(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.3)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class StockLinearNN(nn.Module):
    """
    Defines the neural network architecture for the linear model
    for the stock dataset.
    """

    def __init__(
        self,
        l2: int = 3,  # number of nodes in the second fully connected layer
        l3: int = 3,  # number of nodes in the third fully connected layer
    ):
        super(StockLinearNN, self).__init__()
        self.l2 = l2
        self.l3 = l3
        self.fc1 = nn.LazyLinear(out_features=self.l2)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(self.l2, self.l3)
        self.fc3 = nn.Linear(self.l3, 5)
        # Since we are doing multi-label classification, we use sigmoid as the
        # activation function for the last layer.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass functions of the neural network.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
