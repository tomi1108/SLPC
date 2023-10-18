import torch
import torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.trasnforms import ToTensor

epochs = 5
input_size = 784
hidden_sizes = [128, 640]
output_size = 10

models = [
    nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU()
    ),
    nn.Sequential(
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1)
    ),
]

optimizers = [
    optim.SGD(model.parameters(), lr=0.003,)
    for model in models
]

