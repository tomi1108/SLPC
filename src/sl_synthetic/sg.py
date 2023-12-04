import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import time


############################################################################################################
class mnist():
    def __init__(self):
        train_dataset = dsets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
        # train_dataset, _ = random_split(train_dataset, [len(train_dataset)//2, len(train_dataset)//2])
        test_dataset = dsets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
        # test_dataset, _ = random_split(test_dataset, [len(test_dataset)//2, len(test_dataset)//2])
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

        self.input_dims = 28*28
        self.num_classes = 10
        self.in_channels = 1
        self.num_train = len(train_dataset)

        print(len(train_dataset))
        print(len(test_dataset))
############################################################################################################
class model(nn.Module):
    def __init__(self, input_dims, num_classes, in_channels):
        super(model, self).__init__()

        hidden_size = [128, 64]

        self.layer1 = nn.Sequential(
            nn.Linear(input_dims, hidden_size[0]),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size[1], num_classes),
            nn.LogSoftmax(dim=1)
        )

        self.model = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

        self.optimizers = []
        self.forwards = []
        self.init_optimizers()
        self.init_forwards()

    def init_optimizers(self):
        self.optimizers.append(torch.optim.Adam(self.layer1.parameters(), lr=0.001))
        self.optimizers.append(torch.optim.Adam(self.layer2.parameters(), lr=0.001))
        self.optimizers.append(torch.optim.Adam(self.layer3.parameters(), lr=0.001))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def init_forwards(self):
        self.forwards.append(self.forward_layer1)
        self.forwards.append(self.forward_layer2)
        self.forwards.append(self.forward_layer3)

    def forward_layer1(self, x):
        out = self.layer1(x)
        return out
    
    def forward_layer2(self, x):
        out = self.layer2(x)
        return out
    
    def forward_layer3(self, x):
        out = self.layer3(x)
        return out

############################################################################################################
class classifier():
    def __init__(self, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.input_dims = data.input_dims
        self.num_classes = data.num_classes
        self.in_channels = data.in_channels
        self.num_train = data.num_train

        self.model = model(self.input_dims, self.num_classes, self.in_channels)

        self.classificationCriterion = nn.CrossEntropyLoss()
        self.num_epochs = 10
    
    def train_model(self):
        for epoch in range(self.num_epochs):
            for i, (x, y) in enumerate(self.train_loader):
                out = x.reshape(-1, self.input_dims)
                self.model.optimizer.zero_grad()
                for forward in self.model.forwards:
                    out = forward(out)
                loss = self.classificationCriterion(out, y)
                loss.backward()
                self.model.optimizer.step()
            
            perf = self.test_model()
            print("Epoch: {}, Accuracy: {}".format(epoch+1, perf))
    
    def test_model(self):
        correct = 0
        total = 0
        for x, y in self.test_loader:
            out = x.reshape(-1, self.input_dims)
            for forward in self.model.forwards:
                out = forward(out)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        perf = 100 * correct / total
        return perf
############################################################################################################
data = mnist()
m = classifier(data)
m.train_model()
############################################################################################################