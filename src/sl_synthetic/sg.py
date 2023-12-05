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
        self.num_train = len(train_dataset)

        print(len(train_dataset))
        print(len(test_dataset))
############################################################################################################
class model(nn.Module):
    def __init__(self, input_dims, num_classes):
        super(model, self).__init__()

        self.input_dims = input_dims
        self.num_classes = num_classes
        self.hidden_size = [128, 64]

        # 学習させるモデルを定義 ##############################################################
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_size[0]),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(self.hidden_size[1], num_classes),
            nn.LogSoftmax(dim=1)
        )
        self.model = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

        # 合成勾配を生成するモデルを定義 #######################################################
        self.sg_layer1 = nn.Sequential(
            nn.Linear(self.hidden_size[0], self.hidden_size[0]*self.input_dims),
        )
        self.sg_layer1_bias = nn.Sequential(
            nn.Linear(self.hidden_size[0], self.hidden_size[0])
        )
        self.sg_layer2 = nn.Sequential(
            nn.Linear(self.hidden_size[1], self.hidden_size[1]*self.hidden_size[0]),
        )
        self.sg_layer2_bias = nn.Sequential(
            nn.Linear(self.hidden_size[1], self.hidden_size[1])
        )
        self.sg_layer3 = nn.Sequential(
            nn.Linear(num_classes, num_classes*self.hidden_size[1]),
        )
        self.sg_layer3_bias = nn.Sequential(
            nn.Linear(num_classes, num_classes)
        )

        self.optimizers = []
        self.forwards = []
        self.grad_optimizers = []
        self.init_optimizers()
        self.init_forwards()

    def init_optimizers(self):
        self.optimizers.append(torch.optim.Adam(self.layer1.parameters(), lr=0.001))
        self.optimizers.append(torch.optim.Adam(self.layer2.parameters(), lr=0.001))
        self.optimizers.append(torch.optim.Adam(self.layer3.parameters(), lr=0.001))
        self.grad_optimizers.append(torch.optim.Adam(self.sg_layer1.parameters(), lr=0.001))
        self.grad_optimizers.append(torch.optim.Adam(self.sg_layer1_bias.parameters(), lr=0.001))
        self.grad_optimizers.append(torch.optim.Adam(self.sg_layer2.parameters(), lr=0.001))
        self.grad_optimizers.append(torch.optim.Adam(self.sg_layer2_bias.parameters(), lr=0.001))
        self.grad_optimizers.append(torch.optim.Adam(self.sg_layer3.parameters(), lr=0.001))
        self.grad_optimizers.append(torch.optim.Adam(self.sg_layer3_bias.parameters(), lr=0.001))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def init_forwards(self):
        self.forwards.append(self.forward_layer1)
        self.forwards.append(self.forward_layer2)
        self.forwards.append(self.forward_layer3)

    def forward_layer1(self, x):
        out = self.layer1(x)
        mean_out = out.mean(dim=0)
        grad = self.sg_layer1(mean_out)
        grad = grad.reshape(self.hidden_size[0], self.input_dims)
        bias = self.sg_layer1_bias(mean_out)
        return out, grad, bias
    
    def forward_layer2(self, x):
        out = self.layer2(x)
        mean_out = out.mean(dim=0)
        grad = self.sg_layer2(mean_out)
        grad = grad.reshape(self.hidden_size[1], self.hidden_size[0])
        bias = self.sg_layer2_bias(mean_out)
        return out, grad, bias
    
    def forward_layer3(self, x):
        out = self.layer3(x)
        mean_out = out.mean(dim=0)
        grad = self.sg_layer3(mean_out)
        grad = grad.reshape(self.num_classes, self.hidden_size[1])
        bias = self.sg_layer3_bias(mean_out)
        return out, grad, bias

############################################################################################################
class classifier():
    def __init__(self, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.input_dims = data.input_dims
        self.num_classes = data.num_classes
        self.num_train = data.num_train

        self.model = model(self.input_dims, self.num_classes)

        self.classificationCriterion = nn.CrossEntropyLoss()
        self.syntheticCriterion = nn.MSELoss()
        # self.synthesisCriterion2 = nn.MSELoss()
        self.num_epochs = 10
    
    def train_model(self):
        for epoch in range(self.num_epochs):
            for i, (x, y) in enumerate(self.train_loader):
                false_grad_list = []
                true_grad_list = []                
                out = x.reshape(-1, self.input_dims)
                
                self.model.optimizer.zero_grad()
                for j in range(len(self.model.grad_optimizers)):
                    self.model.grad_optimizers[j].zero_grad()

                for forward in self.model.forwards:
                    out, grad, bias = forward(out)
                    false_grad_list.append(grad)
                    false_grad_list.append(bias)

                loss = self.classificationCriterion(out, y)
                loss.backward(retain_graph=True)

                for j in range(len(self.model.optimizer.param_groups[0]['params'])):
                    true_grad_list.append(self.model.optimizer.param_groups[0]['params'][j].grad)
                
                for j in range(len(self.model.grad_optimizers)):
                    grad_loss = self.syntheticCriterion(false_grad_list[j], true_grad_list[j])
                    grad_loss.backward(retain_graph=True)
                    self.model.grad_optimizers[j].step()
            
                for j in range(len(self.model.optimizer.param_groups[0]['params'])):
                    self.model.optimizer.param_groups[0]['params'][j].grad = true_grad_list[j].detach()

                self.model.optimizer.step()
            
            perf = self.test_model()
            print("Epoch: {}, Accuracy: {}".format(epoch+1, perf))
    
    def test_model(self):
        correct = 0
        total = 0
        for x, y in self.test_loader:
            out = x.reshape(-1, self.input_dims)
            for forward in self.model.forwards:
                out, grad, bias = forward(out)
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