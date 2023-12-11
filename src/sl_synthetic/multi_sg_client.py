import sys
sys.path.append('../')

from torch import nn, optim
import socket
import pickle
import zlib
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import module.send_receive as sr
import torch
import random
import time

############################################################################################################
class mnist():
    def __init__(self):
        self.train_dataset = dsets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
 
        self.input_dims = 28 * 28
        self.num_classes = 10
        self.num_train = len(self.train_dataset)
    
    def split_data_label(self, batch_size):
        self.train_data = []
        self.train_label = []
 
        train_ids = np.arange(self.num_train)
        np.random.shuffle(train_ids)
        for i in range(self.num_train):
            self.train_data.append(self.train_dataset[train_ids[i]][0])
            self.train_label.append(self.train_dataset[train_ids[i]][1])

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=False)
 
############################################################################################################
class model(nn.Module):
    def __init__(self, input_dims, split_layer_size):
        super(model, self).__init__()
        self.input_dims = input_dims
        self.split_layer_size = split_layer_size

        # bottom modelの定義
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dims, self.split_layer_size),
            nn.ReLU()
        )
        # sg modelの定義
        self.sg_layer1 = nn.Sequential(
            nn.Linear(self.split_layer_size, self.split_layer_size*self.input_dims)
        )
        self.sg_layer2 = nn.Sequential(
            nn.Linear(self.split_layer_size, self.split_layer_size)
        )
        self.sg_model = nn.Sequential(
            self.sg_layer1,
            self.sg_layer2
        )

        self.optimizer = optim.Adam(self.layer1.parameters(), lr=0.001)
        self.sg_optimizer = optim.Adam(self.sg_model.parameters(), lr=0.001)
        self.sg_criterion = nn.MSELoss()
    
    def bottom_forward(self, data):
        self.optimizer.zero_grad()
        self.data = data
        self.smashed_data = self.layer1(data)
        if sg_flag == True:
            mean_output = self.smashed_data.mean(dim=0)
            grad = self.sg_layer1(mean_output)
            grad = grad.reshape(self.split_layer_size, self.input_dims)
            bias = self.sg_layer2(mean_output)
            return self.smashed_data, grad, bias
        else:
            return self.smashed_data
    
    def bottom_backward(self, grads):
        print(self.optimizer.param_groups[0]['params'][1].grad)
        self.smashed_data.backward(gradient=grads, retain_graph=True)
        print(self.optimizer.param_groups[0]['params'][1].grad)
        self.optimizer.step()

        time.sleep(5)

    def model_test(self, test_loader):
        self.test_loader = test_loader

        
############################################################################################################
send_progress = 0
epoch = 2
batch_size = 128
chunk_size = 1024

receive_message = b""
start_message = b"START"
end_message = b"END"

hidden_size = [128, 64]

sg_flag = True

############################################################################################################
#サーバと接続
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 4648)
# server_address = ('172.31.113.62', 4649)
client_socket.connect(server_address)

# データの取得
m = mnist()
m.split_data_label(batch_size)

# 学習データの正解ラベルを送信
print('Sending train label...')
sr.send(client_socket, m.train_label)
print('Train label sent!')


###########################################################################################################
# モデルの定義
model = model(m.input_dims, hidden_size[0])

for i in range(epoch):
    print(f"---Epoch {i+1}/{epoch}---")

    for data in m.train_loader:
        grad_loss = 0.0
        false_grad_list = []
        true_grad_list = []

        # bottom modelのforward
        data = data.view(-1, m.input_dims)
        if sg_flag == True:
            smashed_data, sg_grad, sg_bias = model.bottom_forward(data)
            false_grad_list.append(sg_grad)
            false_grad_list.append(sg_bias)
        else:
            smashed_data = model.bottom_forward(data)

        # send smashed data
        sr.send(client_socket, smashed_data)

        # sg modelを用いた更新
        if sg_flag == True:
            for j in range(len(model.optimizer.param_groups[0]['params'])):
                model.optimizer.param_groups[0]['params'][j].grad = false_grad_list[j].detach()
            model.optimizer.step() # ここでパラメータが更新されたかを確認する

        # receive grads
        grads = sr.receive(client_socket)

        # bottom modelのbackward
        model.bottom_backward(grads)

        # sg modelの更新
        if sg_flag == True:
            for j in range(len(model.optimizer.param_groups[0]['params'])):
                true_grad_list.append(model.optimizer.param_groups[0]['params'][j].grad)
                print(false_grad_list[j])
                print(true_grad_list[j])
                loss = model.sg_criterion(false_grad_list[j], true_grad_list[j])
                print(f"loss: {loss}")
                grad_loss += loss
            grad_loss /= len(model.sg_optimizer.param_groups[0]['params'])
            grad_loss.backward()
            model.sg_optimizer.step()
            print(f"grad_loss: {grad_loss}")
    
    # サーバにモデルを送信
    print('Sending model...')
    sr.send(client_socket, model.optimizer.param_groups[0]['params'][0].detach())
    sr.send(client_socket, model.optimizer.param_groups[0]['params'][1].detach())

    # サーバからモデルを受信
    print('Receiving model...')
    model.optimizer.param_groups[0]['params'][0].data = sr.receive(client_socket)
    model.optimizer.param_groups[0]['params'][1].data = sr.receive(client_socket)

client_socket.close()
############################################################################################################