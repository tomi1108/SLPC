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
import random
import time

############################################################################################################
class mnist():
    def __init__(self):
        self.train_dataset = dsets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_dataset = dsets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

        self.input_dims = 28 * 28
        self.num_classes = 10
        self.num_train = len(self.train_dataset)
        self.num_test = len(self.test_dataset)

        print(len(self.train_dataset))
        print(len(self.test_dataset))
    
    def split_data_label(self, batch_size):
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

        train_ids = np.arange(self.num_train)
        np.random.shuffle(train_ids)
        for i in range(self.num_train):
            self.train_data.append(self.train_dataset[train_ids[i]][0])
            self.train_label.append(self.train_dataset[train_ids[i]][1])

        test_ids = np.arange(self.num_test)
        np.random.shuffle(test_ids)
        for i in range(self.num_test):
            self.test_data.append(self.test_dataset[test_ids[i]][0])
            self.test_label.append(self.test_dataset[test_ids[i]][1])
        
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

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
        self.sg_optimizers = []
        self.sg_optimizers.append(optim.Adam(self.sg_layer1.parameters(), lr=0.001))
        self.sg_optimizers.append(optim.Adam(self.sg_layer2.parameters(), lr=0.001))
        self.sg_criterion = nn.MSELoss()
    
    def bottom_forward(self, data):
        self.optimizer.zero_grad()
        self.smashed_data = self.layer1(data)
        mean_output = self.smashed_data.mean(dim=0)
        grad = self.sg_layer1(mean_output)
        grad = grad.reshape(self.split_layer_size, self.input_dims)
        bias = self.sg_layer2(mean_output)
        return self.smashed_data, grad, bias
    
    def sg_backward(self):
        return 0
    
    def bottom_backward(self, grads):
        self.smashed_data.backward(grads, retain_graph=True)
        self.optimizer.step()

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
serialized_train_label = pickle.dumps(m.train_label)
compressed_train_label = zlib.compress(serialized_train_label) + end_message
while True:
    receive_message = client_socket.recv(chunk_size)
    if receive_message == start_message:
        break
while send_progress < len(compressed_train_label):
    send_progress += chunk_size
    client_socket.send(compressed_train_label[send_progress-chunk_size:send_progress])
while True:
    receive_message = client_socket.recv(chunk_size)
    if receive_message == end_message:
        break
print('Train label sent!')
send_progress = 0

# テストデータの正解ラベルを送信
print('Sending test label...')
serialized_test_label = pickle.dumps(m.test_label)
compressed_test_label = zlib.compress(serialized_test_label) + end_message
while True:
    receive_message = client_socket.recv(chunk_size)
    if receive_message == start_message:
        break
while send_progress < len(compressed_test_label):
    send_progress += chunk_size
    client_socket.send(compressed_test_label[send_progress-chunk_size:send_progress])
while True:
    receive_message = client_socket.recv(chunk_size)
    if receive_message == end_message:
        break
print('Test label sent!')
send_progress = 0

###########################################################################################################
# モデルの定義
model = model(m.input_dims, hidden_size[0])

for i in range(epoch):
    print(f"---Epoch {i+1}/{epoch}---")
    count = 0

    for data in m.train_loader:
        grad_loss = 0
        count += 1
        print(f"---Batch {count}/{len(m.train_loader)}---")
        false_grad_list = []
        true_grad_list = []

        # bottom modelのforward
        data = data.view(-1, m.input_dims)
        smashed_data, sg_grad, sg_bias = model.bottom_forward(data)
        false_grad_list.append(sg_grad)
        false_grad_list.append(sg_bias)

        # send smashed data
        serialized_smashed_data = pickle.dumps(smashed_data)
        compressed_smashed_data = zlib.compress(serialized_smashed_data) + end_message
        while True:
            receive_message = client_socket.recv(chunk_size)
            if receive_message == start_message:
                break
        while send_progress < len(compressed_smashed_data):
            send_progress += chunk_size
            client_socket.send(compressed_smashed_data[send_progress-chunk_size:send_progress])
        while True:
            receive_message = client_socket.recv(chunk_size)
            if receive_message == end_message:
                break
        send_progress = 0

        # sg modelを用いた更新
        for j in range(len(model.optimizer.param_groups[0]['params'])):
            model.optimizer.param_groups[0]['params'][j].grad = false_grad_list[j].detach()
        model.optimizer.step()

        # receive grads
        client_socket.send(start_message)
        compressed_grads = b""
        while True:
            chunk = client_socket.recv(chunk_size)
            compressed_grads += chunk
            if compressed_grads.endswith(end_message):
                compressed_grads = compressed_grads[:-len(end_message)]
                break
        client_socket.send(end_message)
        uncompressed_grads = zlib.decompress(compressed_grads)
        grads = pickle.loads(uncompressed_grads)

        # bottom modelのbackward
        model.bottom_backward(grads)

        # sg modelの更新
        for j in range(len(model.optimizer.param_groups[0]['params'])):
            true_grad_list.append(model.optimizer.param_groups[0]['params'][j].grad)
            grad_loss += model.sg_criterion(false_grad_list[j], true_grad_list[j])
        grad_loss /= len(model.optimizer.param_groups[0]['params'])
        grad_loss.backward()
        model.sg_optimizer.step()
    
    # サーバにモデルを送信
    print('Sending model...')
    layer1_param = model.optimizer.param_groups[0]['params'][0].detach()
    layer1_bias = model.optimizer.param_groups[0]['params'][1].detach()
    serialized_layer1_param = pickle.dumps(layer1_param)
    serialized_layer1_bias = pickle.dumps(layer1_bias)
    compressed_layer1_param = zlib.compress(serialized_layer1_param) + end_message
    compressed_layer1_bias = zlib.compress(serialized_layer1_bias) + end_message
    while True:
        receive_message = client_socket.recv(chunk_size)
        if receive_message == start_message:
            break
    while send_progress < len(compressed_layer1_param):
        send_progress += chunk_size
        client_socket.send(compressed_layer1_param[send_progress-chunk_size:send_progress])
    while True:
        receive_message = client_socket.recv(chunk_size)
        if receive_message == end_message:
            break
    send_progress = 0
    while True:
        receive_message = client_socket.recv(chunk_size)
        if receive_message == start_message:
            break
    while send_progress < len(compressed_layer1_bias):
        send_progress += chunk_size
        client_socket.send(compressed_layer1_bias[send_progress-chunk_size:send_progress])
    while True:
        receive_message = client_socket.recv(chunk_size)
        if receive_message == end_message:
            break
    send_progress = 0

    # サーバからモデルを受信
    print('Receiving model...')
    compressed_layer1_param = b""
    compressed_layer1_bias = b""
    client_socket.send(start_message)
    while True:
        chunk = client_socket.recv(chunk_size)
        compressed_layer1_param += chunk
        if compressed_layer1_param.endswith(end_message):
            compressed_layer1_param = compressed_layer1_param[:-len(end_message)]
            break
    client_socket.send(end_message)
    client_socket.send(start_message)
    while True:
        chunk = client_socket.recv(chunk_size)
        compressed_layer1_bias += chunk
        if compressed_layer1_bias.endswith(end_message):
            compressed_layer1_bias = compressed_layer1_bias[:-len(end_message)]
            break
    client_socket.send(end_message)
    uncompressed_layer1_param = zlib.decompress(compressed_layer1_param)
    uncompressed_layer1_bias = zlib.decompress(compressed_layer1_bias)
    layer1_param = pickle.loads(uncompressed_layer1_param)
    layer1_bias = pickle.loads(uncompressed_layer1_bias)
    model.optimizer.param_groups[0]['params'][0].data = layer1_param
    model.optimizer.param_groups[0]['params'][1].data = layer1_bias


client_socket.close()
############################################################################################################