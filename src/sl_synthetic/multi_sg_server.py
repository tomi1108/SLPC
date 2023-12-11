from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import socket
import pickle
import zlib
import time
import module.send_receive as sr

############################################################################################################
class model(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(model, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # top modelの定義
        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(self.hidden_size[1], self.output_size),
            nn.LogSoftmax(dim=1)
        )
        self.top_model = nn.Sequential(
            self.layer2,
            self.layer3
        )
        self.bottom_model = nn.Sequential(
            nn.Linear(28*28, self.hidden_size[0]),
            nn.ReLU()
        )

        self.optimizer = optim.Adam(self.top_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def top_forward(self, data):
        self.optimizer.zero_grad()
        self.smashed_data = data
        self.output = self.top_model(data)
        return self.output
    
    def top_backward(self, label):
        loss = self.criterion(self.output, label)
        loss.backward()
        self.optimizer.step()
        return loss.item(), self.smashed_data.grad

##########################################################################################################
def learning(model, train_iter, connection):
    global g_send_progress
    label = next(train_iter)

    # receive smashed data
    smashed_data = sr.receive(connection)

    # top modelのforward
    output = model.top_forward(smashed_data)

    # top modelのbackward
    loss, grads = model.top_backward(label)

    # send grads
    sr.send(connection, grads)

    return loss

##########################################################################################################
g_send_progress = 0
num_client = 1
epoch = 2
output_size = 10
chunk_size = 1024

receive_message = b""
start_message = b"START"
end_message = b"END"

hidden_size = [128, 64]
connection_list = []
client_address_list = []
train_label_list = []
train_loader_list = []

##########################################################################################################
# クライアントと接続
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 4648)
# server_address = ('0.0.0.0', 4649)
server_socket.bind(server_address)
server_socket.listen(num_client)
print('Waiting for connection...')

while len(connection_list) < num_client:
    connection, client_address = server_socket.accept()
    connection_list.append(connection)
    client_address_list.append(client_address)
    print("Connected to", client_address)
print('All connected!')

# receive train label
print('Receiving train label...')
for connection in connection_list:
    train_label = sr.receive(connection)
    train_label_list.append(train_label)
print('Train label received!')

# データローダの作成
for train_label in train_label_list:
    train_loader = DataLoader(train_label, batch_size=128, shuffle=False)
    train_loader_list.append(train_loader)

test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

##########################################################################################################
# モデルの作成
model = model(hidden_size, output_size)

train_start_time = time.time()
for i in range(epoch):
    # print(f"---Epoch {i+1}/{epoch}---")
    train_iter_list = []
    test_iter_list = []
    layer1_param = torch.zeros(hidden_size[0], 28*28)
    layer1_bias = torch.zeros(hidden_size[0])

    for train_loader in train_loader_list:
        train_iter = iter(train_loader)
        train_iter_list.append(train_iter)

    for j in range(len(train_loader_list[0])):
        loss = 0.0
        for k in range(num_client):
            loss += learning(model, train_iter_list[k], connection_list[k])
        print(f"loss: {loss/num_client}")
    
    for j in range(num_client):
        # receive layer1 param
        layer1_param += sr.receive(connection_list[j])
        # receive layer1 bias
        layer1_bias += sr.receive(connection_list[j])

    # layer1 paramの平均を取る
    layer1_param /= num_client
    layer1_bias /= num_client

    # layer1 param bias を送信
    for j in range(num_client):
        sr.send(connection_list[j], layer1_param)
        sr.send(connection_list[j], layer1_bias)
    
    model.bottom_model[0].weight.data = layer1_param
    model.bottom_model[0].bias.data = layer1_bias

    # test
    correct = 0
    total = 0
    for data, label in test_loader:
        data = data.view(-1, 28*28)
        test_smashed_data = model.bottom_model(data)
        output = model.top_model(test_smashed_data)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    print(f"Epoch {i+1}/{epoch}: {correct/total*100}")


train_end_time = time.time()
print(f"train time: {train_end_time - train_start_time}%")

for connection in connection_list:
    connection.close()
##########################################################################################################