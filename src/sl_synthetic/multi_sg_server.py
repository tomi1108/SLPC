from torch import nn, optim
from torch.utils.data import DataLoader
import torch
import socket
import pickle
import zlib
import time

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
    compressed_smashed_data = b""
    connection.send(start_message)
    while True:
        chunk = connection.recv(chunk_size)
        compressed_smashed_data += chunk
        if compressed_smashed_data.endswith(end_message):
            compressed_smashed_data = compressed_smashed_data[:-len(end_message)]
            break
    connection.send(end_message)
    uncompressed_smashed_data = zlib.decompress(compressed_smashed_data)
    smashed_data = pickle.loads(uncompressed_smashed_data)

    # top modelのforward
    output = model.top_forward(smashed_data)

    # top modelのbackward
    loss, grads = model.top_backward(label)

    # send grads
    serialized_grads = pickle.dumps(grads)
    compressed_grads = zlib.compress(serialized_grads) + end_message
    while True:
        receive_message = connection.recv(chunk_size)
        if receive_message == start_message:
            break
    while g_send_progress < len(compressed_grads):
        g_send_progress += chunk_size
        connection.send(compressed_grads[g_send_progress-chunk_size:g_send_progress])
    while True:
        receive_message = connection.recv(chunk_size)
        if receive_message == end_message:
            break
    g_send_progress = 0

    return loss

##########################################################################################################
g_send_progress = 0
num_client = 1
epoch = 5
output_size = 10
chunk_size = 1024

receive_message = b""
start_message = b"START"
end_message = b"END"

hidden_size = [128, 64]
connection_list = []
client_address_list = []
train_label_list = []
test_label_list = []
train_loader_list = []
test_loader_list = []

##########################################################################################################
# クライアントと接続
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 4649)
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
    compressed_train_label = b""
    connection.send(start_message)
    while True:
        chunk = connection.recv(chunk_size)
        compressed_train_label += chunk
        if compressed_train_label.endswith(end_message):
            compressed_train_label = compressed_train_label[:-len(end_message)]
            break
    connection.send(end_message)
    uncompressed_train_label = zlib.decompress(compressed_train_label)
    train_label = pickle.loads(uncompressed_train_label)
    train_label_list.append(train_label)
print('Train label received!')

# receive test label
print('Receiving test label...')
for connection in connection_list:
    compressed_test_label = b""
    connection.send(start_message)
    while True:
        chunk = connection.recv(chunk_size)
        compressed_test_label += chunk
        if compressed_test_label.endswith(end_message):
            compressed_test_label = compressed_test_label[:-len(end_message)]
            break
    connection.send(end_message)
    uncompressed_test_label = zlib.decompress(compressed_test_label)
    test_label = pickle.loads(uncompressed_test_label)
    test_label_list.append(test_label)
print('Test label received!')

# データローダの作成
for train_label in train_label_list:
    train_loader = DataLoader(train_label, batch_size=128, shuffle=False)
    train_loader_list.append(train_loader)

for test_label in test_label_list:
    test_loader = DataLoader(test_label, batch_size=128, shuffle=False)
    test_loader_list.append(test_loader)

##########################################################################################################
# モデルの作成
model = model(hidden_size, output_size)

train_start_time = time.time()
for i in range(epoch):
    # print(f"---Epoch {i+1}/{epoch}---")
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    train_iter_list = []
    test_iter_list = []

    for train_loader in train_loader_list:
        train_iter = iter(train_loader)
        train_iter_list.append(train_iter)
    for test_loader in test_loader_list:
        test_iter = iter(test_loader)
        test_iter_list.append(test_iter)

    for j in range(len(train_loader_list[0])):
        loss = 0.0
        for k in range(num_client):
            loss += learning(model, train_iter_list[k], connection_list[k])
        print(f"loss: {loss/num_client}")
train_end_time = time.time()
print(f"train time: {train_end_time - train_start_time}")

for connection in connection_list:
    connection.close()
##########################################################################################################