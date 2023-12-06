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
send_progress = 0
num_client = 1
epoch = 5
output_size = 10
chunk_size = 1024

receive_message = b""
start_message = b"START"
end_message = b"END"

hidden_size = [128, 64]
##########################################################################################################
# クライアントと接続
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 4649)
server_socket.bind(server_address)

server_socket.listen(num_client)
print('Waiting for connection...')

connection, client_address = server_socket.accept()
print('Connected!')

# 学習データの正解ラベルを受信
print('Receiving train label...')
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
print('Train label received!')

# テストデータの正解ラベルを受信
print('Receiving test label...')
compressed_test_label = b""
connection.send(start_message)
while True:
    chunk = connection.recv(chunk_size)
    compressed_test_label += chunk
    if compressed_test_label.endswith(end_message):
        compressed_test_label = compressed_test_label[:-len(end_message)]
        break
connection.send(end_message)
umcompressed_test_label = zlib.decompress(compressed_test_label)
test_label = pickle.loads(umcompressed_test_label)
print('Test label received!')

# データローダの作成
train_loader = DataLoader(train_label, batch_size=128, shuffle=False)
test_loader = DataLoader(test_label, batch_size=128, shuffle=False)
##########################################################################################################
# モデルの作成
model = model(hidden_size, output_size)
count = 0

# 学習開始
for i in range(epoch):
    print(f"---Epoch {i+1}/{epoch}---")

    for label in train_loader:
        count += 1
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
        while send_progress < len(compressed_grads):
            send_progress += chunk_size
            connection.send(compressed_grads[send_progress-chunk_size:send_progress])
        while True:
            receive_message = connection.recv(chunk_size)
            if receive_message == end_message:
                break
        send_progress = 0

        # print("Epoch: {}, loss: {}".format(count, loss))
                  
    # correct = 0
    # total = 0
    # for label in test_loader:
    #     compressed_test_smashed_data = b""
    #     connection.send(start_message)
    #     while True:
    #         chunk = connection.recv(chunk_size)
    #         compressed_test_smashed_data += chunk
    #         if compressed_test_smashed_data.endswith(end_message):
    #             compressed_test_smashed_data = compressed_test_smashed_data[:-len(end_message)]
    #             break
    #     connection.send(end_message)
    #     uncompressed_test_smashed_data = zlib.decompress(compressed_test_smashed_data)
    #     test_smashed_data = pickle.loads(uncompressed_test_smashed_data)
    #     output = model.top_model(test_smashed_data)
    #     _, predicted = torch.max(output.data, 1)
    #     total += label.size(0)
    #     correct += (predicted == label).sum()
    # perf = 100 * correct.item() / total
    # print("Epoch: {}, Accuracy: {}".format(i+1, perf))

connection.close()
##########################################################################################################