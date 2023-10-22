import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import socket
import pickle
import zlib
from time import sleep
from torch.utils.data import DataLoader
import numpy as np

class TopSL:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def top_forward(self, smashed_data):
        
        tensor = smashed_data[0].detach().requires_grad_(True)
        gradient_tensor = torch.clone(tensor).detach().requires_grad_(True)
        preds = self.model(tensor)

        # for i in range(len(smashed_data)):
        #     tensor1.append(torch.tensor(smashed_data[i]))

        # tensor2.append(tensor1.detach().requires_grad_())
        # preds.append(self.model(smashed_data))

        # self.preds = preds
        # self.tensor = tensor2

        self.gradient_tensor = gradient_tensor

        return preds
    
    def top_backward(self):
        grads = self.gradient_tensor.grad

        return grads

    def zero_grads(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()   


#基本設定
epochs = 1 #ここはクライアント側と同じ値にする
input_size = 784
hidden_sizes = [128, 640]
output_size = 10
client_size = 1

#データを送るためのバッファサイズ
chunk_size = 1024
start = 0

#モデルの定義
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

#モデルのシリアライズ・圧縮
serialized_model = pickle.dumps(models[0])
compressed_model = zlib.compress(serialized_model) + b"END"

#ソケットの定義
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)
server_socket.bind(server_address)

#クライアントと接続
server_socket.listen(client_size)
print("---Waiting for client connection---")

connection, client_address = server_socket.accept()
print(f">> Client {client_address} connected.\n")

#下位モデルの送信
print("---Sending compressed model to Client---")
while start < len(compressed_model):
    end = start + chunk_size
    connection.sendall(compressed_model[start:end])
    start = end
print(">> Finished sending compressed model to Client\n")

top_model = models[1] #上位モデルの定義
optimizer = optim.SGD(top_model.parameters(), lr=0.003,) #オプティマイザーの定義

#クライアントからラベルを受信
print("---Receiving label from Client---")
compressed_label = b""
while True:
    chunk = connection.recv(1024)
    if chunk.endswith(b"END"):
        chunk = chunk[:-3]
        compressed_label += chunk
        break
    compressed_label += chunk

uncompressed_label = zlib.decompress(compressed_label)
train_label = pickle.loads(uncompressed_label)
print(">> Finished receiving label from Client\n")
print("---Sorting label---")
train_label = sorted(train_label, key=lambda x:x[1])
print(">> Finished sorting label\n")

# label_list = []
# for i in range(len(train_label)):
#     label_list.append(train_label[i][0])
# print(label_list)
# label_list = np.array(label_list)
# label_list = torch.tensor(label_list)
# print(label_list)

# train_label = torch.empty(0)
# train_label = torch.cat([train_label, torch.tensor(label_list)])

connection.close()

##########################################################################################
dataloader = DataLoader(train_label, batch_size=128, shuffle=False)
TopSL = TopSL(top_model, optimizer)

#学習開始
for i in range(epochs):
    print(f"---Epoch {i+1}/{epochs}---")
    running_loss = 0
    correct_preds = 0
    total_preds = 0
    count = 0

    for label, ids in dataloader:
        count += 1
        print(f"---Data {count}/{len(dataloader)}---")

        server_socket.listen(client_size)
        connection, client_address = server_socket.accept()

        TopSL.zero_grads()

        compressed_smashed_data = b""
        while True:
            chunk = connection.recv(1024)
            if chunk.endswith(b"END"):
                chunk = chunk[:-3]
                compressed_smashed_data += chunk
                break
            compressed_smashed_data += chunk

        uncompressed_smashed_data = zlib.decompress(compressed_smashed_data)
        smashed_data = pickle.loads(uncompressed_smashed_data)

        #上位モデルの順伝播
        preds = TopSL.top_forward(smashed_data)
        criterion = nn.NLLLoss()
        loss = criterion(preds, label)

        print(loss)

        loss.backward()
        grads = TopSL.top_backward()
        print(grads)
        TopSL.step()

        serialized_grads = pickle.dumps(grads)
        compressed_grads = zlib.compress(serialized_grads) + b"END"

        start = 0
        while start < len(compressed_grads):
            end = start + chunk_size
            connection.sendall(compressed_grads[start:end])
            start = end

        running_loss += loss.item()
        correct_preds += preds.max(1)[1].eq(label).sum().item()
        total_preds += preds.size(0)

        connection.close()
    print(f"Epoch {i} - Training loss: {running_loss/len(dataloader):.3f} - Training accuracy: {100*correct_preds/total_preds:.3f}\n")

print("---Disconnection---")
server_socket.close()