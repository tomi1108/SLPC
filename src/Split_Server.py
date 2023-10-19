import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import socket
import pickle
import zlib
from time import sleep

class TopSL:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def top_forward(self, smashed_data):
        return None
    
    def top_backward(self):
        return None

    def zero_grads(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()   


#基本設定
epochs = 5
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

#オプティマイザの定義
optimizers = [
    optim.SGD(model.parameters(), lr=0.003,)
    for model in models
]

#モデルのシリアライズ・圧縮
serialized_model = pickle.dumps(models[0])
compressed_model = zlib.compress(serialized_model)

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
#上位モデルの定義
top_model = models[1]

#クライアントからラベルを受信
print("---Receiving label from Client---")
compressed_label = b""
while True:
    chunk = connection.recv(1024)
    compressed_label += chunk
    if len(chunk) < 1024:
        break
uncompressed_label = zlib.decompress(compressed_label)
train_label = pickle.loads(uncompressed_label)
print(">> Finished receiving label from Client\n")
print("---Sorting label---")
train_label = sorted(train_label, key=lambda x:x[1])
print(">> Finished sorting label\n")




sleep(5)

print("---Disconnection---")
connection.close()