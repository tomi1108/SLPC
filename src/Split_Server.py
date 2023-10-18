import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import socket

epochs = 5
input_size = 784
hidden_sizes = [128, 640]
output_size = 10

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

#クライアントと接続
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)
server_socket.bind(server_address)

server_socket.listen(1)
print(">> Waiting for client connection...")

connection, client_address = server_socket.accept()
print(f">> Client {client_address} connected.")

while True:
    data = connection.recv(1024)
    if not data:
        break
    print(f">> Client {client_address} sent {data}.")
    connection.sendall("Hello, Client".encode('utf-8'))

connection.close()