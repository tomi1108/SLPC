import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import socket
import pickle
import zlib

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)

client_socket.connect(server_address)

compressed_model = b""
while True:
    chunk = client_socket.recv(1024)
    if not chunk:
        break
    compressed_model += chunk

#下位モデルの解凍・デシリアライズ
uncomporessed_model = zlib.decompress(compressed_model)
bottom_model = pickle.loads(uncomporessed_model)

print(bottom_model)

client_socket.close()