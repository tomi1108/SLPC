import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import socket
import pickle
import zlib

import module.about_id as id

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)

client_socket.connect(server_address)

print("---Receiving compressed model from Server---")
compressed_model = b""
while True:
    chunk = client_socket.recv(1024)
    compressed_model += chunk
    if len(chunk) < 1024:
        break
print(">> Client received compressed model.\n")

message = ">> Client received bottom model.\n"
client_socket.sendall(message.encode('utf-8'))

#下位モデルの解凍・デシリアライズ
uncomporessed_model = zlib.decompress(compressed_model)
bottom_model = pickle.loads(uncomporessed_model)

#データセットの読み込み
load_file = "./../dataset/MNIST/MNIST.pkl"
with open(load_file, 'rb') as f:
    dataset = pickle.load(f)
train_data, train_label, test_data, test_label = [], [], [], []
train_data = dataset['train_img']
train_label = dataset['train_label']
test_data = dataset['test_img']
test_label = dataset['test_label']

#データセットにIDを付与して、順番を入れ替える
train_data, train_label = id.add_ids(train_data, train_label)

print("---Disconnection---")
client_socket.close()