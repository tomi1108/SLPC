import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import socket
import pickle
import zlib
import module.about_id as id
from torch.utils.data import DataLoader
import numpy as np
from time import sleep

##########################################################################################
class BottomSL:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def bottom_forward(self, input):
        smashed_data = []
        smashed_data.append(self.model(input))

        self.smashed_data = smashed_data

        return smashed_data
    
    def bottom_backward(self, gradient): #多分入力がサーバからの勾配になるはず
        self.smashed_data[0].backward(gradient)
    
    def zero_grads(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()

chunk_size = 1024
epochs = 1 #ここはサーバ側と同じ値にする

##########################################################################################
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)

client_socket.connect(server_address)

print("---Receiving compressed model from Server---")
compressed_model = b""
while True:
    chunk = client_socket.recv(1024)
    if chunk.endswith(b"END"):
        chunk = chunk[:-3]
        compressed_model += chunk
        break
    compressed_model += chunk

print(">> Finished receiving compressed model from Server\n")

#下位モデルの解凍・デシリアライズ
uncomporessed_model = zlib.decompress(compressed_model)
bottom_model = pickle.loads(uncomporessed_model)

optimizer = optim.SGD(bottom_model.parameters(), lr=0.003,) #オプティマイザーの定義

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





#Serverにラベルを送信する
print("---Sending label to Server---")
serialized_label = pickle.dumps(train_label)
compressed_label = zlib.compress(serialized_label) + b"END"

start = 0
while start < len(compressed_label):
    end = start + chunk_size
    client_socket.sendall(compressed_label[start:end])
    start = end
print(">> Finished sending label to Server\n")

client_socket.close()

##########################################################################################
dataloader = DataLoader(train_data, batch_size=128, shuffle=False)
BottomSL = BottomSL(bottom_model, optimizer)

for i in range(epochs):
    print(f"---Epoch {i+1}/{epochs}---")

    for data, ids in dataloader:
        #サーバとのタイミングを同期
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(server_address)

        #1)下位モデルの勾配を初期化
        BottomSL.zero_grads()

        #2)下位モデルにデータを入力し、出力を圧縮
        smashed_data = BottomSL.bottom_forward(data.float())
        serialized_shamshed_data = pickle.dumps(smashed_data)
        compressed_shamshed_data = zlib.compress(serialized_shamshed_data) + b"END"

        #3)圧縮したデータをサーバに送信
        start = 0
        while start < len(compressed_shamshed_data):
            end = start + chunk_size
            client_socket.sendall(compressed_shamshed_data[start:end])
            start = end
        
        #4)サーバから勾配を受信
        compressed_gradient = b""
        while True:
            chunk = client_socket.recv(1024)
            if chunk.endswith(b"END"):
                chunk = chunk[:-3]
                compressed_gradient += chunk
                break
            compressed_gradient += chunk

        uncompressed_gradient = zlib.decompress(compressed_gradient)
        gradient = pickle.loads(uncompressed_gradient)

        BottomSL.bottom_backward(gradient)
        BottomSL.step()

        client_socket.close()

print("---Disconnection---")