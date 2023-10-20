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

##########################################################################################
class BottomSL:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def bottom_forward(self, input):
        return None
    
    def bottom_backward(self, gradient): #多分入力がサーバからの勾配になるはず
        return None
    
    def zero_grads(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()

chunk_size = 1024
epochs = 5

##########################################################################################
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
print(">> Finished receiving compressed model from Server\n")

#下位モデルの解凍・デシリアライズ
uncomporessed_model = zlib.decompress(compressed_model)
bottom_model = pickle.loads(uncomporessed_model)

optimizer = optim.SGD(bottom_model.parameters(), lr=0.003,)

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
"""
ここにテストデータセットについても同様の処理を行うような記述を追加する
"""

#Serverにラベルを送信する
print("---Sending label to Server---")
serialized_label = pickle.dumps(train_label)
compressed_label = zlib.compress(serialized_label)

start = 0
while start < len(compressed_label):
    end = start + chunk_size
    client_socket.sendall(compressed_label[start:end])
    start = end
print(">> Finished sending label to Server\n")

##########################################################################################
dataloader = DataLoader(train_data, batch_size=128, shuffle=False)
BottomSL = BottomSL(bottom_model, optimizer)

"""
train(input, gradient, BottomSL)
入力はデータセットのデータ、サーバからの勾配、BottomSLのインスタンス
"""
def train(input, gradient, BottomSL):
    #1) Zero our grads
    BottomSL.zero_grads()

    #2) Make a smashed data
    smashed_data = BottomSL.bottom_forward(input)

    #3) Backward
    BottomSL.bottom_backward(gradient)

    #4) Change the weights
    BottomSL.step()

    return None

for i in range(epochs):

    break

print("---Disconnection---")
client_socket.close()