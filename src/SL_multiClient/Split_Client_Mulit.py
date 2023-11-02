import sys
sys.path.append('../')

from torch import nn, optim
import socket
import pickle
import zlib
import module.about_id as id
from torch.utils.data import DataLoader
import time

# 基本設定
send_start = 0
epochs = 5
chunck_size = 1024
compressed_model = b""
end_flag = b"END"
load_file = "./../../dataset/MNIST/MNIST.pkl"

class BottomSL:
    def __init__(self, model, optimizer):
        self.model = model
        self.oprimizer = optimizer
    
    def bottom_forward(self, input):
        smashed_data = []
        smashed_data.append(self.model(input))
        self.smashed_data = smashed_data

        return smashed_data
    
    def bottom_backward(self, gradient):
        self.smashed_data[0].backward(gradient)
    
    def zero_grad(self):
        self.oprimizer.zero_grad()

    def step(self):
        self.oprimizer.step()

############################################################################################################
# サーバと接続
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 9876)
client_socket.connect(server_address)

# 下位モデルを受信
print('Receiving bottom model...')
while True:
    chunk = client_socket.recv(chunck_size)
    compressed_model += chunk
    if compressed_model.endswith(end_flag):
        compressed_model = compressed_model[:-len(end_flag)]
        break

uncompressed_model = zlib.decompress(compressed_model)
bottom_model = pickle.loads(uncompressed_model)
optimizer = optim.SGD(bottom_model.parameters(), lr=0.003)
print('Bottom model received!')
print(bottom_model)

# データセットの読み込み
with open(load_file, 'rb') as f:
    dataset = pickle.load(f)
train_data, train_label, test_data, test_label = [], [], [], []
train_data = dataset['train_img']
train_label = dataset['train_label']
test_data = dataset['test_img']
test_label = dataset['test_label']

train_data, train_label =  id.add_ids(train_data, train_label)

# サーバにラベルを送信
print('Sending labels...')
serialized_label = pickle.dumps(train_label)
compressed_label = zlib.compress(serialized_label) + end_flag

while send_start < len(compressed_label):
    send_end = send_start + chunck_size
    client_socket.send(compressed_label[send_start:send_end])
    send_start = send_end
send_start = 0
print('Sent labels!')

############################################################################################################
# 学習前の準備
BottomSL = BottomSL(bottom_model, optimizer)
dataloader = DataLoader(train_data, batch_size=128, shuffle=False)

# 学習開始
for i in range(epochs):
    print(f"---Epoch {i+1}/{epochs}---")
    count = 0

    for data, ids in dataloader:
        count += 1
        print(f"Batch [{count}/{len(dataloader)}]")

        BottomSL.zero_grad()

        time.sleep(0.5)
        smashed_data = BottomSL.bottom_forward(data.float())
        serialized_smashed_data = pickle.dumps(smashed_data)
        compressed_smashed_data = zlib.compress(serialized_smashed_data) + end_flag

        while send_start < len(compressed_smashed_data):
            send_end = send_start + chunck_size
            client_socket.send(compressed_smashed_data[send_start:send_end])
            send_start = send_end
        send_start = 0
        
        compressed_gradient = b""
        while True:
            chunk = client_socket.recv(chunck_size)
            compressed_gradient += chunk
            if compressed_gradient.endswith(end_flag):
                compressed_gradient = compressed_gradient[:-len(end_flag)]
                break
        
        uncompressed_gradient = zlib.decompress(compressed_gradient)
        gradient = pickle.loads(uncompressed_gradient)

        BottomSL.bottom_backward(gradient)
        BottomSL.step()

print('Training completed!')
client_socket.close()