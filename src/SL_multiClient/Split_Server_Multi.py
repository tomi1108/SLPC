from torch import nn, optim
from torch.utils.data import DataLoader
import socket
import threading
import pickle
import zlib
import time

# 基本設定
epochs = 5
input_size = 784
hidden_size = [128, 64]
output_size = 10
client_num = 3
chunk_size = 1024
send_start = 0
connections = []
client_addresses = []
labels = []
dataloaders = []
compressed_label = b""
end_flag = b"END"
train_flag = 0

# モデル
models = [
    nn.Sequential(
        nn.Linear(input_size, hidden_size[0]),
        nn.ReLU(),
        nn.Linear(hidden_size[0], hidden_size[1]),
        nn.ReLU()
    ),
    nn.Sequential(
        nn.Linear(hidden_size[1], output_size),
        nn.LogSoftmax(dim=1)
    ),
]

# 学習プロセスに関するクラス
class TopSL:
    def __init__(self, model, optimizer):
        self.model = model
        self.oprimizer = optimizer

# マルチスレッドに関するクラス
class myThread(threading.Thread):
    def __init__(self, number, dataloader):
        super(myThread, self).__init__()
        self.number = number
        self.dataloader = dataloader
    
    def run(self):
        batch_training(self.number, self.dataloader)

# バッチ学習をするための関数（myThreadに呼び出される）
def batch_training(client_number, dataloader):
    global train_flag
    global client_num
    batch_training_count = 1
    for label, id  in dataloader:
        while (train_flag % client_num) != client_number:
            pass

        print(f"Client {client_number}: {batch_training_count}/{len(dataloader)}")
        batch_training_count += 1
        print(f"train_flag: {train_flag}")
        
        train_flag += 1

############################################################################################################
# クライアントと接続
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 9876)
server_socket.bind(server_address)

server_socket.listen(client_num)
print('Waiting for connection...')

while len(connections) < client_num:
    connection, client_address = server_socket.accept()
    connections.append(connection)
    client_addresses.append(client_address)
    print('Connected to', client_address)
print('All connected!')

# 下位モデルを送信
print('Sending bottom model...')
serialized_bottom_model = pickle.dumps(models[0])
compressed_bottom_model = zlib.compress(serialized_bottom_model) + end_flag

for connection in connections:
    while send_start < len(compressed_bottom_model):
        send_end = send_start + chunk_size
        connection.send(compressed_bottom_model[send_start:send_end])
        send_start = send_end
    send_start = 0
print('Bottom model sent!')

# 上位モデルを定義
top_model = models[1]
optimizer = optim.SGD(top_model.parameters(), lr=0.003)

# クライアントからラベルを受信
print('Receiving labels...')
for connection in connections:
    while True:
        chunk = connection.recv(chunk_size)
        compressed_label += chunk
        if compressed_label.endswith(end_flag):
            compressed_label = compressed_label[:-len(end_flag)]
            break
    labels.append(pickle.loads(zlib.decompress(compressed_label)))
    compressed_label = b""

for i in range(len(labels)):
    labels[i] = sorted(labels[i], key=lambda x: x[1])

for connection in connections:
    connection.close()

############################################################################################################
# 学習前の準備
TopSL = TopSL(top_model, optimizer)

for i in range(client_num):
    dataloaders.append(DataLoader(labels[i], batch_size=128, shuffle=False))

# 学習開始
for i in range(epochs):
    print(f"---Epoch {i+1}/{epochs}---")

    threads = []
    for i in range(client_num):
        threads.append(myThread(i, dataloaders[i]))

    for j in range(client_num):
        threads[j].start()
    for j in range(client_num):
        threads[j].join()
    train_flag = 0


