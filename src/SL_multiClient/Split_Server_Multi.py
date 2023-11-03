"""
変数名のルール
    - 最初にtrainが付く -> batch_training()で使う変数
    - グローバル変数の場合は最初にg_が付く
"""

from torch import nn, optim
from torch.utils.data import DataLoader
import socket
import threading
import pickle
import zlib
import time

# 基本設定
epochs = 2
input_size = 784
hidden_size = [128, 64]
output_size = 10
send_start = 0
connections = []
client_addresses = []
labels = []
dataloaders = []
compressed_label = b""
g_end_flag = b"END"
g_chunk_size = 1024
g_client_num = 3
g_train_flag = 0

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
        self.optimizer = optimizer

    def top_forward(self, smashed_data):
        tensor = smashed_data[0].detach().requires_grad_(True)
        preds = self.model(tensor)
        self.tensor = tensor

        return preds
    
    def top_backward(self):
        grads = self.tensor.grad
        
        return grads

    def zero_grads(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step() 

# マルチスレッドに関するクラス
class myThread(threading.Thread):
    def __init__(self, number, connection, dataloader, TopSL):
        super(myThread, self).__init__()
        self.number = number
        self.connection = connection
        self.dataloader = dataloader
        self.TopSL = TopSL
    
    def run(self):
        batch_training(self.number, self.connection, self.dataloader, self.TopSL)

# バッチ学習をするための関数（myThreadに呼び出される）
def batch_training(client_number, connection, dataloader, TopSL):
    global g_train_flag
    global g_client_num
    global g_chunk_size
    global g_end_flag
    global g_running_loss
    global g_correct_preds
    global g_total_preds

    batch_training_count = 1
    for label, id  in dataloader:

        while (g_train_flag % g_client_num) != client_number:
            pass

        TopSL.zero_grads()

        print(f"Client {client_number}: {batch_training_count}/{len(dataloader)}")
        batch_training_count += 1

        connection.send(b"START")
        train_compressed_smashed_data = b""
        while True:
            chunk = connection.recv(g_chunk_size)
            train_compressed_smashed_data += chunk
            if train_compressed_smashed_data.endswith(g_end_flag):
                train_compressed_smashed_data = train_compressed_smashed_data[:-len(g_end_flag)]
                break
        connection.send(b"END")
        
        train_uncompressed_smashed_data = zlib.decompress(train_compressed_smashed_data)
        train_smashed_data = pickle.loads(train_uncompressed_smashed_data)

        train_preds = TopSL.top_forward(train_smashed_data)
        train_criterion = nn.NLLLoss()
        train_loss = train_criterion(train_preds, label)
        
        train_loss.backward()
        train_grads = TopSL.top_backward()
        TopSL.step()

        train_serialized_grads = pickle.dumps(train_grads)
        train_compressed_grads = zlib.compress(train_serialized_grads) + g_end_flag

        start = b""
        while True:
            start = connection.recv(g_chunk_size)
            if start == b"START":
                break

        train_start = 0
        while train_start < len(train_compressed_grads):
            train_end = train_start + g_chunk_size
            connection.send(train_compressed_grads[train_start:train_end])
            train_start = train_end

        while True:
            end = connection.recv(g_chunk_size)
            if end == b"END":
                break
        
        g_running_loss += train_loss.item()
        g_correct_preds += train_preds.max(1)[1].eq(label).sum().item()
        g_total_preds += train_preds.size(0)

        g_train_flag += 1

############################################################################################################
# クライアントと接続
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 9876)
server_socket.bind(server_address)

server_socket.listen(g_client_num)
print('Waiting for connection...')

while len(connections) < g_client_num:
    connection, client_address = server_socket.accept()
    connections.append(connection)
    client_addresses.append(client_address)
    print('Connected to', client_address)
print('All connected!')

# 下位モデルを送信
print('Sending bottom model...')
serialized_bottom_model = pickle.dumps(models[0])
compressed_bottom_model = zlib.compress(serialized_bottom_model) + g_end_flag

for connection in connections:
    while send_start < len(compressed_bottom_model):
        send_end = send_start + g_chunk_size
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
        chunk = connection.recv(g_chunk_size)
        compressed_label += chunk
        if compressed_label.endswith(g_end_flag):
            compressed_label = compressed_label[:-len(g_end_flag)]
            break
    labels.append(pickle.loads(zlib.decompress(compressed_label)))
    compressed_label = b""

for i in range(len(labels)):
    labels[i] = sorted(labels[i], key=lambda x: x[1])

############################################################################################################
# 学習前の準備
TopSL = TopSL(top_model, optimizer)

for i in range(g_client_num):
    dataloaders.append(DataLoader(labels[i], batch_size=128, shuffle=False))

# 学習開始
for i in range(epochs):
    print(f"---Epoch {i+1}/{epochs}---")
    g_running_loss = 0
    g_correct_preds = 0
    g_total_preds = 0

    threads = []
    for j in range(g_client_num):
        threads.append(myThread(j, connections[j], dataloaders[j], TopSL))

    for j in range(g_client_num):
        threads[j].start()

    for j in range(g_client_num):
        threads[j].join()

    g_train_flag = 0

    print(f"Epoch {i+1} - Training loss: {g_running_loss/g_client_num:.3f} - Training accuracy: {100*g_correct_preds/g_total_preds:.3f}\n")

print("Finished training!")

for connection in connections:
    connection.close()