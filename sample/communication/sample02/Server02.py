import socket
import time

client_size = 2

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_address = ('localhost', 4444)
server_socket.bind(socket_address)

server_socket.listen(client_size) # 接続可能なクライアント数は１
print("サーバーが接続を待機中...")

connections = []
client_addresses = []

while len(connections) < client_size:
    connection, client_address = server_socket.accept()
    connections.append(connection)
    client_addresses.append(client_address)
    print(f"クライアント {client_address} が接続しました。")

for i in range(client_size):
    connections[i].close()

