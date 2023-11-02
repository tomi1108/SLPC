import socket
import time

client_size = 2

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_address = ('localhost', 4444)
server_socket.bind(socket_address)

server_socket.listen(client_size) # 接続可能なクライアント数は１
print("サーバーが接続を待機中...")

