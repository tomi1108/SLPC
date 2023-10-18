import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 5000)

client_socket.connect(server_address)

message = "Hello, Server!"
client_socket.sendall(message.encode('utf-8'))

response = client_socket.recv(1024)
print(f"サーバーから {response} を受信しました。")

client_socket.close()

"""
socket.connect(address)
    addressに示されるリモートソケットに接続する
"""