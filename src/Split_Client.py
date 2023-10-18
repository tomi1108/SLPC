import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)

client_socket.connect(server_address)

message = "Hello, Server!"
client_socket.sendall(message.encode('utf-8'))

response = client_socket.recv(1024)
print(f">> Server sent {response}.")

client_socket.close()