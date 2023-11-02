import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 4444)

client_socket.connect(server_address)

message = "START\n"
while True:
    data_size = client_socket.send(message.encode('utf-8'))
    print(data_size)

    if data_size != 0:
        break

while True:
    data_size = client_socket.send(message.encode('utf-8'))
    print(data_size)

    if data_size != 0:
        break

client_socket.close()