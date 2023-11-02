import socket
import threading

client_size = 3
connections = []
client_addresses = []

class ServerThread(threading.Thread):
    def __init__(self):
        super(ServerThread, self).__init__()
    
    def

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_address = ('localhost', 5674)
server_socket.bind(socket_address)

server_socket.listen(client_size)
print("Server is listening...")

while len(connections) < client_size:
    connection, client_address = server_socket.accept()
    connections.append(connection)
    client_addresses.append(client_address)
    print("Connection has been established with: " + client_address[0])

# これはまだ作成途中です。