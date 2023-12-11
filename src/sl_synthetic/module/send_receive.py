import zlib
import pickle
import socket

def send(connection, data):
    send_progress = 0
    chunk_size = 1024
    start_message = b"START"
    end_message = b"END"
    serialized_data = pickle.dumps(data)
    compressed_data = zlib.compress(serialized_data) + end_message
    while True:
        receive_message = connection.recv(chunk_size)
        if receive_message == start_message:
            break
    while send_progress < len(compressed_data):
        send_progress += chunk_size
        connection.send(compressed_data[send_progress-chunk_size:send_progress])
    while True:
        receive_message = connection.recv(chunk_size)
        if receive_message == end_message:
            break

def receive(connection):
    chunk_size = 1024
    start_message = b"START"
    end_message = b"END"
    compressed_data = b""
    connection.send(start_message)
    while True:
        chunk = connection.recv(chunk_size)
        compressed_data += chunk
        if compressed_data.endswith(end_message):
            compressed_data = compressed_data[:-len(end_message)]
            break
    connection.send(end_message)
    uncompressed_data = zlib.decompress(compressed_data)
    data = pickle.loads(uncompressed_data)
    return data
