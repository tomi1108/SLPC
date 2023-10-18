import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_address = ('localhost', 5000)
server_socket.bind(socket_address)

server_socket.listen(1) # 接続可能なクライアント数は１
print("サーバーが接続を待機中...")

connection, client_address = server_socket.accept()
print(f"クライアント {client_address} が接続しました。")

while True:
    data = connection.recv(1024)
    if not data:
        break
    print(f"クライアント {client_address} から {data} を受信しました。")
    connection.sendall("Hello, Client".encode('utf-8'))

connection.close()

"""
socket.socket(family, type, proto, fileno)
    family: アドレスファミリー
        AF_INET: IPv4, (host, port) ペアがアドレス
    type: 基本的にはSOCK_STREAMかSOCK_DGRAM
        SOCK_STREAM: TCP
        SOCK_DGRAM: UDP
    proto: プロトコル番号、基本は省略 or 0

socket.bind(address)
    ソケットにaddressをbindする
    bind済みのソケットに対しては、bind()を呼び出すことはできない

socket.listen(backlog)
    サーバを有効にして、接続を受付可能にする
    backlogは少なくとも0以上でなければならない

socket.accept()
    ソケットがアドレスにbind済みで、listen中である必要がある
    戻り値は (conn, address) のタプル
    connはデータの送受信のための新しいソケットオブジェクト
        通信のためのインスタンスのようなもの？
    addressは接続したソケットのアドレス

socket.recv(bufsize[,flags])
    ソケットからデータを受信し、結果をbytesオブジェクトで返す
    bufsizeで指定した量までしか受信できない

socket.send(bytes[,flags]))
    リモートソケットにデータを送信する
    戻り値として送信したバイト数を返す
    データの一部だけが送信された場合、アプリケーションで残りのデータを再送信する必要がある

socket.sendall(bytes[,flags])
    リモートソケットにデータを送信する
    sendと異なり、全データを送信するか、エラーが発生するまで処理する
    正常終了の場合はNoneを返す

"""