import socket
import threading

# Configuración
L_HOST = '0.0.0.0'      # Escuchar a todos (Móvil, VM, etc.)
L_PORT = 9999           # Puerto nuevo para conectar desde fuera (usaremos este)
R_HOST = '127.0.0.1'    # Destino: Tu propio PC
R_PORT = 9220           # Puerto donde vive Noraxon

def handle_client(client_socket):
    # Conectamos con el Noraxon oculto
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.connect((R_HOST, R_PORT))
        
        # Hilos para enviar y recibir simultáneamente
        threading.Thread(target=forward, args=(client_socket, server_socket)).start()
        threading.Thread(target=forward, args=(server_socket, client_socket)).start()
    except Exception as e:
        print(f"Error conectando con Noraxon: {e}")
        client_socket.close()

def forward(source, destination):
    try:
        while True:
            data = source.recv(4096)
            if len(data) == 0: break
            destination.send(data)
    except:
        pass
    finally:
        source.close()
        destination.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((L_HOST, L_PORT))
    server.listen(5)
    print(f"[*] REPETIDOR ACTIVO. Conéctate a la IP de tu PC en el puerto {L_PORT}")
    print(f"[*] Redirigiendo tráfico hacia Noraxon ({R_HOST}:{R_PORT})...")

    while True:
        client_sock, addr = server.accept()
        print(f"[+] Conexión aceptada desde: {addr[0]}")
        client_handler = threading.Thread(target=handle_client, args=(client_sock,))
        client_handler.start()

if __name__ == '__main__':
    main()