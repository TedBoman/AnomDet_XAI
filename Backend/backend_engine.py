import socket
import json
import threading
from time import sleep

HOST = "localhost"
PORT = 9524
jobs = {
    "batch-jobs": [],
    "stream-jobs": []
}

def main():
    # Start a thread listening for requests
    listener_thread = threading.Thread(target=__request_listener)
    listener_thread.start()

    i = 1
    print("Counting in main thread...")
    # Main loop serving the backend logic
    while True:
        print(i)
        i += 1
        sleep(1)
    
# Handles the incoming requests and sends a response back to the client
def __handle_api_call(conn, data: dict) -> None:
    if data["METHOD"] == "run-batch":
        test_json = json.dumps({"test": "run-batch-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "run-stream":
        test_json = json.dumps({"test": "run-stream-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "change-model":
        test_json = json.dumps({"test": "change-model-respons" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "change-method":
        test_json = json.dumps({"test": "change-method-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "get-data":
        test_json = json.dumps({"test": "get-data-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "inject-anomaly":
        test_json = json.dumps({"test": "inject-anomaly-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "get-running":
        test_json = json.dumps({"test": "get-running-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "cancel":
        test_json = json.dumps({"test": "cancel-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "get-models":
        test_json = json.dumps({"test": "get-models-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    elif data["METHOD"] == "get-injection-methods":
        test_json = json.dumps({"test": "get-injection-methods-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))
    else: 
        test_json = json.dumps({"test": "error-response" })
        conn.sendall(bytes(test_json, encoding="utf-8"))        

# Listens for incoming requests and handles them through the __handle_api_call function
def __request_listener():
    try: 
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((HOST, PORT))
        sock.listen()

        while True:
            conn, addr = sock.accept()
            print(f'Connected to {addr}')
            recv_data = conn.recv(1024)
            recv_data = json.loads(recv_data)
            __handle_api_call(conn, recv_data)
            print(f"Received request: {recv_data}")
            conn.send(b'{Data was received properly}')
            conn.close() 
    
    except Exception as e:
        print(e)


if __name__ == "__main__": 
    main()