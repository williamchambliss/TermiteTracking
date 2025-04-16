import socket
import struct
import cv2
import numpy as np

HOST = "0.0.0.0"
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)

print("Waiting for connection...")
conn, addr = sock.accept()
print(f"Connected by {addr}")

try:
    while True:
        # Receive frame size
        data = conn.recv(4)
        if not data:
            break
        frame_size = struct.unpack(">I", data)[0]

        # Receive frame data
        frame_data = b""
        while len(frame_data) < frame_size:
            packet = conn.recv(frame_size - len(frame_data))
            if not packet:
                break
            frame_data += packet

        # Decode JPEG
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("Received Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    conn.close()
    sock.close()
    cv2.destroyAllWindows()
