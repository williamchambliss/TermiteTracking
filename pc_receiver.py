import socket
import struct
import cv2
import numpy as np

# Server settings
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000

# Open network socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)

print("Waiting for connection...")
conn, addr = sock.accept()
print(f"Connected by {addr}")

try:
    while True:
        # Receive frame size first
        data = conn.recv(4)
        if not data:
            break
        frame_size = struct.unpack(">I", data)[0]

        # Receive the frame data
        frame_data = b""
        while len(frame_data) < frame_size:
            frame_data += conn.recv(frame_size - len(frame_data))

        # Decode frame
        np_arr = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("Received Video", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    conn.close()
    sock.close()
    cv2.destroyAllWindows()
