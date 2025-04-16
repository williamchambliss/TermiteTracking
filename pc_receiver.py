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
        # Receive frame size (4 bytes)
        header = conn.recv(4)
        if not header:
            print("Connection closed.")
            break

        frame_size = struct.unpack(">I", header)[0]

        # Receive full frame data
        frame_data = b""
        while len(frame_data) < frame_size:
            packet = conn.recv(frame_size - len(frame_data))
            if not packet:
                break
            frame_data += packet

        # Decode JPEG frame
        np_arr = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("Received Video", frame)
            print("Frame received.")
        else:
            print("Failed to decode frame.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting viewer.")
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    conn.close()
    sock.close()
    cv2.destroyAllWindows()
