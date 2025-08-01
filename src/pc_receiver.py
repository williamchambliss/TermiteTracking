import socket
import struct
import cv2
import numpy as np
import os
from datetime import datetime

# Server settings
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000

# Frame properties (must match sender settings)
FRAME_RATE = 1
FRAME_WIDTH = 2000
FRAME_HEIGHT = 2000

# Create video filename with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_filename = f"received_video_{timestamp}.avi"

# Define the codec and initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'mp4v' for .mp4 files
out = cv2.VideoWriter(video_filename, fourcc, FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))

# Open network socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)

print("Waiting for connection...")
conn, addr = sock.accept()
print(f"Connected by {addr}")
print(f"Recording to {video_filename}")

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
            out.write(frame)  # Save frame to video file
            print("Frame received and written to video.")
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
    out.release()  # Finalize video file
    cv2.destroyAllWindows()
