import cv2
import socket
import struct

# Server (PC) IP & Port
SERVER_IP = "192.168.1.252"  # <-- Change this to your PC's IP address
PORT = 5000

# Open network socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))

# Open camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 4056)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 3040)

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Crop to 2000x2000 pixels (center crop)
        cropped_frame = frame[520:2520, 1028:3028]

        # Encode to JPEG
        ret, encoded_img = cv2.imencode(".jpg", cropped_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        data = encoded_img.tobytes()

        # Send frame size first
        sock.sendall(struct.pack(">I", len(data)))

        # Send the frame data
        sock.sendall(data)

except KeyboardInterrupt:
    print("Stopping capture...")

finally:
    camera.release()
    sock.close()
