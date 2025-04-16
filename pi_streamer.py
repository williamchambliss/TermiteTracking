import cv2
import socket
import struct
import numpy as np
import subprocess

# Server (PC) IP & Port
SERVER_IP = "192.168.1.252"  # Change this to the PC's IP
PORT = 5000
BUFFER_SIZE = 4096

# Open network socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))

# Open camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 4056)  # Max resolution for Pi AI Camera
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 3040)

# FFmpeg H.264 encoding
ENCODED_FORMAT = ".mp4"
QUALITY = 23  # Lower is better quality (0–51)
ffmpeg_command = f"ffmpeg -i pipe:0 -c:v libx264 -preset ultrafast -crf {QUALITY} -f mp4 pipe:1"

ffmpeg_process = subprocess.Popen(
    ffmpeg_command.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
)

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Crop to 2000x2000 pixels (center crop)
        cropped_frame = frame[520:2520, 1028:3028]

        # Encode frame using FFmpeg
        ffmpeg_process.stdin.write(cropped_frame.tobytes())

        # Read FFmpeg's output
        encoded_frame = ffmpeg_process.stdout.read(BUFFER_SIZE)

        # Send frame size first
        sock.sendall(struct.pack(">I", len(encoded_frame)))

        # Send the frame
        sock.sendall(encoded_frame)

except KeyboardInterrupt:
    print("Stopping capture...")

finally:
    camera.release()
    sock.close()
    ffmpeg_process.stdin.close()
    ffmpeg_process.stdout.close()
    ffmpeg_process.wait()
