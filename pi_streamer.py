import socket
import struct
import subprocess
import cv2
import numpy as np

# Server (PC) IP & Port
SERVER_IP = "192.168.1.252"  # <-- your PC's IP
PORT = 5000

# Start libcamera-vid to output MJPEG directly
cmd = [
    "libcamera-vid",
    "-t", "0",                      # no timeout (run forever)
    "--inline",                    # needed for MJPEG streaming
    "--width", "640",
    "--height", "480",
    "--framerate", "30",
    "--codec", "mjpeg",
    "-o", "-"                      # output to stdout
]

print("Starting libcamera-vid...")
stream = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))
print("Connected to PC.")

buffer = b""

try:
    while True:
        buffer += stream.stdout.read(1024)

        # Look for JPEG start and end markers
        start = buffer.find(b'\xff\xd8')
        end = buffer.find(b'\xff\xd9')

        if start != -1 and end != -1 and end > start:
            jpeg_data = buffer[start:end+2]
            buffer = buffer[end+2:]

            # Send frame size and data
            sock.sendall(struct.pack(">I", len(jpeg_data)))
            sock.sendall(jpeg_data)

except KeyboardInterrupt:
    print("Stopping capture...")

finally:
    stream.terminate()
    sock.close()
