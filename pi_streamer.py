import socket
import struct
import subprocess
import cv2
import numpy as np

# Server (PC) IP & Port
SERVER_IP = "192.168.1.252"
PORT = 5000

# Start the subprocess that streams MJPEG from libcamera-vid through ffmpeg
cmd = (
    "libcamera-vid -t 0 --inline --width 640 --height 480 --framerate 30 "
    "--codec mjpeg -o - | "
    "ffmpeg -fflags nobuffer -flush_packets 1 -i - -f image2pipe -vcodec mjpeg -"
)
print("Starting libcamera stream...")
stream = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# Connect to the PC
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, PORT))
print("Connected to PC.")

try:
    while True:
        # Read a chunk of data (JPEG frames aren't fixed size, so we may need to buffer smartly)
        data = stream.stdout.read(4096)

        if not data:
            print("No data received from libcamera stream.")
            break

        # Wait until we have a complete JPEG frame (basic JPEG end marker detection)
        while not data.endswith(b'\xff\xd9'):
            more = stream.stdout.read(4096)
            if not more:
                break
            data += more

        # Send frame size
        sock.sendall(struct.pack(">I", len(data)))
        # Send frame data
        sock.sendall(data)

except KeyboardInterrupt:
    print("Stopping capture...")

finally:
    stream.terminate()
    sock.close()
