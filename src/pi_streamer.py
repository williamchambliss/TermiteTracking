import socket
import struct
import subprocess
import threading
import queue
import time

# Server (PC) IP & Port
PC_IP_ADDRESS = "169.254.221.104"  # <-- your PC's IP CHANGE WHEN CHANGED
PORT = 5000
#libcamera-hello -t 0 --width 1000 --height 1000 --roi 0.22,0.0,0.7,0.9 YESSSSS           gvb mv,klbv
# Queue for frame buffering
frame_queue = queue.Queue(maxsize=50)

# Launch libcamera-vid
proc = subprocess.Popen([
    "libcamera-vid",
    "--width", "1000",
    "--height", "1000",
    "--framerate", "10",   # Change this frame rate depending on compute of machine (1 low, 5 high but actually usable)
    "--codec", "mjpeg",
    "--roi", "0.22,0.0,0.7,0.9",
    "--timeout", "0",
    "-o", "-"
], stdout=subprocess.PIPE, bufsize=0)

# Socket setup
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((PC_IP_ADDRESS, 5000))

def capture_loop():
    while True:
        # Read JPEG frame length (2-byte MJPEG marker + JPEG header)
        start = proc.stdout.read(2)
        if not start:
            break
        if start != b'\xff\xd8':
            continue
        jpeg_data = start
        while True:
            byte = proc.stdout.read(1)
            if not byte:
                break
            jpeg_data += byte
            if jpeg_data[-2:] == b'\xff\xd9':
                break
        try:
            frame_queue.put_nowait(jpeg_data)
        except queue.Full:
            print("Dropped frame: queue full")

def sender_loop():
    while True:
        jpeg_data = frame_queue.get()
        timestamp = time.time()
        try:
            sock.sendall(struct.pack(">I", timestamp, len(jpeg_data)))
            sock.sendall(jpeg_data)
        except BrokenPipeError:
            break

# Start threads
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=sender_loop, daemon=True).start()

# Keep alive
while True:
    time.sleep(1)

