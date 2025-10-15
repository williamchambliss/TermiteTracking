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
    buf = b""
    while True:
        # Read in chunks for efficiency (not one byte at a time)
        data = proc.stdout.read(4096)
        if not data:
            break
        buf += data

        while True:
            # Find the start of a JPEG frame
            start = buf.find(b'\xff\xd8')
            if start == -1:
                # No start marker yet; wait for more data
                buf = b""
                break

            # Find the end of this JPEG frame
            end = buf.find(b'\xff\xd9', start)
            if end == -1:
                # Incomplete frame; wait for more data
                buf = buf[start:]
                break

            # Extract full JPEG frame
            jpeg_data = buf[start:end+2]

            # Remove it from buffer for next iteration
            buf = buf[end+2:]

            # Push to queue without blocking
            try:
                frame_queue.put_nowait(jpeg_data)
            except queue.Full:
                print("Dropped frame: queue full")

def sender_loop():
    while True:
        jpeg_data = frame_queue.get()
        timestamp = time.time()
        try:
            sock.sendall(struct.pack(">dI", timestamp, len(jpeg_data)))
            sock.sendall(jpeg_data)
        except BrokenPipeError:
            break

# Start threads
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=sender_loop, daemon=True).start()

# Keep alive
while True:
    time.sleep(1)



