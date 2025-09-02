import socket
import struct
import subprocess
import threading
import queue
import time
import cv2
import numpy as np

# Server (PC) IP & Port
PC_IP_ADDRESS = "192.168.4.71"  # <-- your PC's IP
PORT = 5000

# Queue for frame buffering
frame_queue = queue.Queue(maxsize=50)

# Launch libcamera-vid (full resolution, we crop later)
proc = subprocess.Popen([
    "libcamera-vid",
    "--width", "1920",      # capture larger, crop manually
    "--height", "1080",
    "--framerate", "1",
    "--codec", "mjpeg",
    "--timeout", "0",
    "-o", "-"
], stdout=subprocess.PIPE, bufsize=0)

# Socket setup
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((PC_IP_ADDRESS, PORT))

roi = None  # (x, y, w, h)

def init_crop_manual(jpeg_data):
    """Show first frame, let user draw ROI."""
    global roi
    img_arr = np.frombuffer(jpeg_data, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    r = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    roi = r  # (x, y, w, h)
    print(f"Selected ROI: {roi}")

def crop_and_resize(jpeg_data):
    """Crop with stored ROI and resize to 1000x1000."""
    global roi
    img_arr = np.frombuffer(jpeg_data, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if roi is not None:
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]

    img = cv2.resize(img, (1000, 1000))
    ret, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

def capture_loop():
    while True:
        # Read JPEG frame
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
    first = True
    while True:
        jpeg_data = frame_queue.get()
        global roi
        if first:
            init_crop_manual(jpeg_data)
            first = False
        cropped_data = crop_and_resize(jpeg_data)
        try:
            sock.sendall(struct.pack(">I", len(cropped_data)))
            sock.sendall(cropped_data)
        except BrokenPipeError:
            break

# Start threads
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=sender_loop, daemon=True).start()

# Keep alive
while True:
    time.sleep(1)
