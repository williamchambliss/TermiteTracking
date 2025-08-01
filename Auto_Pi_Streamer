import socket
import struct
import subprocess
import threading
import queue
import time
import sys

PC_IP_ADDRESS = "192.168.4.71"  # Replace with your actual PC IP
PORT = 5000

frame_queue = queue.Queue(maxsize=50)

def start_camera():
    return subprocess.Popen([
        "libcamera-vid",
        "--width", "1000",
        "--height", "1000",
        "--framerate", "1",
        "--codec", "mjpeg",
        "--timeout", "0",
        "-o", "-"
    ], stdout=subprocess.PIPE, bufsize=0)

def capture_loop(proc):
    while True:
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
    try:
        with socket.create_connection((PC_IP_ADDRESS, PORT)) as sock:
            print("Connected to host")
            while True:
                jpeg_data = frame_queue.get()
                sock.sendall(struct.pack(">I", len(jpeg_data)))
                sock.sendall(jpeg_data)
    except (ConnectionRefusedError, OSError) as e:
        print(f"Connection failed or lost: {e}")


def main():
    proc = start_camera()
    threading.Thread(target=capture_loop, args=(proc,), daemon=True).start()
    threading.Thread(target=sender_loop, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        proc.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
