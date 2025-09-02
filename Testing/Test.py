import subprocess
import threading
import queue
import cv2
import numpy as np
import time

# Queue for frame buffering
frame_queue = queue.Queue(maxsize=50)

# Launch libcamera-vid (capture bigger than 1000x1000, so we can crop)
proc = subprocess.Popen([
    "libcamera-vid",
    "--width", "1920",
    "--height", "1080",
    "--framerate", "5",   # adjust FPS depending on Pi performance
    "--codec", "mjpeg",
    "--timeout", "0",
    "-o", "-"
], stdout=subprocess.PIPE, bufsize=0)

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
    return img

def capture_loop():
    """Read MJPEG frames from libcamera-vid and put into queue."""
    while True:
        # Read JPEG frame start
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

def display_loop():
    """Display cropped frames live."""
    first = True
    while True:
        jpeg_data = frame_queue.get()
        global roi
        if first:
            init_crop_manual(jpeg_data)  # Pick ROI once
            first = False

        frame = crop_and_resize(jpeg_data)
        cv2.imshow("Cropped Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Start threads
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=display_loop, daemon=True).start()

# Keep alive
while True:
    time.sleep(1)
