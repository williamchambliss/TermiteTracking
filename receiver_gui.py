# receiver_gui.py
import sys
import threading
import socket
import struct
import cv2
import numpy as np
import math
import sleap
from sleap.nn.inference import load_model
from sleap import Video
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import json
import os

class ReceiverGUI(QWidget):
    def __init__(self):
        super().__init__()
        sleap.disable_preallocation()
        self.lock = threading.Lock()

        # UI Elements
        self.centroidPath  = QLineEdit()
        self.posePath      = QLineEdit()
        self.inferPath     = QLineEdit()
        self.videoPath     = QLineEdit()
        self.btnBrowseCentroid = QPushButton("Browse…")
        self.btnBrowsePose     = QPushButton("Browse…")
        self.btnBrowseInfer    = QPushButton("Browse…")
        self.btnBrowseVideo    = QPushButton("Browse…")
        self.btnRun       = QPushButton("Run")
        self.btnStop      = QPushButton("Stop")
        self.preview      = QLabel(alignment=Qt.AlignCenter)

        # Layout helper
        def makeRow(label, line, btn):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(line)
            h.addWidget(btn)
            return h

        # Layout setup
        layout = QVBoxLayout()
        layout.addLayout(makeRow("Centroid model:", self.centroidPath, self.btnBrowseCentroid))
        layout.addLayout(makeRow("Pose model:",     self.posePath,     self.btnBrowsePose))
        layout.addLayout(makeRow("Infer out:",      self.inferPath,    self.btnBrowseInfer))
        layout.addLayout(makeRow("Video save:",     self.videoPath,    self.btnBrowseVideo))
        layout.addWidget(self.preview)
        layout.addWidget(self.btnRun)
        layout.addWidget(self.btnStop)
        self.setLayout(layout)

        # Signals
        self.btnBrowseCentroid.clicked.connect(lambda: self._browse(self.centroidPath))
        self.btnBrowsePose.clicked.connect(lambda: self._browse(self.posePath))
        self.btnBrowseInfer.clicked.connect(lambda: self._browse(self.inferPath))
        self.btnBrowseVideo.clicked.connect(lambda: self._browse(self.videoPath))
        self.btnRun.clicked.connect(self.start_receiving)
        self.btnStop.clicked.connect(self.stop_receiving)

        # State
        self.sock        = None
        self.conn        = None
        self.thread      = None
        self.running     = False
        self.predictor   = None
        self.timer       = QTimer()
        self.timer.timeout.connect(self._update_preview)
        self.writer      = None
        self.writer_path = None
        self.video_fps   = 5
        self.json_buffer = []
        self.frame_idx   = 0

        self.setWindowTitle("SLEAP Receiver")
        self.resize(800, 600)

    def _browse(self, lineedit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            lineedit.setText(path)

    def start_receiving(self):
        if self.running:
            return
        self.frame_idx   = 0
        self.json_buffer = []

        # Load top-down models
        centroid_dir = self.centroidPath.text().strip()
        pose_dir     = self.posePath.text().strip()
        self.predictor = load_model([centroid_dir, pose_dir], batch_size=1)

        # Video writer path
        vid_dir = self.videoPath.text().rstrip('/').strip()
        if vid_dir:
            os.makedirs(vid_dir, exist_ok=True)
            self.writer_path = os.path.join(vid_dir, "output.avi")
            print(f"Saving video to: {self.writer_path}")

        # Inference JSON path
        inf_dir = self.inferPath.text().rstrip('/').strip()
        if inf_dir:
            os.makedirs(inf_dir, exist_ok=True)
            self.json_path = os.path.join(inf_dir, "inference.json")
            print(f"Writing inference to: {self.json_path}")

        # Start thread and timer
        self.running = True
        self.thread  = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        self.timer.start(30)

    def stop_receiving(self):
        self.running = False
        self.timer.stop()

        if self.conn:
            self.conn.close(); self.conn = None
        if self.sock:
            self.sock.close(); self.sock = None
        if self.writer:
            self.writer.release(); print("Video saved.")

        if hasattr(self, 'json_path'):
            with open(self.json_path, 'w') as f:
                json.dump(self.json_buffer, f, indent=2)
            print("Inference JSON saved.")

        print("Stopped receiving.")

    def _recv_loop(self):
        try:
            # Server setup
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(("0.0.0.0", 5000))
            self.sock.listen(1)
            print("Server listening on port 5000")

            self.conn, addr = self.sock.accept()
            print(f"Connected by {addr}")

            buf = b""; first_frame = True
            while self.running:
                # Read frame size
                while len(buf) < 4:
                    data = self.conn.recv(1024)
                    if not data:
                        return
                    buf += data
                frame_size = struct.unpack(">I", buf[:4])[0]
                buf = buf[4:]

                # Read frame data
                while len(buf) < frame_size:
                    data = self.conn.recv(frame_size - len(buf))
                    if not data:
                        return
                    buf += data
                frame_bytes, buf = buf[:frame_size], buf[frame_size:]

                frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Init VideoWriter
                if first_frame and self.writer_path:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self.writer = cv2.VideoWriter(self.writer_path, fourcc, self.video_fps, (w, h))
                    if not self.writer.isOpened():
                        print("ERROR: VideoWriter failed to open.")
                    first_frame = False

                # Run inference
                vid       = Video.from_numpy(np.expand_dims(frame, 0))
                lfs       = self.predictor.predict(vid)
                instances = lfs[0].instances

                                                # Draw keypoints and buffer JSON
                for inst_idx, inst in enumerate(instances):
                    pts = inst.points
                    scrs = inst.scores

                    # Determine coordinate arrays
                    if isinstance(pts, tuple) and len(pts) == 2 and not isinstance(pts[0], (float, int)):
                        # Tuple of two arrays/lists
                        x_coords = np.asarray(pts[0], dtype=float)
                        y_coords = np.asarray(pts[1], dtype=float)
                    else:
                        # Flat sequence of numbers? convert to array
                        try:
                            arr = np.asarray(pts, dtype=float).ravel()
                        except Exception:
                            print(f"Skipping non-numeric inst.points: {type(pts)}")
                            continue
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            x_coords = arr[:, 0]
                            y_coords = arr[:, 1]
                        elif arr.ndim == 1 and arr.size % 2 == 0:
                            pairs = arr.reshape(-1, 2)
                            x_coords = pairs[:, 0]
                            y_coords = pairs[:, 1]
                        else:
                            print(f"Skipping unsupported inst.points length {arr.size}")
                            continue

                    num_kp = len(x_coords)
                    for kp_idx in range(num_kp):
                        x = x_coords[kp_idx]
                        y = y_coords[kp_idx]
                        score = scrs[kp_idx] if len(scrs) > kp_idx else 0.0

                        # Skip invalid
                        if not (math.isfinite(x) and math.isfinite(y)):
                            continue

                        ix, iy = int(x), int(y)
                        cv2.circle(frame, (ix, iy), 3, (0, 255, 0), -1)
                        self.json_buffer.append({
                            'frame':    self.frame_idx,
                            'instance': inst_idx,
                            'keypoint': kp_idx,
                            'x':        float(x),
                            'y':        float(y),
                            'score':    float(score),
                        })

                # Write annotated frame
                    self.writer.write(frame)
                with self.lock:
                    self.latest_frame = frame.copy()
                self.frame_idx += 1

        except Exception as e:
            print("Error in receive loop:", e)
        finally:
            self.running = False

    def _update_preview(self):
        with self.lock:
            if not hasattr(self, 'latest_frame'):
                return
            frame = self.latest_frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(pix)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ReceiverGUI()
    gui.show()
    sys.exit(app.exec_())

