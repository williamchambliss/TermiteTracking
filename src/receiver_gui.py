import sys
import threading
import socket
import struct
import cv2
import numpy as np
import math
import time
import sleap
from sleap.nn.inference import load_model
from sleap import Labels, LabeledFrame, Video
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import json
import os
import shutil
import tensorflow as tf
import gc  # 🧹 Added for memory cleanup

BATCH_SIZE = 32
BATCH_TIMEOUT = 0.5  # seconds
MODEL_RELOAD_INTERVAL = 100  # reload every 100 chunks


class ReceiverGUI(QWidget):
    def __init__(self):
        super().__init__()
        sleap.disable_preallocation()
        self.lock = threading.Lock()

        # UI Elements
        self.centroidPath = QLineEdit()
        self.posePath = QLineEdit()
        self.inferPath = QLineEdit()
        self.videoPath = QLineEdit()
        self.btnBrowseCentroid = QPushButton("Browse…")
        self.btnBrowsePose = QPushButton("Browse…")
        self.btnBrowseInfer = QPushButton("Browse…")
        self.btnBrowseVideo = QPushButton("Browse…")
        self.btnRun = QPushButton("Run")
        self.btnStop = QPushButton("Stop")
        self.preview = QLabel(alignment=Qt.AlignCenter)

        def makeRow(label, line, btn):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(line)
            h.addWidget(btn)
            return h

        layout = QVBoxLayout()
        layout.addLayout(makeRow("Centroid model:", self.centroidPath, self.btnBrowseCentroid))
        layout.addLayout(makeRow("Pose model:", self.posePath, self.btnBrowsePose))
        layout.addLayout(makeRow("Infer out:", self.inferPath, self.btnBrowseInfer))
        layout.addLayout(makeRow("Video save:", self.videoPath, self.btnBrowseVideo))
        layout.addWidget(self.preview)
        layout.addWidget(self.btnRun)
        layout.addWidget(self.btnStop)
        self.setLayout(layout)

        # Connect buttons
        self.btnBrowseCentroid.clicked.connect(lambda: self._browse(self.centroidPath))
        self.btnBrowsePose.clicked.connect(lambda: self._browse(self.posePath))
        self.btnBrowseInfer.clicked.connect(lambda: self._browse(self.inferPath))
        self.btnBrowseVideo.clicked.connect(lambda: self._browse(self.videoPath))
        self.btnRun.clicked.connect(self.start_receiving)
        self.btnStop.clicked.connect(self.stop_receiving)

        # Networking
        self.sock = None
        self.running = False
        self.client_threads = []
        self.clients_lock = threading.Lock()

        # SLEAP & output
        self.predictor = None
        self.net_labels = Labels()
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_preview)
        self.writer = None
        self.writer_path = None
        self.video_fps = 10
        self.frame_idx = 0
        self.latest_frame = None
        self.video_ref = None
        self.chunk_counter = 0
        self.batch_since_save = 0
        self.chunk_dir = None

        self.setWindowTitle("SLEAP Receiver")
        self.resize(800, 600)

    def _browse(self, lineedit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            lineedit.setText(path)

    def start_receiving(self):
        if self.running:
            print("Already running")
            return
        self.running = True
        self.frame_idx = 0
        self.net_labels = Labels()
        self.video_ref = Video(backend="opencv")  # dummy Video object for LabeledFrame

        self.centroid_dir = self.centroidPath.text().strip()
        self.pose_dir = self.posePath.text().strip()
        if not self.centroid_dir or not self.pose_dir:
            print("Please specify both centroid and pose model directories.")
            return

        print("Loading models...")
        self.predictor = load_model([self.centroid_dir, self.pose_dir], batch_size=1)
        print("Models loaded.")

        skeleton_src = os.path.join(self.centroid_dir, "skeleton.json")
        skeleton_dst = os.path.join(self.inferPath.text().strip(), "skeleton.json")
        if os.path.exists(skeleton_src):
            os.makedirs(os.path.dirname(skeleton_dst), exist_ok=True)
            shutil.copy(skeleton_src, skeleton_dst)
            print(f"Skeleton copied to {skeleton_dst}")
        else:
            print("Warning: skeleton.json not found in model directory.")

        vid_dir = self.videoPath.text().strip()
        if vid_dir:
            os.makedirs(vid_dir, exist_ok=True)
            self.writer_path = os.path.join(vid_dir, "output.mp4")
            print(f"Saving video to: {self.writer_path}")
        else:
            self.writer_path = None

        inf_dir = self.inferPath.text().strip()
        if inf_dir:
            os.makedirs(inf_dir, exist_ok=True)
            self.json_path = os.path.join(inf_dir, "inference.json")
            self.chunk_dir = os.path.join(inf_dir, "chunks")
            os.makedirs(self.chunk_dir, exist_ok=True)
            print(f"Writing inference JSON to: {self.json_path}")
        else:
            self.json_path = None

        # Start network listener
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", 5000))
        self.sock.listen(1)
        threading.Thread(target=self.accept_clients, daemon=True).start()
        self.timer.start(30)

    def stop_receiving(self):
        print("Stopping receiver...")
        self.running = False
        self.timer.stop()

        with self.clients_lock:
            for th in self.client_threads:
                try:
                    th.conn.shutdown(socket.SHUT_RDWR)
                    th.conn.close()
                except:
                    pass
            self.client_threads.clear()

        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.sock.close()
            self.sock = None

        if self.writer:
            self.writer.release()
            print(f"Video saved to {self.writer_path}")
            self.writer = None

        # Save remaining labels
        if self.net_labels and len(self.net_labels) > 0:
            chunk_file = os.path.join(self.chunk_dir, f"labels_chunk_{self.chunk_counter}.slp")
            try:
                self.net_labels.save(chunk_file)
                print(f"Saved final .slp chunk: {chunk_file}")
            except Exception as e:
                print(f"Error saving final .slp chunk: {e}")

        if getattr(self, 'json_path', None):
            try:
                with open(self.json_path, 'w') as f:
                    json.dump(self.net_labels.to_dict(), f, indent=2)
                print(f"Inference JSON saved to {self.json_path}")
            except Exception as e:
                print(f"Error saving inference JSON: {e}")

        print("Stopped receiving.")

    def accept_clients(self):
        while self.running:
            try:
                conn, addr = self.sock.accept()
            except:
                break
            print(f"Connected by {addr}")
            th = threading.Thread(target=self.handle_client, args=(conn,), daemon=True)
            th.conn = conn
            with self.clients_lock:
                self.client_threads.append(th)
            th.start()

    def handle_client(self, conn):
        buf = b""
        first = True
        frames, indices = [], []
        last = time.time()
        while self.running:
            try:
                while len(buf) < 4:
                    data = conn.recv(4096)
                    if not data:
                        raise ConnectionResetError
                    buf += data
                size = struct.unpack(">I", buf[:4])[0]
                buf = buf[4:]
                while len(buf) < size:
                    data = conn.recv(size-len(buf))
                    if not data:
                        raise ConnectionResetError
                    buf += data
                chunk, buf = buf[:size], buf[size:]
                frame = cv2.imdecode(np.frombuffer(chunk, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if first and self.writer_path:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.writer = cv2.VideoWriter(self.writer_path, fourcc, self.video_fps, (w, h))
                    first = False
                frames.append(frame)
                indices.append(self.frame_idx)
                self.frame_idx += 1
                last = time.time()
                if len(frames) >= BATCH_SIZE or (frames and time.time() - last > BATCH_TIMEOUT):
                    self._process_batch(frames, indices)
                    frames.clear()
                    indices.clear()
            except ConnectionResetError:
                break
            except Exception as e:
                print("Error in client loop:", e)
                break
        if frames:
            self._process_batch(frames, indices)
        try:
            conn.shutdown(socket.SHUT_RDWR)
            conn.close()
        except:
            pass
        print("Client closed")

    def _process_batch(self, frames, indices):
        if not frames:
            return

        lfs = self.predictor.predict(np.array(frames))

        for i, lf_data in enumerate(lfs):
            frame_i = indices[i]
            lf = LabeledFrame(video=self.video_ref, frame_idx=frame_i)
            lf.instances = lf_data.instances

            try:
                self.net_labels.append(lf)
            except Exception as e:
                print(f"Error appending LabeledFrame: {e}")

            if self.writer:
                self.writer.write(frames[i])

            if i == len(frames) - 1:
                img_preview = frames[i].copy()
                for inst in lf.instances:
                    for pt in inst.points:
                        if math.isfinite(pt.x) and math.isfinite(pt.y):
                            cv2.circle(img_preview, (int(pt.x), int(pt.y)), 3, (0, 255, 0), -1)
                with self.lock:
                    self.latest_frame = img_preview

        # Save .slp every 2 batches
        self.batch_since_save += 1
        if self.batch_since_save >= 2:
            chunk_file = os.path.join(self.chunk_dir, f"labels_chunk_{self.chunk_counter}.slp")
            try:
                self.net_labels.save(chunk_file)
                print(f"Saved .slp chunk: {chunk_file}")
            except Exception as e:
                print(f"Error saving .slp chunk: {e}")
            self.net_labels = Labels()
            self.chunk_counter += 1
            self.batch_since_save = 0

            # 🧠 Reload model every MODEL_RELOAD_INTERVAL chunks
            if self.chunk_counter % MODEL_RELOAD_INTERVAL == 0:
                print("Reloading SLEAP models to free memory...")
                self.predictor = load_model([self.centroid_dir, self.pose_dir], batch_size=1)
                print("Model reloaded successfully.")

        # 🧹 Memory cleanup after every batch
        tf.keras.backend.clear_session()
        gc.collect()

    def _update_preview(self):
        with self.lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ReceiverGUI()
    gui.show()
    sys.exit(app.exec_())
