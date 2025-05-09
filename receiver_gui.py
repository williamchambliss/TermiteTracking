# receiver_gui.py
import sys
import threading
import socket
import struct
import cv2
import numpy as np
import sleap
from sleap.nn.inference import load_model
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import csv
import os

class ReceiverGUI(QWidget):
    def __init__(self):
        super().__init__()
        sleap.disable_preallocation()

        # — UI Elements —
        self.modelPath = QLineEdit()
        self.inferPath = QLineEdit()
        self.videoPath = QLineEdit()
        self.btnBrowseModel = QPushButton("Browse…")
        self.btnBrowseInfer = QPushButton("Browse…")
        self.btnBrowseVideo = QPushButton("Browse…")
        self.btnRun  = QPushButton("Run")
        self.btnStop = QPushButton("Stop")
        self.preview = QLabel(alignment=Qt.AlignCenter)

        # Layout helper
        def makeRow(label, line, btn):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(line)
            h.addWidget(btn)
            return h

        # — Layout —
        layout = QVBoxLayout()
        layout.addLayout(makeRow("Model:",      self.modelPath,  self.btnBrowseModel))
        layout.addLayout(makeRow("Infer out:",  self.inferPath,  self.btnBrowseInfer))
        layout.addLayout(makeRow("Video save:", self.videoPath,  self.btnBrowseVideo))
        layout.addWidget(self.preview)
        layout.addWidget(self.btnRun)
        layout.addWidget(self.btnStop)
        self.setLayout(layout)

        # — Signals —
        self.btnBrowseModel.clicked.connect(lambda: self._browse(self.modelPath))
        self.btnBrowseInfer.clicked.connect(lambda: self._browse(self.inferPath))
        self.btnBrowseVideo.clicked.connect(lambda: self._browse(self.videoPath))
        self.btnRun.clicked.connect(self.start_receiving)
        self.btnStop.clicked.connect(self.stop_receiving)

        # — State —
        self.sock = None
        self.thread = None
        self.running = False
        self.predictor = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_preview)
        # Video saving state
        self.writer = None
        self.video_fps = 5  # Match Pi stream rate
        # Inference saving state
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0

        self.setWindowTitle("SLEAP Receiver")
        self.resize(800, 600)

    def _browse(self, lineedit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            lineedit.setText(path)

    def start_receiving(self):
        if self.running:
            return
        # Reset frame counter
        self.frame_idx = 0
        # Load SLEAP model
        model_dir = self.modelPath.text()
        self.predictor = load_model(model_dir, batch_size=1)

        # Open socket to Pi
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("192.168.1.252", 5000))

        # Prepare video writer if a save path was provided
        vid_dir = self.videoPath.text().rstrip('/')
        if vid_dir:
            os.makedirs(vid_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_path = f"{vid_dir}/output.avi"
            # Assuming incoming frames are 2000x2000
            self.writer = cv2.VideoWriter(out_path, fourcc, self.video_fps, (2000, 2000))
            print(f"Saving video to: {out_path}")

        # Prepare CSV writer if an inference path was provided
        inf_dir = self.inferPath.text().rstrip('/')
        if inf_dir:
            os.makedirs(inf_dir, exist_ok=True)
            csv_path = f"{inf_dir}/inference.csv"
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['frame', 'instance', 'keypoint', 'x', 'y', 'score'])
            print(f"Writing inference to: {csv_path}")

        # Start receiving thread and GUI updates
        self.running = True
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        self.timer.start(30)  # ~30 Hz GUI refresh
        print("Started receiving and inference.")

    def stop_receiving(self):
        # Stop threads and close socket
        self.running = False
        self.timer.stop()
        if self.sock:
            self.sock.close()
            self.sock = None
        # Release video writer
        if self.writer:
            self.writer.release()
            self.writer = None
            print("Video saved and writer released.")
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            print("Inference CSV saved and closed.")
        print("Stopped receiving.")

    def _recv_loop(self):
        buf = b""
        while self.running:
            # Read 4-byte frame size header
            while len(buf) < 4:
                data = self.sock.recv(1024)
                if not data:
                    return
                buf += data
            size = struct.unpack(">I", buf[:4])[0]
            buf = buf[4:]

            # Read full JPEG frame
            while len(buf) < size:
                data = self.sock.recv(size - len(buf))
                if not data:
                    return
                buf += data
            frame_data, buf = buf[:size], buf[size:]
            arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Run SLEAP inference and draw keypoints
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch = img_rgb[None, ...]
            res = self.predictor.inference_model.predict_on_batch(batch)
            for inst_idx, (score, peaks) in enumerate(zip(res["instance_scores"][0], res["instance_peaks"][0])):
                if score > 0.5:
                    for kp_idx, (x, y) in enumerate(peaks):
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                        # Write inference row
                        if self.csv_writer:
                            self.csv_writer.writerow([self.frame_idx, inst_idx, kp_idx, float(x), float(y), float(score)])

            # Write annotated frame to video if enabled
            if self.writer:
                self.writer.write(frame)

            # Update latest_frame for GUI preview and increment frame
            self.latest_frame = frame
            self.frame_idx += 1

    def _update_preview(self):
        if hasattr(self, "latest_frame"):
            rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.preview.setPixmap(pix.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ReceiverGUI()
    gui.show()
    sys.exit(app.exec_())
