# receiver_gui.py
import sys, threading, socket, struct, cv2, numpy as np
import sleap
from sleap.nn.inference import load_model
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

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

        # Layout
        def makeRow(label, line, btn):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(line)
            h.addWidget(btn)
            return h

        layout = QVBoxLayout()
        layout.addLayout(makeRow("Model:",      self.modelPath,  self.btnBrowseModel))
        layout.addLayout(makeRow("Infer out:",  self.inferPath,  self.btnBrowseInfer))
        layout.addLayout(makeRow("Video save:", self.videoPath,  self.btnBrowseVideo))
        layout.addWidget(self.preview)
        layout.addWidget(self.btnRun)
        layout.addWidget(self.btnStop)
        self.setLayout(layout)

        # — Signals —
        self.btnBrowseModel.clicked .connect(lambda: self._browse(self.modelPath,  True))
        self.btnBrowseInfer.clicked .connect(lambda: self._browse(self.inferPath,  False))
        self.btnBrowseVideo.clicked.connect(lambda: self._browse(self.videoPath, False))
        self.btnRun.clicked   .connect(self.start_receiving)
        self.btnStop.clicked  .connect(self.stop_receiving)

        # — State —
        self.sock     = None
        self.thread   = None
        self.running  = False
        self.predictor= None
        self.timer    = QTimer()
        self.timer.timeout .connect(self._update_preview)

        self.setWindowTitle("SLEAP Receiver")
        self.resize(800, 600)

    def _browse(self, lineedit, isModel):
        path = QFileDialog.getExistingDirectory(self, "Select Folder") if isModel else QFileDialog.getExistingDirectory(self, "Select Folder")
        if path: lineedit.setText(path)

    def start_receiving(self):
        if self.running: return
        # Load model
        self.predictor = load_model(self.modelPath.text(), batch_size=1)
        # Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("192.168.1.252", 5000))
        self.running = True
        # Start recv thread
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        # Start GUI update
        self.timer.start(30)  # ~30 Hz
        print("Started")

    def stop_receiving(self):
        self.running = False
        self.timer.stop()
        if self.sock:
            self.sock.close()
        print("Stopped")

    def _recv_loop(self):
        buf = b""
        while self.running:
            # read 4‑byte length
            while len(buf) < 4:
                buf += self.sock.recv(1024)
            size = struct.unpack(">I", buf[:4])[0]; buf = buf[4:]
            # read frame
            while len(buf) < size:
                buf += self.sock.recv(size - len(buf))
            data, buf = buf[:size], buf[size:]
            arr      = np.frombuffer(data, np.uint8)
            frame    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None: continue

            # SLEAP inference
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch = img[None, ...]
            res = self.predictor.inference_model.predict_on_batch(batch)
            # draw peaks
            for sc, pk in zip(res["instance_scores"][0], res["instance_peaks"][0]):
                if sc > 0.5:
                    for x, y in pk:
                        cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)

            # store for display & optional saving
            self.latest_frame = frame
            # TODO: write to video or inference‐csv using self.videoPath, self.inferPath

    def _update_preview(self):
        if hasattr(self, "latest_frame"):
            rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(img).scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ReceiverGUI()
    gui.show()
    sys.exit(app.exec_())
