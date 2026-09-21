import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


class AIWorker(QThread):
    frame_signal = Signal(QImage, dict)
    event_signal = Signal(str, dict)
    log_signal = Signal(str)
    result_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, runner):
        super().__init__()
        self.runner = runner
        self.last_emit = -1.0
        self.show_skeleton = True
        self.show_angles = False
        self.show_roi = True

    def on_frame(self, frame, details, stats):
        # Video-time throttling is deterministic; no sleeping in the analysis path.
        if stats["time"] - self.last_emit < 1 / self.runner.settings.gui_fps:
            return
        self.last_emit = stats["time"]
        # The GUI paints presentation overlays in source coordinates. The shared
        # analysis path and evidence renderer remain unchanged.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        q = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.frame_signal.emit(q, {**stats, "people_overlay": details})

    def run(self):
        try:
            result = self.runner.run(
                on_frame=self.on_frame,
                on_event=self.event_signal.emit,
                on_status=self.log_signal.emit,
                cancel=self.isInterruptionRequested,
            )
            self.result_signal.emit(str(result))
        except Exception as exc:
            self.error_signal.emit(f"{type(exc).__name__}: {exc}")

    def stop(self):
        self.requestInterruption()
