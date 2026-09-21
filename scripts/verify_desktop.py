"""Offscreen integration check using the real Qt worker and GPU backend."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys, json
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from src.config import Settings, load_rois
from src.pipeline import AnalysisRunner
from src.ui.main_window import MainWindow
from src.ui.ai_worker import AIWorker
from src.storage import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    output_root = parser.parse_args().output.resolve()
    app = QApplication([])
    w = MainWindow()
    w.show()
    errors = []
    results = []
    frames = []
    latencies = []
    w.canvas.painted_latency.connect(latencies.append)
    w.video_edit.setText(str(ROOT / "data/video_1.mp4"))
    w.preview()
    w.error = lambda text: errors.append(text)
    settings = Settings()
    rois = load_rois(ROOT / "data/roi_config.json")
    runner = AnalysisRunner(
        ROOT / "data/video_1.mp4",
        ROOT / "models/yolo11n-pose.pt",
        settings,
        rois,
        "cuda:0",
        output_root / "qt_verification",
    )
    worker = AIWorker(runner)
    w.worker = worker
    w.rois = rois
    w.update_settings()
    w.set_busy(True)
    worker.frame_signal.connect(w.update_frame)
    worker.frame_signal.connect(lambda img, data: frames.append(data["time"]))
    worker.event_signal.connect(w.update_event)
    worker.log_signal.connect(w.log.append)
    worker.error_signal.connect(errors.append)
    worker.result_signal.connect(results.append)

    def finished():
        w.worker = None
        w.set_busy(False)
        if results:
            w.load_session(results[0])
            w.status.setText("GPU / Qt 完整视频验证已完成；列表支持原始视频回放")
            out = output_root / "desktop_verification"
            out.mkdir(parents=True, exist_ok=True)
            app.processEvents()
            w.grab().save(str(out / "desktop.png"))
            before = len(w.events)
            if before:
                w.replay_event(0)

            def end():
                w.stop_replay()
                result = {
                    "errors": errors,
                    "session": results[0],
                    "frames_displayed": len(frames),
                    "events_before_replay": before,
                    "events_after_replay": len(w.events),
                    "replay_preserves_events": len(w.events) == before,
                    "actual_paint_samples": len(latencies),
                    "decode_to_qt_paint_p50_ms": float(np.percentile(latencies, 50))
                    if latencies
                    else None,
                    "decode_to_qt_paint_p95_ms": float(np.percentile(latencies, 95))
                    if latencies
                    else None,
                    "render_surface": "Qt offscreen raster; not physical monitor scanout",
                }
                write_json(out / "result.json", result)
                w.close()
                app.quit()

            QTimer.singleShot(2500, end)
        else:
            w.close()
            app.quit()

    worker.finished.connect(finished)
    QTimer.singleShot(600000, lambda: (worker.stop(), errors.append("timeout")))
    worker.start()
    app.exec()
    worker.wait()
    print(
        json.dumps(
            {"errors": errors, "sessions": results, "displayed": len(frames)}, ensure_ascii=False
        )
    )
    return int(bool(errors) or not results)


if __name__ == "__main__":
    raise SystemExit(main())
