import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from PySide6.QtCore import QPointF, Qt, QEvent
from PySide6.QtGui import QMouseEvent
from src.ui.main_window import MainWindow, VideoCanvas


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_canvas_rejects_black_bar_click(app):
    c = VideoCanvas()
    c.resize(800, 800)
    c.set_image(QImage(1920, 1080, QImage.Format.Format_RGB888))
    c.target = "left"
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(400, 20),
        QPointF(400, 20),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    c.mousePressEvent(event)
    assert not c.points
    c.close()


def test_ui_invalid_start_does_not_lock_buttons(app, tmp_path):
    w = MainWindow()
    errors = []
    w.error = errors.append
    w.start_analysis()
    assert errors and w.start_btn.isEnabled() and not w.stop_btn.isEnabled()
    w.video_edit.setText(str(tmp_path / "missing.mp4"))
    w.preview()
    assert len(errors) == 2
    w.close()


def test_history_updates_counts_and_invalidates_other_video_roi(app, tmp_path):
    from src.storage import write_json, sha256
    import json

    source = tmp_path / "placeholder.bin"
    source.write_bytes(b"hash-only fixture")
    session = tmp_path / "session"
    session.mkdir()
    write_json(
        session / "manifest.json",
        {"video_path": str(source), "video_sha256": sha256(source), "rois": {}},
    )
    write_json(session / "summary.json", {"last_video_time": 4})
    event = {
        "event_id": "e1",
        "track_id": 1,
        "event_type": "bend",
        "roi": "",
        "start_time": 0,
        "end_time": 3,
        "effective_duration": 3,
        "alert_time": None,
        "end_reason": "eof",
    }
    (session / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    w = MainWindow()
    w.loaded_video = "other-video"
    w.roi_confirmed = True
    w.load_session(session)
    assert "姿态候选 1" in w.stats.text()
    assert w.loaded_video is None and not w.roi_confirmed
    errors = []
    w.error = errors.append
    w.confirm_roi()
    assert errors
    w.close()


def test_partial_roi_cannot_be_confirmed_and_clear_cancels_drawing(app):
    w = MainWindow()
    w.canvas.target = "left"
    w.canvas.points = [(0.1, 0.1), (0.2, 0.2)]
    errors = []
    w.error = errors.append
    w.confirm_roi()
    assert errors and not w.roi_confirmed
    w.clear_roi()
    assert w.canvas.target is None and w.canvas.points == []
    w.close()
