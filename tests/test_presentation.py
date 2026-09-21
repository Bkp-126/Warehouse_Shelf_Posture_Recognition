import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import copy
import json
import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from src.ui.main_window import MainWindow
from src.ui.presentation import shelf_profiles
from src.config import ROOT, load_rois
from src.storage import sha256, write_json


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_display_layers_do_not_change_decision_rois(app):
    w = MainWindow()
    original = load_rois(ROOT / "data/roi_config.json")
    w.rois = copy.deepcopy(original)
    w.load_display("9a50b062eacaff7aff003106c1429c20cf074e67e7aed8ddcc735ea26a5e5949")
    assert w.canvas.outlines and w.canvas.outlines != w.rois
    w.show_roi.setChecked(True)
    assert w.canvas.rois == original
    w.show_roi.setChecked(False)
    assert not w.canvas.rois and w.rois == original
    w.show_shelves.setChecked(False)
    assert not w.canvas.show_outlines and w.rois == original
    w.load_display("a-different-video")
    assert w.canvas.outlines == {} and w.rois == original
    w.close()


def test_bad_display_profile_fails_without_changing_rules(app, tmp_path, monkeypatch):
    import src.ui.presentation as module

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/shelf_display.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    w = MainWindow()
    w.rois = {"fixture": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    w.load_display("example")
    assert not w.canvas.outlines
    assert "未加载" in w.log.toPlainText()
    assert w.rois["fixture"][1] == [1, 0]
    w.close()


def test_unknown_source_never_gets_demo_shelf_geometry():
    assert shelf_profiles("unknown") == {}


def test_raw_replay_seek_pause_and_cards_preserve_events(app, tmp_path):
    source = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90))
    assert writer.isOpened()
    for i in range(30):
        writer.write(np.full((90, 160, 3), i * 6, dtype=np.uint8))
    writer.release()
    session = tmp_path / "session"
    session.mkdir()
    write_json(
        session / "manifest.json",
        {"video_path": str(source), "video_sha256": sha256(source), "rois": {}},
    )
    write_json(session / "summary.json", {"last_video_time": 2.9})
    event = dict(
        event_id="e",
        track_id=1,
        roi="",
        event_type="bend",
        start_time=0.6,
        end_time=1.4,
        effective_duration=0.8,
        alert_time=None,
        end_reason="recovered",
    )
    (session / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    w = MainWindow()
    w.load_session(session)
    assert w.timeline.can_seek and w.event_list.count() == 1
    before = copy.deepcopy(w.events)
    w.canvas.people = [{"stale": True}]
    w.seek_video(0.5)
    w.replay_tick()
    assert not w.canvas.people and not w.canvas.image.isNull()
    assert 0.4 <= w.playback_time <= 0.7
    w.toggle_playback()
    assert not w.timer.isActive()
    w.toggle_playback()
    assert w.timer.isActive()
    w.event_list.setCurrentItem(w.card_items["e"])
    assert w.replay_btn.isEnabled()
    w.replay_selected()
    assert w.replay_end == 3.4
    w.stop_replay()
    assert w.events == before
    w.close()


def test_layer_controls_clear_people_on_new_image(app):
    w = MainWindow()
    w.skeleton.setChecked(False)
    w.angles.setChecked(True)
    assert not w.canvas.show_skeleton and w.canvas.show_angles
    w.canvas.people = [{"stale": True}]
    w.canvas.set_image(QImage(100, 100, QImage.Format.Format_RGB888))
    assert not w.canvas.people
    w.close()
