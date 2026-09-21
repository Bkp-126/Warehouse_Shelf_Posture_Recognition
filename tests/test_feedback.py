"""Presentation state must agree with, and never drive, the event machine."""

import copy
import numpy as np
from src.config import Settings
from src.events import EventMachine
from src.rules import observe
from src.feedback import reach_feedback

ROIS = {
    "left": [[0, 0], [0.49, 0], [0.49, 1], [0, 1]],
    "right": [[0.51, 0], [1, 0], [1, 1], [0.51, 1]],
}


def person(tid=1, x=20, confidence=1):
    k = np.zeros((17, 3), dtype=float)
    k[9] = [x, 40, confidence]
    k[10] = [x + 2, 50, confidence]
    return {"track_id": tid, "keypoints": k, "bbox": [5, 5, 95, 95]}


def step(machine, t, people):
    obs, details = observe(people, ROIS, 100, 100, machine.settings, machine)
    machine.update(t, obs)
    before = [e.to_dict() for e in machine.events]
    channels = copy.deepcopy(
        {k: (ch.effective, ch.last_t, ch.gap_start) for k, ch in machine.channels.items()}
    )
    feedback = reach_feedback(details, ROIS, 100, 100, machine.settings, machine, obs)
    assert [e.to_dict() for e in machine.events] == before
    assert {
        k: (ch.effective, ch.last_t, ch.gap_start) for k, ch in machine.channels.items()
    } == channels
    return details, feedback


def confirmed(machine, people):
    for t in (0, 0.1, 0.2, 0.3):
        details, feedback = step(machine, t, people)
    return details, feedback


def test_candidate_does_not_light_shelf_and_two_wrists_make_one_event():
    m = EventMachine("s", Settings())
    _, f = step(m, 0, [person()])
    assert f[0]["phase"] == "candidate" and f[0]["event_id"] is None
    assert len(m.events) == 0
    for t in (0.1, 0.2, 0.3):
        _, f = step(m, t, [person()])
    assert len(f) == 1 and f[0]["phase"] == "active" and len(f[0]["wrists"]) == 2
    assert len(m.events) == 1 and f[0]["duration"] == m.events[0].effective_duration


def test_two_people_regions_independent_and_departure_removes_hand():
    m = EventMachine("s", Settings())
    _, f = confirmed(m, [person(1, 20), person(2, 75)])
    assert {(x["track_id"], x["roi"]) for x in f if x["phase"] == "active"} == {
        (1, "left"),
        (2, "right"),
    }
    _, f = step(m, 0.4, [person(1, 49), person(2, 75)])
    left = next(x for x in f if x["track_id"] == 1)
    assert left["phase"] == "leaving" and not left["wrists"]
    assert any(x["phase"] == "active" and x["roi"] == "right" for x in f)


def test_occlusion_is_unknown_and_does_not_accumulate_duration():
    m = EventMachine("s", Settings())
    _, f = confirmed(m, [person()])
    duration = f[0]["duration"]
    _, f = step(m, 0.4, [person(confidence=0)])
    assert f[0]["phase"] == "unknown" and not f[0]["wrists"] and f[0]["duration"] == duration
    _, f = step(m, 0.61, [person(confidence=0)])
    assert all(x["phase"] == "unknown" and not x["event_id"] and not x["wrists"] for x in f)
    assert m.events[0].end_reason == "unobservable"


def test_lost_track_cannot_keep_a_hand_marker():
    m = EventMachine("s", Settings())
    confirmed(m, [person()])
    _, f = step(m, 0.4, [])
    assert f[0]["phase"] == "unknown" and not f[0]["wrists"]
    _, f = step(m, 0.61, [])
    assert not f and m.events[0].end_reason == "track_lost"


def test_one_valid_inside_wrist_suffices_but_boundary_does_not():
    m = EventMachine("s", Settings())
    p = person()
    p["keypoints"][10, 2] = 0
    _, f = confirmed(m, [p])
    assert f[0]["phase"] == "active" and len(f[0]["wrists"]) == 1
    p["keypoints"][9, 0] = 49
    _, f = step(m, 0.4, [p])
    assert f[0]["phase"] == "unknown" and not f[0]["wrists"]


def test_baseline_does_not_fabricate_confirmed_person_identity():
    m = EventMachine("s", Settings(mode="baseline"))
    _, f = confirmed(m, [person()])
    assert f[0]["phase"] == "candidate" and f[0]["event_id"] is None


def test_gui_clear_overlay_on_occlusion_and_raw_replay():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QImage
    from src.ui.presentation import VideoCanvas

    app = QApplication.instance() or QApplication([])
    m = EventMachine("s", Settings())
    details, f = confirmed(m, [person()])
    canvas = VideoCanvas()
    image = QImage(100, 100, QImage.Format.Format_RGB888)
    image.fill(0)
    canvas.set_image(image, people=details, feedback=f)
    assert canvas.active_regions == {"left"}
    details, f = step(m, 0.4, [person(confidence=0)])
    canvas.set_image(image, people=details, feedback=f)
    assert not canvas.active_regions and canvas._levels.get("left", 0) == 0
    canvas.set_image(image)
    assert not canvas.feedback and not canvas.people and not canvas._animation.isActive()
    canvas.close()
    app.processEvents()


def test_event_card_started_is_not_marked_ended_and_filter_works():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    event = dict(
        event_id="a",
        track_id=1,
        roi="left",
        event_type="reach",
        start_time=1,
        end_time=1.3,
        effective_duration=0.3,
        alert_time=None,
        end_reason="",
    )
    w.update_event("started", event)
    assert "进行中" in w.card_items["a"].text()
    w.event_filter.setCurrentIndex(2)
    assert w.card_items["a"].isHidden()
    event = {**event, "end_time": 2, "end_reason": "recovered"}
    w.update_event("ended", event)
    w.event_filter.setCurrentIndex(0)
    assert not w.card_items["a"].isHidden() and "已结束" in w.card_items["a"].text()
    w.close()
    app.processEvents()


def test_current_work_compacts_unconfirmed_unknown_people():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QImage
    from src.ui.presentation import CurrentWorkPanel

    app = QApplication.instance() or QApplication([])
    panel = CurrentWorkPanel()
    active = dict(track_id=1, roi="left", phase="active", wrists=[], duration=2.0, event_id="e1")
    unknown = dict(track_id=2, roi="", phase="unknown", wrists=[], duration=0.0, event_id=None)
    panel.set_feedback(QImage(), [], [active, unknown])
    assert set(panel.rows) == {(1, "left")}
    assert "T02" in panel.uncertain.text() and not panel.uncertain.isHidden()
    assert panel.height() <= 270
    panel.reset("回放")
    assert not panel.rows and panel.uncertain.isHidden()
    panel.close()
    app.processEvents()
