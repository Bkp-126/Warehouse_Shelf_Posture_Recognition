import json
import cv2
import numpy as np
import pytest
from src.pipeline import AnalysisRunner
from src.config import Settings


class FakeBackend:
    def infer(self, frame):
        k = np.zeros((17, 3), np.float32)
        k[:, 2] = 0.9
        k[5] = [10, 20, 0.9]
        k[11] = [10, 40, 0.9]
        k[13] = [30, 40, 0.9]
        k[6] = [14, 20, 0.9]
        k[12] = [14, 40, 0.9]
        k[14] = [34, 40, 0.9]
        k[9] = [20, 30, 0.9]
        k[10] = [21, 30, 0.9]
        return [
            {"track_id": 1, "keypoints": k, "bbox": [5, 5, 50, 60]},
            {"track_id": 2, "keypoints": k, "bbox": [5, 5, 50, 60]},
        ]


@pytest.fixture
def assets(tmp_path):
    video = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (80, 80))
    for i in range(60):
        writer.write(np.full((80, 80, 3), i, np.uint8))
    writer.release()
    model = tmp_path / "model.pt"
    model.write_bytes(b"test-double-not-weights")
    return video, model


def run(tmp_path, assets, callback=None):
    v, m = assets
    return AnalysisRunner(
        v,
        m,
        Settings(alert_seconds=0.5),
        {"left": [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.8]]},
        output=tmp_path / "sessions",
    ).run(on_frame=callback, backend=FakeBackend())


def canonical(path):
    rows = [json.loads(s) for s in (path / "events.jsonl").read_text().splitlines()]
    return [
        {k: v for k, v in r.items() if k not in ("event_id", "session_id", "evidence_paths")}
        for r in rows
    ]


def test_eof_unique_evidence_and_display_callback_parity(tmp_path, assets):
    frames = []
    first = run(tmp_path, assets)
    second = run(tmp_path, assets, lambda f, d, s: frames.append(s["time"]))
    assert canonical(first) == canonical(second)
    assert len(frames) == 60
    rows = [json.loads(s) for s in (first / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    paths = [p for r in rows for p in r["evidence_paths"]]
    assert len(paths) == len(set(paths)) == 6
    assert all((first / p).exists() for p in paths)
    assert all(r["end_reason"] == "eof" for r in rows)
    assert json.loads((first / "summary.json").read_text())["frames"] == 60


def test_missing_model_bad_video_and_unwritable_output(tmp_path, assets):
    v, m = assets
    with pytest.raises(FileNotFoundError):
        AnalysisRunner(v, tmp_path / "missing").run(backend=FakeBackend())
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"broken")
    with pytest.raises(ValueError):
        AnalysisRunner(bad, m).run(backend=FakeBackend())
    out = tmp_path / "not-a-directory"
    out.write_text("x")
    with pytest.raises(OSError):
        AnalysisRunner(v, m, output=out).run(backend=FakeBackend())


def test_user_stop_finishes_partial_event(tmp_path, assets):
    v, m = assets
    seen = []
    path = AnalysisRunner(v, m, output=tmp_path / "sessions").run(
        on_frame=lambda f, d, s: seen.append(s["time"]),
        cancel=lambda: len(seen) >= 20,
        backend=FakeBackend(),
    )
    rows = canonical(path)
    assert rows and all(e["end_reason"] == "user_stop" for e in rows)
    assert json.loads((path / "manifest.json").read_text())["status"] == "stopped"


def test_evaluate_rejects_unreviewed_and_duplicate_predictions(tmp_path, assets):
    from src.evaluation import evaluate
    from src.storage import write_json

    session = run(tmp_path, assets)
    manifest = json.loads((session / "manifest.json").read_text())
    labels = tmp_path / "labels.json"
    write_json(labels, {"review_status": "pending"})
    with pytest.raises(ValueError, match="拒绝评分"):
        evaluate(session, labels, tmp_path / "eval")
    events = [json.loads(s) for s in (session / "events.jsonl").read_text().splitlines()]
    gt = [
        {
            "person_id": str(e["track_id"]),
            "event_type": e["event_type"],
            "roi": e["roi"],
            "start_time": e["start_time"],
            "end_time": e["end_time"],
        }
        for e in events
    ]
    data = {
        "review_status": "verified",
        "reviewer": "test fixture only",
        "reviewed_at": "2026-09-21",
        "annotation_complete": True,
        "video_sha256": manifest["video_sha256"],
        "mapping_session_id": manifest["session_id"],
        "track_person_map": {"1": "1", "2": "2"},
        "evaluation_range": [0, 2.4],
        "events": gt,
    }
    write_json(labels, data)
    r = evaluate(session, labels, tmp_path / "eval")
    assert r["tp"] == 4 and r["fp"] == r["fn"] == 0 and r["alert_tp"] == 2
    with (session / "events.jsonl").open("a") as f:
        duplicate = dict(events[0])
        duplicate["event_id"] = "duplicate"
        f.write(json.dumps(duplicate) + "\n")
    r = evaluate(session, labels, tmp_path / "eval2")
    assert r["fp"] == 1 and r["duplicate_events"] == 1
    data["evaluated_event_types"] = ["bend"]
    data["events"] = [e for e in gt if e["event_type"] == "bend"]
    write_json(labels, data)
    r = evaluate(session, labels, tmp_path / "eval_bend_only")
    assert r["tp"] == 2 and r["evaluated_event_types"] == ["bend"]
    assert r["fn"] == 0


def test_image_write_failure_marks_session_failed(tmp_path, assets, monkeypatch):
    from src.storage import SessionStore

    def fail(*args):
        raise OSError("simulated disk full")

    monkeypatch.setattr(SessionStore, "evidence", fail)
    with pytest.raises(OSError, match="disk full"):
        run(tmp_path, assets)
    manifest = next((tmp_path / "sessions").glob("*/manifest.json"))
    assert json.loads(manifest.read_text())["status"] == "failed"


def test_alert_label_validates_all_intervals_even_after_threshold():
    from src.evaluation import derived_alert

    with pytest.raises(ValueError):
        derived_alert({"start_time": 0, "end_time": 20, "valid_intervals": [[0, 15], [12, 20]]}, 10)
