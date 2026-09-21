import numpy as np
from src.config import Settings
from src.events import EventMachine
from src.rules import observe


def person(tid=1):
    k = np.zeros((17, 3))
    k[5] = [20, 20, 0.9]
    k[11] = [20, 40, 0.9]
    k[13] = [40, 40, 0.9]
    k[9] = [30, 30, 0.9]
    k[10] = [31, 31, 0.9]
    return {"track_id": tid, "keypoints": k, "bbox": [10, 10, 60, 60]}


def test_left_side_survives_right_occlusion_and_two_hands_one_roi():
    s = Settings()
    m = EventMachine("s", s)
    obs, details = observe([person()], {"left": [[0, 0], [1, 0], [1, 1], [0, 1]]}, 100, 100, s, m)
    assert obs[(1, "bend", "")] is True
    assert obs[(1, "reach", "left")] is True
    assert len(obs) == 2 and details[0]["side"] == "left"


def test_baseline_preserves_right_only_and_scene_edge_semantics():
    s = Settings(mode="baseline")
    m = EventMachine("s", s)
    obs, _ = observe(
        [person(1), person(2)], {"left": [[0, 0], [1, 0], [1, 1], [0, 1]]}, 100, 100, s, m
    )
    assert obs == {(-1, "bend", ""): False, (-1, "reach", "any"): True}


def test_degenerate_points_are_unknown_in_stable_mode():
    p = person()
    p["keypoints"][5] = p["keypoints"][11]
    s = Settings()
    obs, _ = observe([p], {}, 100, 100, s, EventMachine("s", s))
    assert obs[(1, "bend", "")] is None
