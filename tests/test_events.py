import pytest
from src.config import Settings
from src.events import EventMachine
from src.geometry import widget_to_normalized, image_rect


def test_ten_second_boundary_once_and_two_people():
    m = EventMachine("test", Settings())
    messages = []
    for i in range(1101):
        messages += m.update(i / 100, {(1, "bend", ""): True, (2, "bend", ""): True})
    assert len(m.events) == 2
    assert [e.alert_time for e in m.events] == [10.0, 10.0]
    assert sum(k == "alert" for k, e in messages) == 2
    m.close("eof")
    assert all(e.end_reason == "eof" for e in m.events)


def test_missing_time_not_accumulated_and_no_duplicate():
    m = EventMachine("test", Settings())
    for i in range(102):
        m.update(i / 10, {(1, "bend", ""): None if i == 50 else True})
    assert len(m.events) == 1 and m.events[0].alert_time is None
    assert m.events[0].effective_duration == pytest.approx(9.9)
    m.update(10.2, {(1, "bend", ""): True})
    assert m.events[0].alert_time == pytest.approx(10.2)


@pytest.mark.parametrize("value,reason", [(None, "unobservable"), (False, "recovered")])
def test_long_gap_ends_at_last_positive(value, reason):
    m = EventMachine("test", Settings())
    for i in range(11):
        m.update(i / 10, {(1, "bend", ""): True})
    for i in range(11, 21):
        m.update(i / 10, {(1, "bend", ""): value})
    assert len(m.events) == 1 and m.events[0].end_time == 1
    assert m.events[0].end_reason == reason
    assert m.events[0].effective_duration == pytest.approx(1)


def test_single_frame_does_not_trigger_and_disappeared_track_closes():
    m = EventMachine("test", Settings())
    m.update(0, {(1, "reach", "left"): True})
    m.update(0.04, {(1, "reach", "left"): False})
    m.update(0.6, {})
    assert not m.events
    for i in range(20):
        m.update(1 + i * 0.04, {(2, "reach", "left"): True, (2, "reach", "right"): True})
    m.update(1.8, {})
    m.update(2.01, {})
    assert len(m.events) == 2 and all(e.end_reason == "track_lost" for e in m.events)


def test_roi_mapping_black_bars_and_resize():
    for ww, wh in [(800, 800), (1280, 720), (1441, 839), (640, 1080)]:
        x, y, w, h = image_rect(ww, wh, 1920, 1080)
        for u, v in [(0.01, 0.02), (0.5, 0.5), (0.99, 0.98)]:
            actual = widget_to_normalized(x + u * w, y + v * h, ww, wh, 1920, 1080)
            assert abs(actual[0] - u) * 1920 < 1 and abs(actual[1] - v) * 1080 < 1
        if y > 0:
            assert widget_to_normalized(ww / 2, y / 2, ww, wh, 1920, 1080) is None


def test_invalid_settings_and_roi():
    from src.config import validate_rois

    with pytest.raises(ValueError):
        Settings(alert_seconds=float("nan")).validate()
    with pytest.raises(ValueError):
        Settings(bend_on=160).validate()
    with pytest.raises(ValueError):
        validate_rois({"left": [[0, 0], [1, 1], [0, 1], [1, 0]]})


def test_isolated_candidate_pulses_never_accumulate_into_event():
    m = EventMachine("test", Settings())
    for i in range(100):
        m.update(i * 0.04, {(1, "bend", ""): i % 3 != 0})
    assert not m.events


def test_exit_hysteresis_only_applies_after_entry_confirmation():
    m = EventMachine("test", Settings())
    key = (1, "bend", "")
    m.update(0, {key: True})
    m.update(0.1, {key: True})
    assert not m.active(key)
    m.update(0.2, {key: True})
    m.update(0.3, {key: True})
    assert m.active(key)
