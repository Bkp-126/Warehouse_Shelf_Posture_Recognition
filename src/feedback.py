"""Read-only presentation snapshots from the existing event machine.

This module never updates channels, counters, timestamps, or decision regions.
Wrists are located using exactly the same confidence and polygon test as rules.py.
"""

import cv2
import numpy as np
from .rules import valid


def reach_feedback(details, rois, width, height, settings, machine, observations):
    people = {d["track_id"]: d for d in details}
    polygons = {
        name: np.array([(x * width, y * height) for x, y in pts], np.float32)
        for name, pts in rois.items()
    }
    threshold = 0.5 if settings.mode == "baseline" else settings.keypoint_conf
    keys = {(tid, "reach", name) for tid in people for name in rois}
    keys |= {
        key
        for key, ch in machine.channels.items()
        if key[1] == "reach" and key[0] >= 0 and ch.event is not None
    }
    result = []
    covered = set()
    for key in sorted(keys):
        tid, _, name = key
        person = people.get(tid)
        channel = machine.channels.get(key)
        event = channel.event if channel is not None else None
        wrists = []
        if person is not None and name in polygons:
            for i in (9, 10):
                k = person["keypoints"]
                if valid(k, [i], threshold):
                    xy = tuple(float(v) for v in k[i][:2])
                    if cv2.pointPolygonTest(polygons[name], xy, False) > 0:
                        wrists.append(
                            {"index": i, "x": xy[0], "y": xy[1], "confidence": float(k[i][2])}
                        )
        value = observations.get(key)
        if settings.mode == "baseline" and wrists:
            value = True  # Legacy global events cannot be attributed to a person.
        if value is True and wrists:
            phase = "active" if event is not None else "candidate"
        elif event is not None:
            phase = "unknown" if value is None else "leaving"
            wrists = []  # Never attach an old hand position to a missing/outside wrist.
        else:
            continue
        covered.add(tid)
        result.append(
            {
                "track_id": tid,
                "roi": name,
                "phase": phase,
                "wrists": wrists,
                "duration": float(channel.effective) if channel is not None else 0.0,
                "event_id": event.event_id if event is not None else None,
            }
        )
    # If no active per-region event exists, report uncertainty once per person.
    # Baseline mode intentionally never fabricates a confirmed per-person event.
    for tid, person in people.items():
        if rois and tid not in covered and not valid(person["keypoints"], [9, 10], threshold):
            result.append(
                {
                    "track_id": tid,
                    "roi": "",
                    "phase": "unknown",
                    "wrists": [],
                    "duration": 0.0,
                    "event_id": None,
                }
            )
    return result
