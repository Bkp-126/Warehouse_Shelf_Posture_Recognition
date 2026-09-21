import cv2
import numpy as np
from .geometry import angle


def valid(k, indices, threshold):
    return all(np.isfinite(k[i]).all() and k[i][2] > threshold for i in indices)


def observe(people, rois, width, height, settings, machine):
    polygons = {
        n: np.array([(x * width, y * height) for x, y in p], np.float32) for n, p in rois.items()
    }
    observations, details = {}, []
    legacy_reach = legacy_bend = False
    for person in people:
        tid, k = person["track_id"], person["keypoints"]
        threshold = 0.5 if settings.mode == "baseline" else settings.keypoint_conf
        sides = []
        for side, idx in [("left", (5, 11, 13)), ("right", (6, 12, 14))]:
            if valid(k, idx, threshold):
                a = angle(*(k[i][:2] for i in idx))
                if a is not None:
                    sides.append((min(k[i][2] for i in idx), side, a))
        if settings.mode == "baseline":
            selected = None
            if valid(k, [6, 12, 14], 0.5):
                ba = k[6][:2] - k[12][:2]
                bc = k[14][:2] - k[12][:2]
                legacy_angle = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
                                -1,
                                1,
                            )
                        )
                    )
                )
                selected = (float(min(k[i][2] for i in (6, 12, 14))), "right", legacy_angle)
        else:
            selected = max(sides, default=None)
        a = selected[2] if selected else None
        limit = (
            settings.bend_off
            if settings.mode == "stable" and machine.active((tid, "bend", ""))
            else settings.bend_on
        )
        bent = None if a is None else bool(a < (140 if settings.mode == "baseline" else limit))
        if settings.mode == "baseline":
            legacy_bend |= bent is True
        else:
            observations[(tid, "bend", "")] = bent
        hits = []
        for name, poly in polygons.items():
            wrist_valid = [valid(k, [i], threshold) for i in (9, 10)]
            inside = any(
                ok and cv2.pointPolygonTest(poly, tuple(float(v) for v in k[i][:2]), False) > 0
                for i, ok in zip((9, 10), wrist_valid)
            )
            # A known hit suffices; if no hit and either wrist is missing we cannot conclude absence.
            reach = True if inside else (False if all(wrist_valid) else None)
            if inside:
                hits.append(name)
            if settings.mode == "baseline":
                legacy_reach |= inside
            else:
                observations[(tid, "reach", name)] = reach
        details.append(
            {
                "track_id": tid,
                "bbox": person["bbox"],
                "keypoints": k,
                "angle": a,
                "side": selected[1] if selected else None,
                "bend": bent,
                "rois": hits,
            }
        )
    if settings.mode == "baseline":
        observations = {
            (-1, "bend", ""): bool(legacy_bend),
            (-1, "reach", "any"): bool(legacy_reach),
        }
    return observations, details
