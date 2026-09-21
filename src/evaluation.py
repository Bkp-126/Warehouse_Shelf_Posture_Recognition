"""Fail-closed evaluation: never score unreviewed model-generated labels."""

from pathlib import Path
import json
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .storage import write_json


def tiou(a, b):
    overlap = max(0.0, min(a["end_time"], b["end_time"]) - max(a["start_time"], b["start_time"]))
    union = max(a["end_time"], b["end_time"]) - min(a["start_time"], b["start_time"])
    return overlap / union if union > 0 else 0.0


def validate_interval(start, end, lower, upper):
    if (
        any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in [start, end])
        or not lower <= start <= end <= upper
    ):
        raise ValueError("标签时间区间无效或超出评测范围")


def derived_alert(event, threshold):
    accumulated = 0.0
    last = event["start_time"]
    intervals = event.get("valid_intervals", [[event["start_time"], event["end_time"]]])
    for start, end in intervals:
        validate_interval(start, end, event["start_time"], event["end_time"])
        if start < last:
            raise ValueError("有效姿态区间重叠或乱序")
        last = end
    for start, end in intervals:
        if accumulated + end - start >= threshold - 1e-8:
            return start + threshold - accumulated
        accumulated += end - start
    return None


def evaluate(session, labels_path, output):
    session = Path(session)
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8-sig"))
    if (
        labels.get("review_status") != "verified"
        or not labels.get("reviewer")
        or not labels.get("reviewed_at")
        or labels.get("annotation_complete") is not True
    ):
        raise ValueError(
            "拒绝评分：需要人工 verified、reviewer、reviewed_at 和 annotation_complete=true"
        )
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("拒绝评分：会话未正常完成")
    if labels.get("video_sha256") != manifest["video_sha256"]:
        raise ValueError("标签与视频 SHA256 不匹配")
    if manifest["settings"]["mode"] == "baseline":
        raise ValueError("历史模式无人物身份和分区语义，仅作场景级计数对照，不混入逐人事件指标")
    lower, upper = labels["evaluation_range"]
    validate_interval(lower, upper, 0, manifest["video"]["frame_count"] / manifest["video"]["fps"])
    configured_end = (
        manifest["range"]["end"] or manifest["video"]["frame_count"] / manifest["video"]["fps"]
    )
    if lower < manifest["range"]["start"] or upper > configured_end + 1 / manifest["video"]["fps"]:
        raise ValueError("标签范围超出实际分析范围")
    mapping = labels.get("track_person_map", {})
    if labels.get("mapping_session_id") != manifest["session_id"]:
        raise ValueError("人物对应表必须核验并绑定当前会话编号")
    event_types = labels.get("evaluated_event_types", ["bend", "reach"])
    if (
        not isinstance(event_types, list)
        or not event_types
        or not set(event_types) <= {"bend", "reach"}
    ):
        raise ValueError("评测类型必须显式为 bend / reach 的非空列表")
    if "reach" in event_types and not manifest.get("rois"):
        raise ValueError("未设置货架 ROI，不能将伸手未分析解释为零预测；请明确仅评测 bend")
    truth = [dict(e) for e in labels["events"] if e["event_type"] in event_types]
    for e in truth:
        validate_interval(e["start_time"], e["end_time"], lower, upper)
        if not e.get("person_id") or e["event_type"] not in ("reach", "bend"):
            raise ValueError("标签人物或事件类型无效")
        e.setdefault("roi", "")
        e["alert_time"] = (
            derived_alert(e, manifest["settings"]["alert_seconds"])
            if e["event_type"] == "bend"
            else None
        )
    pred = []
    for line in (session / "events.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        if p["event_type"] not in event_types:
            continue
        if p["end_time"] < lower or p["start_time"] > upper:
            continue
        if p["start_time"] < lower or p["end_time"] > upper:
            raise ValueError("评测边界截断事件：请核验完整区间，避免裁剪影响持续时长与提醒")
        if str(p["track_id"]) not in mapping:
            raise ValueError(f"轨迹 {p['track_id']} 缺少人工人物对应；不能静默丢弃预测")
        p["person_id"] = mapping[str(p["track_id"])]
        pred.append(p)
    scores = np.zeros((len(pred), len(truth)))
    compatible = np.zeros_like(scores, dtype=bool)
    for i, p in enumerate(pred):
        for j, g in enumerate(truth):
            same = all(p.get(k, "") == g.get(k, "") for k in ["person_id", "event_type", "roi"])
            compatible[i, j] = same
            if same:
                scores[i, j] = tiou(p, g)
    pairs = []
    if len(pred) and len(truth):
        # Maximize number of valid matches before maximizing total IoU.
        weights = (scores >= 0.5) * (min(len(pred), len(truth)) + 1) + scores
        ii, jj = linear_sum_assignment(-weights)
        pairs = [
            (int(i), int(j)) for i, j in zip(ii, jj) if compatible[i, j] and scores[i, j] >= 0.5
        ]
    matched_p = {i for i, j in pairs}
    matched_g = {j for i, j in pairs}
    tp = len(pairs)
    fp = len(pred) - tp
    fn = len(truth) - tp
    duplicates = sum(
        i not in matched_p and any(compatible[i, j] and scores[i, j] >= 0.5 for j in matched_g)
        for i in range(len(pred))
    )
    alert_tp = sum(
        pred[i]["alert_time"] is not None and truth[j]["alert_time"] is not None for i, j in pairs
    )
    alert_pred = sum(p["alert_time"] is not None for p in pred)
    alert_gt = sum(g["alert_time"] is not None for g in truth)
    normal = labels.get("normal_intervals", [])
    normal_seconds = 0.0
    last = lower
    for a, b in normal:
        validate_interval(a, b, lower, upper)
        if a < last:
            raise ValueError("正常区间必须有序且不重叠")
        if any(min(b, g["end_time"]) > max(a, g["start_time"]) for g in truth):
            raise ValueError("正常区间与真实事件重叠")
        normal_seconds += b - a
        last = b
    normal_fp = sum(
        i not in matched_p and any(a <= p["start_time"] < b for a, b in normal)
        for i, p in enumerate(pred)
    )
    unknown = total = 0.0
    last_t = None
    for line in (session / "observations.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        t = row["video_time"]
        if lower <= t <= upper:
            dt = 0 if last_t is None else t - last_t
            channels = [c for c in row["channels"] if c["type"] in event_types]
            total += len(channels) * dt
            unknown += sum(c["value"] is None for c in channels) * dt
            last_t = t
    result = {
        "status": "human_verified_same_scene",
        "data_role": labels.get("data_role", "unspecified_same_scene"),
        "evaluated_event_types": event_types,
        "annotation_time_resolution_seconds": labels.get("annotation_time_resolution_seconds"),
        "tiou_threshold": 0.5,
        "session_id": manifest["session_id"],
        "evaluation_range": [lower, upper],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / len(pred) if pred else None,
        "recall": tp / len(truth) if truth else None,
        "duplicate_events": int(duplicates),
        "duplicate_fraction_of_predictions": duplicates / len(pred) if pred else None,
        "start_mae_seconds": float(
            np.mean([abs(pred[i]["start_time"] - truth[j]["start_time"]) for i, j in pairs])
        )
        if pairs
        else None,
        "end_mae_seconds": float(
            np.mean([abs(pred[i]["end_time"] - truth[j]["end_time"]) for i, j in pairs])
        )
        if pairs
        else None,
        "alert_tp": alert_tp,
        "alert_fp": alert_pred - alert_tp,
        "alert_fn": alert_gt - alert_tp,
        "normal_seconds": normal_seconds,
        "false_events_in_normal": normal_fp,
        "false_events_per_hour_normal": normal_fp * 3600 / normal_seconds
        if normal_seconds
        else None,
        "unknown_channel_coverage": unknown / total if total else None,
        "matches": [
            {"prediction": pred[i]["event_id"], "truth_index": j, "tiou": float(scores[i, j])}
            for i, j in pairs
        ],
        "unmatched_prediction_ids": [
            p["event_id"] for i, p in enumerate(pred) if i not in matched_p
        ],
        "unmatched_truth_indices": [j for j in range(len(truth)) if j not in matched_g],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", result)
    return result
