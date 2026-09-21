"""Validated, serializable configuration shared by CLI and desktop."""

from dataclasses import dataclass, asdict, fields
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    mode: str = "stable"
    imgsz: int = 640
    detection_conf: float = 0.1
    keypoint_conf: float = 0.5
    bend_on: float = 140.0
    bend_off: float = 150.0
    enter_seconds: float = 0.3
    exit_seconds: float = 0.5
    missing_seconds: float = 0.2
    alert_seconds: float = 10.0
    gui_fps: float = 10.0
    save_evidence: bool = True

    def validate(self):
        if self.mode not in ("baseline", "tracked", "stable"):
            raise ValueError("mode 必须为 baseline / tracked / stable")
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name not in ("mode", "save_evidence") and (
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            ):
                raise ValueError(f"{f.name} 必须为有限数值")
        if not isinstance(self.save_evidence, bool):
            raise ValueError("save_evidence 必须为布尔值")
        if not isinstance(self.imgsz, int) or self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("imgsz 必须是正的 32 倍数")
        if not 0 < self.detection_conf <= 1 or not 0 < self.keypoint_conf <= 1:
            raise ValueError("置信度必须在 (0, 1]")
        if not 0 < self.bend_on < self.bend_off <= 180:
            raise ValueError("弯腰角度必须满足 0 < on < off <= 180")
        if (
            min(self.enter_seconds, self.exit_seconds, self.missing_seconds) < 0
            or min(self.alert_seconds, self.gui_fps) <= 0
        ):
            raise ValueError("持续时间或刷新率无效")
        return self

    @classmethod
    def load(cls, path=None):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"未知配置项: {sorted(unknown)}")
        return cls(**data).validate()

    def to_dict(self):
        return asdict(self)


def validate_rois(data):
    if not isinstance(data, dict):
        raise ValueError("ROI 必须为对象")
    result = {}
    for name, points in data.items():
        if not isinstance(name, str) or not name:
            raise ValueError("ROI 名称无效")
        if not points:
            continue
        if len(points) != 4 or any(len(p) != 2 for p in points):
            raise ValueError("每个 ROI 需要 4 个归一化坐标点")
        if any(
            not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
            for p in points
            for v in p
        ):
            raise ValueError("ROI 坐标必须在 [0,1]")
        # Require a consistently ordered convex quadrilateral, rejecting bow-ties.
        cross = []
        for i in range(4):
            a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
            cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if not (all(v > 1e-8 for v in cross) or all(v < -1e-8 for v in cross)):
            raise ValueError("ROI 必须按边界顺序绘制非退化凸四边形")
        result[name] = [[float(v) for v in p] for p in points]
    return result


def load_rois(path):
    return validate_rois(json.loads(Path(path).read_text(encoding="utf-8-sig"))) if path else {}
