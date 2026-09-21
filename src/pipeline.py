"""One analysis implementation for headless runs and Qt worker threads."""

from pathlib import Path
from time import perf_counter
import os
import math
import cv2
import numpy as np
from .config import ROOT, Settings, validate_rois
from .events import EventMachine
from .rules import observe
from .storage import SessionStore, sha256, environment


class VideoClock:
    def __init__(self, fps):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("视频 FPS 无效，无法可靠计算时间")
        self.fps = fps
        self.last = None
        self.fallback = 0

    def timestamp(self, frame_index, msec):
        t = msec / 1000
        if not math.isfinite(t) or t < 0 or (self.last is not None and t <= self.last):
            t = frame_index / self.fps
            if self.last is not None:
                t = max(t, self.last + 1 / self.fps)
            self.fallback += 1
        self.last = t
        return t


class PoseBackend:
    def __init__(self, model_path, device, settings):
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith(("cuda", "0")) and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境未检测到可用 GPU")
        self.model = YOLO(str(model_path), task="pose")
        self.device = device
        self.settings = settings
        self.hardware = {
            "device": device,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        self.model.predict(
            np.zeros((settings.imgsz, settings.imgsz, 3), np.uint8),
            device=device,
            imgsz=settings.imgsz,
            verbose=False,
            save=False,
        )

    def infer(self, frame):
        s = self.settings
        kwargs = dict(device=self.device, imgsz=s.imgsz, verbose=False, save=False)
        if s.mode == "baseline":
            result = self.model.predict(frame, conf=0.5, **kwargs)[0]
        else:
            result = self.model.track(
                frame,
                persist=True,
                conf=s.detection_conf,
                tracker=str(ROOT / "configs" / "bytetrack.yaml"),
                **kwargs,
            )[0]
        people = []
        if result.keypoints is None or result.boxes is None:
            return people
        k = result.keypoints.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None
        for i in range(len(k)):
            if s.mode != "baseline" and ids is None:
                continue
            people.append(
                {
                    "track_id": int(ids[i]) if ids is not None else i,
                    "keypoints": k[i],
                    "bbox": boxes[i].tolist(),
                }
            )
        return people


def render_frame(frame, details, rois, show_skeleton=True, show_angles=False, show_roi=True):
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    if show_roi:
        for name, pts in rois.items():
            poly = np.array([(int(x * w), int(y * h)) for x, y in pts], np.int32)
            cv2.polylines(canvas, [poly], True, (0, 210, 220), 2)
            cv2.putText(
                canvas, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 220), 2
            )
    links = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    for d in details:
        k = d["keypoints"]
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = (
            (0, 180, 255)
            if d["bend"]
            else ((140, 140, 140) if d["bend"] is None else (80, 210, 90))
        )
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        state = "UNKNOWN" if d["bend"] is None else ("BEND CANDIDATE" if d["bend"] else "OBSERVED")
        cv2.putText(
            canvas,
            f"ID {d['track_id']} {state}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        if show_skeleton:
            for a, b in links:
                if k[a][2] > 0.5 and k[b][2] > 0.5:
                    cv2.line(
                        canvas,
                        tuple(k[a][:2].astype(int)),
                        tuple(k[b][:2].astype(int)),
                        (220, 130, 70),
                        2,
                    )
        if show_angles and d["angle"] is not None:
            cv2.putText(
                canvas,
                f"{d['side']} hip {d['angle']:.1f}",
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
    return canvas


class AnalysisRunner:
    def __init__(
        self, video, model, settings=None, rois=None, device="auto", output=None, start=0, end=None
    ):
        self.video = Path(video).resolve()
        self.model = Path(model).resolve()
        self.settings = (settings or Settings()).validate()
        self.rois = validate_rois(rois or {})
        self.device = device
        self.output = Path(output or ROOT / "output" / "sessions")
        self.start = float(start)
        self.end = float(end) if end is not None else None
        if (
            not math.isfinite(self.start)
            or self.start < 0
            or (self.end is not None and (not math.isfinite(self.end) or self.end <= self.start))
        ):
            raise ValueError("视频时间范围无效")
        self.session_path = None

    def run(self, on_frame=None, on_event=None, on_status=None, cancel=None, backend=None):
        for name, path in [("视频", self.video), ("模型", self.model)]:
            if not path.is_file():
                raise FileNotFoundError(f"{name}不存在: {path}")
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {self.video}")
        store = None
        machine = None
        error = None
        reason = "eof"
        n = 0
        timeline = []
        timings = {
            name: []
            for name in [
                "decode",
                "inference_tracking",
                "rules",
                "render",
                "io",
                "total",
                "display_callback",
            ]
        }
        unknown = observable = 0.0
        last_t = None
        begin = perf_counter()
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            clock = VideoClock(fps)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = count / fps
            if self.start >= duration and count > 0:
                raise ValueError("开始时间超过视频长度")
            start_idx = round(self.start * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            manifest = {
                "video_path": str(self.video),
                "video_sha256": sha256(self.video),
                "model_path": str(self.model),
                "model_sha256": sha256(self.model),
                "settings": self.settings.to_dict(),
                "rois": self.rois,
                "environment": environment(),
                "video": {
                    "fps": fps,
                    "frame_count": count,
                    "width": int(cap.get(3)),
                    "height": int(cap.get(4)),
                },
                "range": {"start": self.start, "end": self.end},
                "time_basis": "decoder POS_MSEC; monotonic frame/fps fallback",
                "data_role": "unreviewed_development_or_fixed_regression_not_independent_validation",
            }
            store = SessionStore(self.output, manifest)
            self.session_path = store.path
            machine = EventMachine(store.id, self.settings)
            if on_status:
                on_status(f"加载模型；输出 {store.path}")
            backend = backend or PoseBackend(self.model, self.device, self.settings)
            store.manifest["hardware"] = getattr(backend, "hardware", {"device": "test_backend"})
            index = start_idx
            while True:
                if cancel and cancel():
                    reason = "user_stop"
                    break
                tick = perf_counter()
                ok, frame = cap.read()
                decode_ms = (perf_counter() - tick) * 1000
                if not ok:
                    if count > 0 and index < count - 1:
                        raise RuntimeError(f"视频在第 {index} 帧提前解码失败，预期 {count} 帧")
                    reason = "eof"
                    break
                t = clock.timestamp(index, cap.get(cv2.CAP_PROP_POS_MSEC))
                if self.end is not None and t >= self.end - 1e-8:
                    reason = "range_end"
                    break
                stamp = perf_counter()
                people = backend.infer(frame)
                infer_ms = (perf_counter() - stamp) * 1000
                stamp = perf_counter()
                obs, details = observe(
                    people, self.rois, frame.shape[1], frame.shape[0], self.settings, machine
                )
                channel_observations = {
                    key: obs.get(key) for key in set(machine.channels) | set(obs)
                }
                messages = machine.update(t, obs)
                rules_ms = (perf_counter() - stamp) * 1000
                stamp = perf_counter()
                needs_evidence = self.settings.save_evidence and any(
                    kind in ("started", "alert") for kind, _ in messages
                )
                evidence_frame = (
                    render_frame(frame, details, self.rois, show_angles=True)
                    if needs_evidence
                    else None
                )
                render_ms = (perf_counter() - stamp) * 1000
                dt = 0 if last_t is None else max(0, t - last_t)
                unknown += sum(v is None for v in channel_observations.values()) * dt
                observable += len(channel_observations) * dt
                stamp = perf_counter()
                for kind, ev in messages:
                    if evidence_frame is not None and kind in ("started", "alert"):
                        store.evidence(ev, evidence_frame, kind)
                    store.transition(kind, ev)
                    if kind == "started":
                        timeline.append((ev.start_time, ev.event_type))
                slim = [{k: v for k, v in d.items() if k != "keypoints"} for d in details]
                store.observation(
                    {
                        "frame_index": index,
                        "video_time": t,
                        "people": slim,
                        "channels": [
                            {"track_id": key[0], "type": key[1], "roi": key[2], "value": v}
                            for key, v in sorted(channel_observations.items())
                        ],
                    }
                )
                io_ms = (perf_counter() - stamp) * 1000
                counters = {
                    "people": len(people),
                    "reach": sum(e.event_type == "reach" for e in machine.events),
                    "bend": sum(e.event_type == "bend" for e in machine.events),
                    "alerts": sum(e.alert_time is not None for e in machine.events),
                    "unknown": sum(d["bend"] is None for d in details),
                }
                values = dict(
                    decode=decode_ms,
                    inference_tracking=infer_ms,
                    rules=rules_ms,
                    render=render_ms,
                    io=io_ms,
                    total=(perf_counter() - tick) * 1000,
                )
                for key, val in values.items():
                    timings[key].append(val)
                stamp = perf_counter()
                for kind, ev in messages:
                    if on_event:
                        on_event(kind, ev.to_dict())
                if on_frame:
                    on_frame(
                        frame,
                        details,
                        {
                            "time": t,
                            "counters": counters,
                            "timeline": timeline,
                            "rois": self.rois,
                            "processing_started": tick,
                        },
                    )
                timings["display_callback"].append((perf_counter() - stamp) * 1000)
                n += 1
                index += 1
                last_t = t
                if on_status and n % 250 == 0:
                    on_status(f"已处理 {n} 帧，视频时间 {t:.2f}s")
        except KeyboardInterrupt:
            reason = "user_stop"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            reason = "error"
            raise
        finally:
            cap.release()
            if store:
                for kind, ev in machine.close(reason):
                    store.transition(kind, ev)
                    if on_event:
                        on_event(kind, ev.to_dict())
                elapsed = perf_counter() - begin
                stats = {
                    k: {
                        "mean_ms": float(np.mean(v)),
                        "p50_ms": float(np.percentile(v, 50)),
                        "p95_ms": float(np.percentile(v, 95)),
                    }
                    for k, v in timings.items()
                    if v
                }
                summary = {
                    "frames": n,
                    "end_reason": reason,
                    "last_video_time": last_t,
                    "wall_seconds_including_setup": elapsed,
                    "throughput_fps_including_setup": n / elapsed if elapsed else 0,
                    "processing_fps": 1000 / np.mean(timings["total"])
                    if timings["total"]
                    else None,
                    "timings": stats,
                    "unknown_channel_seconds": unknown,
                    "total_channel_seconds": observable,
                    "unknown_coverage": unknown / observable if observable else None,
                    "events": len(machine.events),
                    "alerts": sum(e.alert_time is not None for e in machine.events),
                    "timestamp_fallback_frames": getattr(locals().get("clock"), "fallback", 0),
                    "accuracy_status": "pending_human_verified_labels",
                }
                store.finalize(
                    machine.events,
                    summary,
                    "failed" if error else ("stopped" if reason == "user_stop" else "completed"),
                    error,
                )
        return self.session_path
