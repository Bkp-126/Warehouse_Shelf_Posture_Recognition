from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
import cv2
from .config import ROOT


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    tmp.replace(path)


def environment():
    versions = {}
    for name in ["torch", "ultralytics", "opencv-python", "numpy", "PySide6", "matplotlib", "lap"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "-C",
                str(ROOT),
                "rev-parse",
                "HEAD",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        diff = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "diff", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        status = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "-C",
                str(ROOT),
                "status",
                "--porcelain",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        diff = b""
        status = "unavailable"
    source_hash = hashlib.sha256()
    for folder in ["src", "configs"]:
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".json", ".yaml"):
                source_hash.update(p.relative_to(ROOT).as_posix().encode())
                source_hash.update(p.read_bytes())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": commit,
        "git_status": status,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_sha256": source_hash.hexdigest(),
    }


class SessionStore:
    def __init__(self, output, manifest):
        self.id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
        self.path = Path(output).resolve() / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "images").mkdir()
        self.manifest = {
            **manifest,
            "session_id": self.id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "initializing",
            "notice": "10秒为可配置演示规则；姿态候选及提醒需人工复核。",
        }
        write_json(self.path / "manifest.json", self.manifest)
        self.observations = (self.path / "observations.jsonl").open("w", encoding="utf-8")
        self.journal = (self.path / "transitions.jsonl").open("w", encoding="utf-8")

    def evidence(self, event, frame, kind):
        name = f"images/{event.event_id}_{kind}.jpg"
        # imencode + tofile supports non-ASCII Windows paths.
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise OSError("证据图像编码失败")
        target = self.path / name
        with target.open("xb") as f:
            f.write(encoded.tobytes())
        event.evidence_paths.append(name)

    def transition(self, kind, event):
        self.journal.write(
            json.dumps({"transition": kind, **event.to_dict()}, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        self.journal.flush()

    def observation(self, data):
        self.observations.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")

    def finalize(self, events, summary, status, error=None):
        self.observations.close()
        self.journal.close()
        rows = [e.to_dict() for e in events]
        with (self.path / "events.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        fieldnames = [
            "session_id",
            "event_id",
            "track_id",
            "event_type",
            "roi",
            "start_time",
            "end_time",
            "effective_duration",
            "alert_time",
            "end_reason",
            "evidence_paths",
        ]
        with (self.path / "events.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row["evidence_paths"] = ";".join(row["evidence_paths"])
                writer.writerow(row)
        write_json(self.path / "summary.json", summary)
        self.manifest.update(status=status, error=error)
        write_json(self.path / "manifest.json", self.manifest)
