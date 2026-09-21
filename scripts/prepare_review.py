"""Create a frozen, same-source review bundle. Never auto-verify labels."""

import argparse
import sys, subprocess
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.storage import sha256, write_json
from scripts.repair_review import decode_windows


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="25fps source, at least 1385 seconds")
    parser.add_argument("--short", type=Path, help="Optional short video for approximate overlap check")
    parser.add_argument("--inventory-video", action="append", type=Path, default=[], help="Additional source inventory entry; repeatable")
    parser.add_argument("--output", type=Path, default=ROOT / "output/review_bundle_v1")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    short = args.short.resolve(strict=True) if args.short else None
    extra = [p.resolve(strict=True) for p in args.inventory_video]
    out = args.output.resolve()
    probe = cv2.VideoCapture(str(source))
    source_fps, source_frames = probe.get(cv2.CAP_PROP_FPS), probe.get(cv2.CAP_PROP_FRAME_COUNT)
    probe.release()
    if source_fps != 25 or source_frames < 1385 * 25:
        raise ValueError("固定采样需要至少1385秒、25fps的源视频；未创建复核包")
    if (out / "manifest.json").exists():
        raise RuntimeError("已冻结的复核包不覆盖；请使用已有产物")
    out.mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(5)
    total = cap.get(7)
    cap.release()
    source_hash = sha256(source)
    inventory = []
    for path in [source, *([short] if short else []), *extra]:
        inventory.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    # Approximate cross-video overlap check; a non-match is not proof of independence.
    queries = []
    if short:
        raw = subprocess.check_output(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(source),
                "-vf",
                "fps=0.5,scale=64:36",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ]
        )
        sampled = np.frombuffer(raw, np.uint8).reshape(-1, 36, 64).astype(np.float32)
        cap = cv2.VideoCapture(str(short))
        for t in [0, 20, 40]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok:
                q = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
                distances = np.mean((sampled - q) ** 2, axis=(1, 2))
                idx = int(np.argmin(distances))
                queries.append(
                    {"short_time": t, "nearest_long_time_approx": idx * 2, "mse": float(distances[idx])}
                )
        cap.release()
    write_json(
        out / "source_inventory.json",
        {
            "files": inventory,
            "approximate_overlap_probe": queries,
            "overlap_conclusion": "自动近邻仅供人工检查；未确认两段视频独立。短视频及其同哈希副本均排除本次冻结片段集。",
        },
    )
    clips = []
    for i in range(30):
        start = i * 45 if i < 20 else 960 + (i - 20) * 45
        end = start + 20
        if end > total / fps:
            raise ValueError("原始视频不足以生成预定片段")
        role = "development" if i < 20 else "frozen_same_scene_regression"
        cid = f"clip_{i + 1:02d}"
        path = out / "clips" / f"{cid}.mp4"
        if i == 0:
            windows = [
                {
                    "source_start": j * 45 if j < 20 else 960 + (j - 20) * 45,
                    "source_end": (j * 45 if j < 20 else 960 + (j - 20) * 45) + 20,
                }
                for j in range(30)
            ]
            decode_windows(source, windows, out)
        entry = {
            "clip_id": cid,
            "path": str(path),
            "sha256": sha256(path),
            "source_sha256": source_hash,
            "source_start": start,
            "source_end": end,
            "duration": 20,
            "role": role,
            "batch": i // 5 + 1,
            "coverage": "pending_human_review",
        }
        clips.append(entry)
        write_json(
            out / "labels" / f"{cid}.json",
            {
                "schema_version": 1,
                "clip_id": cid,
                "review_status": "pending",
                "reviewer": "",
                "reviewed_at": "",
                "annotation_complete": False,
                "video_sha256": entry["sha256"],
                "evaluation_range": [0, 20],
                "mapping_session_id": "",
                "track_person_map": {},
                "events": [],
                "normal_intervals": [],
                "notes": "空 events 表示尚未标注，不能解释成没有事件。模型初标另存 candidates。",
            },
        )
        print(f"{cid}: {role} source {start}-{end}", flush=True)
    manifest = {
        "version": 1,
        "source": str(source),
        "source_sha256": source_hash,
        "roles": "same_scene_only_not_independent_validation",
        "sampling": "30 fixed 20-second windows; no selection based on model score",
        "split_rule": "first 20 development; final 10 fixed regression; minimum cross-role gap 85 seconds",
        "clips": clips,
        "missing_coverage": [
            "持续弯腰",
            "下蹲",
            "遮挡",
            "多人交叉：待人工确认实际覆盖，不能从模型预测自行认定",
        ],
    }
    write_json(out / "manifest.json", manifest)
    for batch in range(1, 7):
        cards = []
        for c in clips:
            if c["batch"] == batch:
                cards.append(
                    f'<article><h2>{c["clip_id"]} · {c["role"]}</h2><p>原视频 {c["source_start"]}–{c["source_end"]} 秒。这里播放未叠加预测的原始片段。</p><video controls preload="metadata" src="clips/{c["clip_id"]}.mp4"></video><p>核验：人物编号、动作种类、开始／结束、遮挡／无法判断；不要把预测直接当答案。</p><p><a href="labels/{c["clip_id"]}.json">待填标签</a></p></article>'
                )
        page = (
            '<meta charset="utf-8"><title>人工复核批次</title><style>body{font:16px sans-serif;background:#101c2a;color:#dfebf7;max-width:1100px;margin:30px auto}video{width:100%}article{padding:20px;background:#1c3043;margin:25px 0}a{color:#67d7df}</style><h1>人工复核第 '
            + str(batch)
            + " 批 · 最多5段</h1><p>10秒为演示提醒阈值。先看原始视频，再修订初标；正式评分需要人工 verified。</p>"
            + "".join(cards)
        )
        (out / f"batch_{batch:02d}.html").write_text(page, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
