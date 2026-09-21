"""Rebuild the same frozen windows by sequential decoding; preserve v1 and all feedback."""

import json
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.storage import sha256, write_json


def decode_windows(source, clips, out):
    """Encode fixed 20-second windows in one sequential pass, retaining warnings."""
    if any((out / "clips").glob("*.mp4")):
        raise RuntimeError("拒绝覆盖已有剪辑")
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps != 25:
        raise ValueError("当前固定时间段重建要求已核查的25fps源视频")
    expressions = [
        f"between(n,{int(c['source_start'] * fps)},{int(c['source_end'] * fps) - 1})" for c in clips
    ]
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "warning",
        "-i",
        str(source),
        "-vf",
        "select='" + "+".join(expressions) + "',setpts=N/(25*TB)",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-r",
        "25",
        "-force_key_frames",
        "expr:gte(t,n_forced*20)",
        "-fps_mode",
        "cfr",
        "-f",
        "segment",
        "-segment_time",
        "20",
        "-segment_start_number",
        "1",
        "-reset_timestamps",
        "1",
        str(out / "clips/clip_%02d.mp4"),
    ]
    write_json(
        out / "rebuild_command.json",
        {
            "command": cmd,
            "reason": "v1 fast input seeking introduced visible gray reference-frame artifacts; sequential full-source decode",
        },
    )
    with (out / "decode_warnings.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=log)


def main():
    old = ROOT / "output/review_bundle_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output/review_bundle_v2")
    parser.add_argument(
        "--finalize-only", action="store_true", help="核查已编码的30段并补齐清单，不重新编码"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and not args.finalize_only:
        raise RuntimeError("不覆盖复核包；已存在时请核查日志与清单")
    manifest = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["source"])
    if sha256(source) != manifest["source_sha256"]:
        raise ValueError("源视频变化，拒绝重建")
    (out / "clips").mkdir(parents=True, exist_ok=args.finalize_only)
    (out / "labels").mkdir(exist_ok=args.finalize_only)
    if not args.finalize_only:
        decode_windows(source, manifest["clips"], out)
    if len(list((out / "clips").glob("*.mp4"))) != 30:
        raise ValueError("必须完整生成30段后才能登记清单")
    changes = []
    for clip in manifest["clips"]:
        cid = clip["clip_id"]
        existing_label = out / "labels" / f"{cid}.json"
        if (
            existing_label.exists()
            and json.loads(existing_label.read_text(encoding="utf-8")).get("review_status")
            == "verified"
        ):
            raise ValueError("拒绝覆盖已核验标签")
        path = out / "clips" / f"{cid}.mp4"
        cap = cv2.VideoCapture(str(path))
        count, actual_fps = cap.get(7), cap.get(5)
        cap.release()
        if count != 500 or actual_fps != 25:
            raise ValueError(f"{cid}: unexpected frames/fps {count}/{actual_fps}")
        old_hash = clip["sha256"]
        clip.update(path=str(path), sha256=sha256(path), prior_v1_sha256=old_hash)
        label = json.loads((old / "labels" / f"{cid}.json").read_text(encoding="utf-8"))
        label.update(
            review_status="pending",
            annotation_complete=False,
            video_sha256=clip["sha256"],
            reviewer="",
            reviewed_at="",
            mapping_session_id="",
            track_person_map={},
            events=[],
            normal_intervals=[],
        )
        label["notes"] = "v2顺序解码重建；v1原始人工反馈单独保留，未自动搬入新哈希的正式真值。"
        write_json(out / "labels" / f"{cid}.json", label)
        # Compare sparse aligned frames solely to flag changed visual content, never to select easier clips.
        a, b = cv2.VideoCapture(str(old / "clips" / f"{cid}.mp4")), cv2.VideoCapture(str(path))
        errors = []
        prior_decoded_frames = 0
        for i in range(500):
            oka, fa = a.read()
            okb, fb = b.read()
            if not okb:
                raise ValueError(f"{cid}: reconstructed video is truncated")
            prior_decoded_frames += int(oka)
            if i % 25 == 0 and oka:
                import numpy as np

                diff = cv2.resize(fa, (160, 90)).astype(np.float32) - cv2.resize(
                    fb, (160, 90)
                ).astype(np.float32)
                errors.append({"second": i / 25, "mse": float(np.mean(diff * diff))})
        a.release()
        b.release()
        changes.append(
            {
                "clip_id": cid,
                "aligned_pixel_changes": errors,
                "prior_decoded_frames_compared": prior_decoded_frames,
                "reconstructed_decoded_frames": 500,
                "note": "按帧号粗略对齐，旧片段可能少于500帧且有起始偏移；像素差异只提示复核，不是动作标签",
            }
        )
        print(cid, "verified 500 frames", flush=True)
    manifest.update(
        version=2,
        replaces_for_review="review_bundle_v1",
        repair="sequential decode; unchanged source time windows and data roles",
    )
    write_json(out / "manifest.json", manifest)
    write_json(out / "visual_change_audit.json", changes)
    for batch in range(1, 7):
        page = (old / f"batch_{batch:02d}.html").read_text(encoding="utf-8")
        page = page.replace(
            "<h1>",
            "<p>v2：修正快速定位引入的灰屏，同一原视频时间段和数据划分。原始反馈保留在 v1，不静默覆盖标签。</p><h1>",
        )
        (out / f"batch_{batch:02d}.html").write_text(page, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
