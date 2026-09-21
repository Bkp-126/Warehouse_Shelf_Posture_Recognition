"""Render a 75-second walkthrough from saved predictions, without rerunning models.
The recording is explicitly labeled offline replay; alert examples are not fabricated.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys, json, subprocess
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow, qimage
from src.storage import write_json


def main():
    import imageio_ffmpeg

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output/portfolio_demo")
    parser.add_argument("--comparison", type=Path, default=ROOT / "output/comparison_results.json")
    args = parser.parse_args()

    app = QApplication([])
    w = MainWindow()
    w.show()
    app.processEvents()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    selected = next(
        r for r in comparison["runs"] if r["mode"] == "stable" and r["model"] == "yolo11n-pose.pt"
    )
    session = Path(selected["session"])
    m = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    observations = [
        json.loads(line)
        for line in (session / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = [
        json.loads(line)
        for line in (session / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cap = cv2.VideoCapture(m["video_path"])
    fps = cap.get(5)
    frames = []
    # Decode once sequentially to avoid seek-related codec artifacts.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) % 5 == 0:
            frames.append(cv2.resize(frame, (960, 540)))
    cap.release()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "walkthrough.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    pipe = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1440x930",
            "-r",
            "10",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for i in range(750):
            wall = i / 10
            if wall < 8:
                video_t = 0.0
                title = "01 选择视频并确认货架 ROI（离线演示回放）"
            elif wall < 58:
                video_t = (wall - 8) * 58.44 / 50
                title = "02 根据保存的逐人预测回放 · 未运行模型"
            elif wall < 68:
                target = events[0] if events else {"start_time": 0, "end_time": 4}
                video_t = (
                    max(0, target["start_time"] - 2)
                    + (wall - 58) * min(10, target["end_time"] - target["start_time"] + 4) / 10
                )
                title = "03 事件前后回放 · 不增加计数"
            else:
                video_t = 32.0
                title = "04 失败边界：遮挡／投影角度需复核；无真实长弯腰提醒样例"
            frame = frames[min(len(frames) - 1, int(video_t * fps / 5))].copy()
            row = observations[min(len(observations) - 1, round(video_t * fps))]
            for person in row["people"]:
                x1, y1, x2, y2 = [int(v * 0.5) for v in person["bbox"]]
                color = (0, 200, 255) if person["bend"] else (120, 190, 140)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID {person['track_id']}",
                    (x1, max(12, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    1,
                )
            shown = events if wall >= 58 else [e for e in events if e["start_time"] <= video_t]
            counts = {
                "people": len(row["people"]),
                "reach": sum(e["event_type"] == "reach" for e in shown),
                "bend": sum(e["event_type"] == "bend" for e in shown),
                "alerts": sum(
                    e["alert_time"] is not None and (wall >= 58 or e["alert_time"] <= video_t)
                    for e in shown
                ),
                "unknown": sum(p["bend"] is None for p in row["people"]),
            }
            w.canvas.rois = m["rois"]
            w.update_frame(
                qimage(frame),
                {
                    "time": video_t,
                    "counters": counts,
                    "timeline": [(e["start_time"], e["event_type"]) for e in shown],
                },
            )
            w.status.setText(title)
            for e in shown:
                w.update_event("ended", e)
            if i == 0:
                w.log.append("本视频由已保存预测生成，用于说明流程；没有伪造操作、提醒或准确率。")
            if i == 580:
                w.log.append("回放路径与分析路径分开，已通过真实 Qt 工作线程一致性验证。")
            if i == 680:
                w.log.append("正式识别指标等待人工核验；新场景验证尚未开展。")
            app.processEvents()
            image = w.grab().toImage().convertToFormat(w.canvas.image.Format.Format_RGB888)
            if image.width() != 1440 or image.height() != 930:
                image = image.scaled(1440, 930)
            buf = np.frombuffer(image.bits(), np.uint8, count=image.sizeInBytes()).reshape(
                image.height(), image.bytesPerLine()
            )[:, : image.width() * 3]
            pipe.stdin.write(buf.tobytes())
            if i == 350:
                w.grab().save(str(out / "preview.png"))
        pipe.stdin.close()
        if pipe.wait() != 0:
            raise RuntimeError("演示编码失败")
    finally:
        if pipe.poll() is None:
            pipe.kill()
        w.close()
    write_json(
        out / "manifest.json",
        {
            "duration_seconds": 75,
            "fps": 10,
            "kind": "offline_walkthrough_rendered_from_saved_predictions",
            "source_session": str(session),
            "accuracy_claim": False,
            "real_prolonged_bend_alert_example": False,
        },
    )
    print(dest)


if __name__ == "__main__":
    main()
