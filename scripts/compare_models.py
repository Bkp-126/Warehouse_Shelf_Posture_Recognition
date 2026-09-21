"""Four full-video runs; descriptive predictions, never unreviewed accuracy."""

import sys, json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import Settings, load_rois
from src.pipeline import AnalysisRunner
from src.storage import write_json


def main():
    from ultralytics.utils.downloads import attempt_download_asset

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    out = parser.parse_args().output.resolve()

    model26 = ROOT / "models/yolo26n-pose.pt"
    if not model26.exists():
        # Explicit official release; downloads only, does not train or replace the baseline.
        attempt_download_asset(str(model26), repo="ultralytics/assets", release="v8.4.0")
    if not model26.exists():
        raise FileNotFoundError("YOLO26 权重下载失败")
    entries = []
    for mode, model in [
        ("baseline", "yolo11n-pose.pt"),
        ("tracked", "yolo11n-pose.pt"),
        ("stable", "yolo11n-pose.pt"),
        ("stable", "yolo26n-pose.pt"),
    ]:
        settings = Settings(mode=mode)
        path = AnalysisRunner(
            ROOT / "data/video_1.mp4",
            ROOT / "models" / model,
            settings,
            load_rois(ROOT / "data/roi_config.json"),
            "cuda:0",
            out / "comparison_sessions",
        ).run(on_status=print)
        summary = json.loads((path / "summary.json").read_text())
        entries.append({"mode": mode, "model": model, "session": str(path), "summary": summary})
        write_json(
            out / "comparison_results.json",
            {
                "status": "predictions_only_pending_human_review",
                "notes": [
                    "baseline为画面级规则；其身份与ROI语义不同，不与逐人指标混为一谈。",
                    "tracked使用逐人左右侧质量选择但不做时序平滑；stable增加时序和角度滞回。",
                    "baseline推理检测置信度固定0.5；跟踪组使用0.1向ByteTrack提供低分检测，因此该步是组合改动，不能把差异全部归因于跟踪。",
                    "全部为同场景已知示例，未使用这些结果调阈值。",
                ],
                "runs": entries,
            },
        )
    print(f"比较完成：{out / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
