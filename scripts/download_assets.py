from __future__ import annotations
import argparse
from pathlib import Path
import urllib.request

# Large files are provided via GitHub Releases (not committed to the repo).
# - model: weights for inference
# - sample_video: input video used for quick start
# - ui_demo: screen-recorded UI demo video (for preview)
ASSETS = {
    "model": {
        "path": "models/yolo11n-pose.pt",
        "url": "https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/yolo11n-pose.pt",
        "desc": "YOLOv11 pose weights (baseline)",
    },
    "sample_video": {
        "path": "data/video_1.mp4",
        "url": "https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/video_1.mp4",
        "desc": "Sample input video for quick start",
    },
    "ui_demo": {
        "path": "output/ui_demo.mp4",
        "url": "https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/ui_demo.mp4",
        "desc": "UI demo video (screen recording)",
    },
}


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading -> {dst}")
    urllib.request.urlretrieve(url, dst)  # nosec


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download large assets (model / sample video / UI demo) from Releases."
    )
    p.add_argument("--model", action="store_true", help="Download model weights")
    p.add_argument("--video", action="store_true", help="Download sample input video (data/video_1.mp4)")
    p.add_argument("--ui-demo", action="store_true", help="Download UI demo video (output/ui_demo.mp4)")
    p.add_argument("--all", action="store_true", help="Download all assets")
    args = p.parse_args()

    targets: list[str] = []
    if args.all or (not args.model and not args.video and not args.ui_demo):
        targets = ["model", "sample_video", "ui_demo"]
    else:
        if args.model:
            targets.append("model")
        if args.video:
            targets.append("sample_video")
        if args.ui_demo:
            targets.append("ui_demo")

    for key in targets:
        url = ASSETS[key]["url"]
        if not url:
            print(f"[SKIP] {key}: url is empty. Please set it in scripts/download_assets.py")
            continue
        dst = Path(ASSETS[key]["path"])
        download(url, dst)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
