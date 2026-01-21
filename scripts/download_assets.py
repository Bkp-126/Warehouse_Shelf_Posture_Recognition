from __future__ import annotations
import argparse
from pathlib import Path
import sys
import urllib.request


ASSETS = {
    # 你后面会把实际链接填进来（GitHub Release / 网盘直链 / HuggingFace）
    "model": {
        "path": "models/yolo11n-pose.pt",
        "url": "https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/yolo11n-pose.pt",
        "desc": "YOLOv11 pose weights",
    },
    "video": {
        "path": "data/video_1.mp4",
        "url": "https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/video_1.mp4",
        "desc": "Demo video for quick start",
    },
}


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading -> {dst}")
    urllib.request.urlretrieve(url, dst)  # nosec


def main() -> int:
    p = argparse.ArgumentParser(description="Download large assets (model/video) for this repo.")
    p.add_argument("--model", action="store_true", help="Download model weights")
    p.add_argument("--video", action="store_true", help="Download demo video")
    p.add_argument("--all", action="store_true", help="Download all assets")
    args = p.parse_args()

    targets = []
    if args.all or (not args.model and not args.video):
        targets = ["model", "video"]
    else:
        if args.model:
            targets.append("model")
        if args.video:
            targets.append("video")

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
