"""Explicit official/resource downloads, checked against the shipped SHA256 manifest."""

from pathlib import Path
import argparse, json, hashlib, urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--ui-demo", action="store_true")
    p.add_argument("--model26", action="store_true")
    args = p.parse_args()
    manifest = json.loads((ROOT / "configs/assets.json").read_text(encoding="utf-8"))
    names = [
        k for k in ["model", "video", "ui-demo", "model26"] if getattr(args, k.replace("-", "_"))
    ]
    if args.all:
        names = ["model", "video", "ui-demo"]
    if not names:
        names = ["model", "video"]
    for key in names:
        asset = manifest[key]
        path = ROOT / asset["path"]
        if path.exists():
            if sha(path) != asset["sha256"]:
                raise ValueError(f"已有文件哈希不同，不覆盖: {path}")
            print("verified", path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".download")
        if temp.exists():
            raise FileExistsError(f"发现未完成下载，请核查后处理: {temp}")
        try:
            with urllib.request.urlopen(asset["url"], timeout=60) as response, temp.open("xb") as f:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    f.write(block)
            if sha(temp) != asset["sha256"]:
                raise ValueError(f"下载 SHA256 不匹配，保留临时文件供核查: {temp}")
            temp.replace(path)
            print("downloaded and verified", path)
        except Exception as exc:
            raise RuntimeError(f"{key} 下载失败: {exc}") from exc


if __name__ == "__main__":
    main()
