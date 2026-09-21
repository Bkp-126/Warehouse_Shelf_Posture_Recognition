"""Generate separate model proposals for all review clips without changing ground truth."""

import sys, json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import AnalysisRunner
from src.config import Settings
from src.storage import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ROOT / "output/review_bundle_v1")
    root = parser.parse_args().bundle.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    candidates = root / "candidates"
    candidates.mkdir(exist_ok=True)
    results = []
    for clip in manifest["clips"]:
        dst = candidates / f"{clip['clip_id']}.json"
        if dst.exists():
            results.append(json.loads(dst.read_text(encoding="utf-8")))
            continue
        # The long recording uses a different frame/camera context. Never transplant a ROI blindly.
        session = AnalysisRunner(
            clip["path"],
            ROOT / "models/yolo11n-pose.pt",
            Settings(),
            {},
            "cuda:0",
            root / "prediction_sessions",
        ).run()
        events = [
            json.loads(line)
            for line in (session / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        proposal = {
            "clip_id": clip["clip_id"],
            "status": "model_proposal_not_ground_truth",
            "session": str(session),
            "video_sha256": clip["sha256"],
            "events": events,
            "reach_status": "not_evaluated_roi_for_this_source_not_yet_confirmed",
            "notes": "弯腰候选不能自动认定为真实弯腰；下蹲、转身及遮挡需要人工核验。",
        }
        write_json(dst, proposal)
        results.append(proposal)
        print(clip["clip_id"], len(events), "candidates", flush=True)
    write_json(root / "candidate_index.json", {"status": "pending_human_review", "clips": results})


if __name__ == "__main__":
    main()
