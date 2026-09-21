import argparse
import json
import sys
from .config import ROOT, Settings, load_rois


def main(argv=None):
    p = argparse.ArgumentParser(description="仓储姿态辅助提醒：离线分析与人工核验评测")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--video", required=True)
    a.add_argument("--model", default=str(ROOT / "models/yolo11n-pose.pt"))
    a.add_argument("--config")
    a.add_argument("--roi")
    a.add_argument("--device", default="auto")
    a.add_argument("--output", default=str(ROOT / "output/sessions"))
    a.add_argument("--mode", choices=["baseline", "tracked", "stable"])
    a.add_argument("--start", type=float, default=0)
    a.add_argument("--end", type=float)
    e = sub.add_parser("evaluate")
    e.add_argument("--session", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--output", required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "analyze":
            from .pipeline import AnalysisRunner

            settings = Settings.load(args.config)
            if args.mode:
                settings.mode = args.mode
            runner = AnalysisRunner(
                args.video,
                args.model,
                settings,
                load_rois(args.roi),
                args.device,
                args.output,
                args.start,
                args.end,
            )
            print(runner.run(on_status=print))
        else:
            from .evaluation import evaluate

            print(
                json.dumps(
                    evaluate(args.session, args.labels, args.output), ensure_ascii=False, indent=2
                )
            )
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
