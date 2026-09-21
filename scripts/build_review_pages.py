"""Present unreviewed proposals separately, collapsed until the original video is viewed."""

import argparse
import html
import json
from pathlib import Path


def build(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for batch in range(1, 7):
        cards = []
        for clip in manifest["clips"]:
            if clip["batch"] != batch:
                continue
            cid = clip["clip_id"]
            proposal_path = bundle / "candidates" / f"{cid}.json"
            proposals = "模型初标尚未生成；这不表示没有动作。"
            if proposal_path.exists():
                proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
                rows = []
                for e in proposal["events"]:
                    evidence = ""
                    if e["evidence_paths"]:
                        relative = (
                            (Path(proposal["session"]) / e["evidence_paths"][0])
                            .relative_to(bundle)
                            .as_posix()
                        )
                        evidence = (
                            f'<a href="{html.escape(relative, quote=True)}">查看临时轨迹与骨架</a>'
                        )
                    rows.append(
                        f"<tr><td>{e['track_id']}</td><td>姿态候选</td><td>{e['start_time']:.2f}–{e['end_time']:.2f}</td><td>{e['effective_duration']:.2f}</td><td>{evidence}</td></tr>"
                    )
                proposals = (
                    (
                        "<table><tr><th>临时轨迹</th><th>模型输出</th><th>视频秒</th><th>有效秒</th><th>定位证据</th></tr>"
                        + "".join(rows)
                        + "</table>"
                    )
                    if rows
                    else "模型未生成姿态候选；不等于人工确认无事件。"
                )
            role = "开发片段" if clip["role"] == "development" else "同场景固定回归"
            cards.append(f"""<article><h2>{cid} · {role}</h2>
<p>原视频 {clip["source_start"]}–{clip["source_end"]} 秒。v2顺序解码修正版，窗口与划分未变。</p>
<video controls preload="metadata" src="clips/{cid}.mp4"></video>
<p>先核验原画面：人物、弯腰／伸手起止、是否可判断。人物可描述衣着和画面位置；时间按此片段0秒起算。</p>
<details><summary>看完原视频后展开模型初标（不是人工真值）</summary>{proposals}
<p>本视频未确认货架ROI，模型只分析姿态，不能据此评价伸手。临时轨迹编号不代表跨会话身份。</p></details>
<p><a href="labels/{cid}.json">独立人工标签文件</a> · 空标签或pending均不表示“没有事件”。</p></article>""")
        page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>仓储姿态人工复核</title>
<style>body{font:16px "Microsoft YaHei",sans-serif;background:#101c2a;color:#dfebf7;max-width:1080px;margin:30px auto;padding:0 16px}article{padding:22px;background:#1c3043;margin:26px 0;border-radius:10px}video{width:100%;max-height:650px}a{color:#67d7df}p{line-height:1.7}summary{cursor:pointer;color:#ffc857;padding:12px 0}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #446078;padding:10px;text-align:left}</style>"""
        page += f"<h1>人工复核第 {batch} 批 · 每批最多5段</h1><p>10秒是演示提醒规则。先看原视频，再展开模型初标；不要照抄模型结果。首次灰屏剪辑已修复，旧反馈保持单独留存。</p>"
        page += "".join(cards) + "</html>"
        (bundle / f"batch_{batch:02d}.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    build(parser.parse_args().bundle.resolve())
