"""Offline desktop UI; analysis and replay are separate paths."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import math

import cv2
from matplotlib.font_manager import FontProperties
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QImage
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QTableWidgetItem,
)
from src.config import ROOT, Settings, load_rois, validate_rois
from src.pipeline import AnalysisRunner
from src.storage import sha256, write_json
from src.ui.ai_worker import AIWorker
from src.ui.layout import build_window
from src.ui.presentation import shelf_profiles, time_text
from src.ui.presentation import VideoCanvas as VideoCanvas


def qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                QApplication.instance().setFont(QFont(families[0], 9))
        self.setWindowTitle("仓储作业复盘 · 本地视频与姿态事件")
        self.resize(1520, 960)
        self.worker = None
        self.session = None
        self.events = {}
        self.rows = {}
        self.video_hash = None
        self.loaded_video = None
        self.rois = {}
        self.roi_confirmed = False
        self.replay = None
        self.duration = 0.0
        self.playback_time = 0.0
        self.card_items = {}
        build_window(self)

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", str(ROOT), "视频 (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.video_edit.setText(path)
            self.preview()

    def toggle_regions(self, visible):
        self.region_panel.setVisible(visible)
        self.show_roi.setChecked(visible)

    def load_display(self, video_hash):
        try:
            self.canvas.outlines = shelf_profiles(video_hash)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.canvas.outlines = {}
            self.log.append(f"货架外观配置未加载：{exc}")
        self.layer_hint.setText(
            "实线仅为货架外观 · 判定区可单独显示"
            if self.canvas.outlines
            else "此视频无货架外观配置 · 可显示实际判定区"
        )
        self.canvas.update()

    def update_clock(self, seconds):
        self.playback_time = seconds
        self.timeline.position = seconds
        self.timeline.duration = self.duration
        self.timeline.update()
        self.clock_label.setText(f"{time_text(seconds)} / {time_text(self.duration)}")

    def select_event_card(self, current, previous=None):
        if current is not None:
            eid = current.data(Qt.ItemDataRole.UserRole)
            if eid in self.rows:
                self.table.selectRow(self.rows[eid])
        self.replay_btn.setEnabled(bool(current and self.session and not self.worker))

    def replay_selected(self):
        item = self.event_list.currentItem()
        if item is not None:
            eid = item.data(Qt.ItemDataRole.UserRole)
            if eid in self.rows:
                self.replay_event(self.rows[eid])

    def refresh_card(self, kind, event):
        eid = event["event_id"]
        if eid not in self.card_items:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, eid)
            item.setSizeHint(QSize(280, 94))
            self.event_list.insertItem(0, item)
            self.card_items[eid] = item
        item = self.card_items[eid]
        name = "弯腰候选" if event["event_type"] == "bend" else "伸手作业"
        region = {"left": "左侧货架", "right": "右侧货架"}.get(
            event["roi"], event["roi"] or "姿态观察"
        )
        state = "进行中" if not event["end_reason"] else "已结束"
        alert = " · 辅助提醒" if event["alert_time"] is not None else ""
        item.setText(
            f"T{event['track_id']:02d}  ·  {name}{alert}\n"
            f"{region}   {time_text(event['start_time'])} — {time_text(event['end_time'])}\n"
            f"有效 {event['effective_duration']:.1f} 秒  ·  {state}"
        )
        item.setForeground(QColor("#edbd62" if event["event_type"] == "bend" else "#a3dace"))
        source = self.session or getattr(getattr(self.worker, "runner", None), "session_path", None)
        if source and event.get("evidence_paths"):
            evidence = Path(source) / event["evidence_paths"][0]
            if evidence.exists():
                item.setIcon(QIcon(str(evidence)))
        self.filter_events()
        self.timeline.events = list(self.events.values())
        self.timeline.update()

    def filter_events(self):
        selected = {0: None, 1: "reach", 2: "bend"}[self.event_filter.currentIndex()]
        for eid, item in self.card_items.items():
            item.setHidden(selected is not None and self.events[eid]["event_type"] != selected)

    def reset_live_display(self, message="等待作业"):
        self.canvas.people = []
        self.canvas.clear_feedback()
        self.current_work.reset(message)

    def toggle_playback(self):
        if self.worker or not self.session:
            return
        if self.replay is not None:
            if self.timer.isActive():
                self.timer.stop()
                self.play_btn.setText("继续回放")
                self.status.setText("回放已暂停 · 不运行分析")
            else:
                self.timer.start()
                self.play_btn.setText("暂停回放")
                self.status.setText("原始视频回放 · 不运行分析")
        else:
            self.seek_video(0 if self.playback_time >= self.duration - 0.1 else self.playback_time)

    def open_replay(self, start, end):
        self.stop_replay()
        m = json.loads((self.session / "manifest.json").read_text(encoding="utf-8"))
        path = m["video_path"]
        if sha256(path) != m["video_sha256"]:
            raise ValueError("源视频哈希发生变化，拒绝错配回放")
        self.replay = cv2.VideoCapture(path)
        fps = self.replay.get(cv2.CAP_PROP_FPS)
        if not self.replay.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError("原始视频不可用或 FPS 无效")
        self.replay.set(cv2.CAP_PROP_POS_MSEC, max(0, start) * 1000)
        self.replay_end = end
        self.reset_live_display("原始视频回放\n\n回放不运行模型；当前作业高亮已关闭")
        self.canvas.rois = m["rois"] if self.show_roi.isChecked() else {}
        self.load_display(m["video_sha256"])
        self.status.setText("原始视频回放 · 不运行模型或增加计数")
        self.play_btn.setText("暂停回放")
        self.timer.start(max(1, round(1000 / fps)))

    def seek_video(self, seconds):
        if self.worker or not self.session:
            return
        try:
            self.open_replay(min(seconds, max(0, self.duration - 0.1)), self.duration)
        except Exception as exc:
            self.stop_replay()
            self.error(str(exc))

    def pick(self, edit, filter):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", str(ROOT), filter)
        if path:
            edit.setText(path)

    def error(self, text):
        self.log.append("错误：" + text)
        self.status.setText("出现错误，请查看日志")
        QMessageBox.warning(self, "无法完成", text)

    def pick_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择会话保存目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def preview(self):
        try:
            self.stop_replay()
            self.reset_live_display("等待作业\n\n预览并确认作业判定区域后开始分析")
            path = Path(self.video_edit.text()).resolve()
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
            cap.release()
            if not ok:
                raise ValueError("视频无法打开或首帧无法解码")
            self.duration = duration if math.isfinite(duration) else 0
            self.session = None
            self.events, self.rows, self.card_items = {}, {}, {}
            self.event_list.clear()
            self.table.setRowCount(0)
            self.timeline.events = []
            self.timeline.can_seek = False
            self.play_btn.setEnabled(False)
            self.replay_btn.setEnabled(False)
            self.stats.setText("画面人数 —\n作业事件 0\n姿态候选 0\n辅助提醒 0\n无法判断 —")
            self.draw_chart([], 0)
            self.update_clock(0)
            self.video_name.setText(path.name)
            self.video_name.setToolTip(str(path))
            self.video_hash = sha256(path)
            self.load_display(self.video_hash)
            self.loaded_video = str(path)
            self.canvas.target = None
            self.canvas.points = []
            self.canvas.set_image(qimage(frame))
            saved = ROOT / "data" / "roi_profiles" / f"{self.video_hash}.json"
            legacy = ROOT / "data/roi_config.json"
            default_video = ROOT / "data/video_1.mp4"
            self.rois = (
                load_rois(saved)
                if saved.exists()
                else (
                    load_rois(legacy)
                    if legacy.exists()
                    and default_video.exists()
                    and self.video_hash == sha256(default_video)
                    else {}
                )
            )
            self.canvas.rois = self.rois if self.show_roi.isChecked() else {}
            self.roi_confirmed = False
            self.region_btn.setChecked(True)
            self.status.setText("请核对虚线判定区域，再点击“确认判定区”")
            self.canvas.update()
        except Exception as exc:
            self.error(str(exc))

    def begin_roi(self, name):
        if self.canvas.image.isNull():
            self.error("请先预览视频")
            return
        self.show_roi.setChecked(True)
        self.canvas.target = name
        self.canvas.points = []
        self.roi_confirmed = False
        self.status.setText("沿区域边界依次点击 4 点；黑边点击无效")

    def roi_finished(self, name, pts):
        try:
            self.rois = validate_rois({**self.rois, name: pts})
            self.canvas.rois = self.rois if self.show_roi.isChecked() else {}
            self.canvas.update()
            self.status.setText("区域已绘制，请确认")
        except ValueError as exc:
            self.error(str(exc))

    def clear_roi(self):
        self.canvas.target = None
        self.canvas.points = []
        self.rois = {}
        self.canvas.rois = {}
        self.canvas.update()
        self.roi_confirmed = False

    def confirm_roi(self):
        try:
            if self.canvas.target is not None:
                raise ValueError("当前区域尚未画完4个点，请完成绘制或清除 ROI")
            if self.loaded_video != str(Path(self.video_edit.text()).resolve()):
                raise ValueError("视频已更换，请重新预览")
            if not self.video_hash:
                raise ValueError("请先预览")
            path = ROOT / "data/roi_profiles" / f"{self.video_hash}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, validate_rois(self.rois))
            self.roi_confirmed = True
            self.region_btn.setChecked(False)
            self.status.setText("ROI 已确认" if self.rois else "已确认无 ROI：只分析姿态")
        except Exception as exc:
            self.error(str(exc))

    def update_settings(self):
        self.canvas.rois = self.rois if self.show_roi.isChecked() else {}
        self.canvas.show_outlines = self.show_shelves.isChecked()
        self.canvas.show_skeleton = self.skeleton.isChecked()
        self.canvas.show_angles = self.angles.isChecked()
        self.canvas.update()
        if self.worker:
            self.worker.show_skeleton = self.skeleton.isChecked()
            self.worker.show_angles = self.angles.isChecked()
            self.worker.show_roi = False  # canvas draws ROI once

    def set_busy(self, busy):
        for w in [
            self.video_edit,
            self.model_edit,
            self.browse_video,
            self.browse_model,
            self.preview_btn,
            self.device,
            self.alert,
            self.draw_left,
            self.draw_right,
            self.confirm_btn,
            self.clear_btn,
            self.start_btn,
            self.load_session_btn,
            self.output_edit,
            self.output_btn,
            self.region_btn,
        ]:
            w.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.table.setEnabled(not busy)
        self.play_btn.setEnabled(not busy and self.session is not None)
        self.timeline.can_seek = not busy and self.session is not None
        self.replay_btn.setEnabled(
            not busy and self.session is not None and self.event_list.currentItem() is not None
        )

    def start_analysis(self):
        try:
            if not self.roi_confirmed or self.loaded_video != str(
                Path(self.video_edit.text()).resolve()
            ):
                raise ValueError("请先预览所选视频并确认 ROI")
            self.stop_replay()
            self.reset_live_display("正在加载模型…")
            settings = Settings.load(ROOT / "configs/default.json")
            settings.alert_seconds = self.alert.value()
            runner = AnalysisRunner(
                self.video_edit.text(),
                self.model_edit.text(),
                settings,
                self.rois,
                self.device.currentText(),
                output=self.output_edit.text(),
            )
            self.worker = AIWorker(runner)
            self.worker.frame_signal.connect(self.update_frame)
            self.worker.event_signal.connect(self.update_event)
            self.worker.log_signal.connect(self.log.append)
            self.worker.error_signal.connect(self.error)
            self.worker.result_signal.connect(self.analysis_result)
            self.worker.finished.connect(self.analysis_finished)
            self.session = None
            self.events = {}
            self.rows = {}
            self.card_items = {}
            self.event_list.clear()
            self.timeline.events = []
            self.table.setRowCount(0)
            self.chart_time = -1
            self.update_settings()
            self.set_busy(True)
            self.status.setText("初始化中")
            self.worker.start()
        except Exception as exc:
            self.error(str(exc))
            self.set_busy(False)

    def stop_analysis(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("正在停止并保存事件…")
            self.stop_btn.setEnabled(False)

    def analysis_finished(self):
        worker = self.worker
        self.worker = None
        self.reset_live_display("分析已结束\n\n从事件记录选择片段回放")
        self.set_busy(False)
        if worker:
            worker.deleteLater()

    def analysis_result(self, path):
        self.load_session(path)
        self.status.setText("分析已结束；双击事件回放")
        self.log.append(f"记录已保存：{path}")

    def update_frame(self, img, data):
        people = data.get("people_overlay", [])
        feedback = data.get("reach_feedback", [])
        self.canvas.set_image(img, data.get("processing_started"), people, feedback)
        self.current_work.set_feedback(img, people, feedback)
        for event in data.get("active_events", []):
            self.update_event("updated", event)
        self.update_clock(data["time"])
        c = data["counters"]
        self.status.setText(f"分析中 · 视频 {data['time']:.2f} 秒")
        self.stats.setText(
            f"画面人数 {c['people']}\n作业事件 {c['reach']}\n姿态候选 {c['bend']}\n辅助提醒 {c['alerts']}\n无法判断 {c['unknown']}"
        )
        if data["time"] - self.chart_time >= 1:
            self.draw_chart(data["timeline"], data["time"])
            self.chart_time = data["time"]

    def draw_chart(self, timeline, t):
        font = (
            FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
            if Path("C:/Windows/Fonts/msyh.ttc").exists()
            else FontProperties()
        )
        self.ax.clear()
        self.ax.set_facecolor("#18283a")
        self.ax.tick_params(colors="#b8c9d9")
        self.ax.set_title("每视频分钟的事件数", color="white", fontsize=10, fontproperties=font)
        bucket = int(t // 60)
        minutes = list(range(max(0, bucket - 5), bucket + 1))
        for event_type, color in [("reach", "#4fd1c5"), ("bend", "#ffc857")]:
            counts = [
                sum(int(s // 60) == m and k == event_type for s, k in timeline) for m in minutes
            ]
            self.ax.plot(
                [m + 0.5 for m in minutes],
                counts,
                "o-",
                label="伸手" if event_type == "reach" else "姿态候选",
                color=color,
            )
        self.ax.set_xticks([m + 0.5 for m in minutes], [f"{m}:00" for m in minutes])
        self.ax.set_xlim(minutes[0], minutes[-1] + 1)
        self.ax.set_ylim(bottom=0)
        self.ax.legend(facecolor="#18283a", labelcolor="white", prop=font)
        self.chart.figure.tight_layout()
        self.chart.draw_idle()

    def update_event(self, kind, event):
        eid = event["event_id"]
        self.events[eid] = event
        if eid not in self.rows:
            self.rows[eid] = self.table.rowCount()
            self.table.insertRow(self.rows[eid])
        reason = {
            "recovered": "姿态已恢复",
            "unobservable": "无法继续判断",
            "track_lost": "人物轨迹丢失",
            "eof": "视频结束",
            "range_end": "片段结束",
            "user_stop": "用户停止",
            "error": "运行错误",
            "timestamp_gap": "视频时间中断",
        }.get(event["end_reason"], event["end_reason"] or "进行中")
        vals = [
            f"{event['track_id']} / {event['roi'] or '—'}",
            "姿态候选" if event["event_type"] == "bend" else "作业事件",
            f"{event['start_time']:.2f}",
            f"{event['end_time']:.2f}",
            f"{event['effective_duration']:.2f}",
            f"{event['alert_time']:.2f}" if event["alert_time"] is not None else "—",
            reason,
        ]
        for col, v in enumerate(vals):
            self.table.setItem(self.rows[eid], col, QTableWidgetItem(v))
        self.refresh_card(kind, event)
        if kind == "alert":
            self.log.append(
                f"辅助提醒：人物 {event['track_id']}，视频 {event['alert_time']:.2f}s，待人工复核"
            )

    def load_session_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "选择会话目录", str(ROOT / "output/sessions"))
        if path:
            try:
                self.load_session(path)
            except Exception as exc:
                self.error(str(exc))

    def load_session(self, path):
        self.stop_replay()
        self.reset_live_display("历史会话\n\n选择事件，回放对应的原始视频")
        self.canvas.target = None
        self.canvas.points = []
        self.session = Path(path)
        manifest = json.loads((self.session / "manifest.json").read_text(encoding="utf-8"))
        if sha256(manifest["video_path"]) != manifest["video_sha256"]:
            raise ValueError("源视频内容与会话哈希不同，拒绝错配回放")
        self.events = {}
        self.rows = {}
        self.card_items = {}
        self.event_list.clear()
        self.table.setRowCount(0)
        for line in (self.session / "events.jsonl").read_text(encoding="utf-8").splitlines():
            self.update_event("ended", json.loads(line))
        events = list(self.events.values())
        self.stats.setText(
            f"画面人数 —\n作业事件 {sum(e['event_type'] == 'reach' for e in events)}"
            f"\n姿态候选 {sum(e['event_type'] == 'bend' for e in events)}"
            f"\n辅助提醒 {sum(e['alert_time'] is not None for e in events)}\n无法判断 —"
        )
        summary = json.loads((self.session / "summary.json").read_text(encoding="utf-8"))
        self.draw_chart(
            [(e["start_time"], e["event_type"]) for e in events],
            summary.get("last_video_time") or 0,
        )
        self.load_display(manifest["video_sha256"])
        self.video_name.setText(Path(manifest["video_path"]).name)
        cap = cv2.VideoCapture(manifest["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.duration = (
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            if fps > 0
            else (summary.get("last_video_time") or 0)
        )
        cap.release()
        self.update_clock(summary.get("last_video_time") or 0)
        self.play_btn.setEnabled(self.worker is None)
        self.timeline.can_seek = self.worker is None
        self.rois = manifest["rois"]
        if str(Path(manifest["video_path"]).resolve()) != self.loaded_video:
            self.loaded_video = None
            self.video_hash = None
            self.canvas.set_image(QImage())
        self.roi_confirmed = False
        self.update_settings()
        self.status.setText("已加载历史会话；双击事件回放")

    def replay_event(self, row, col=0):
        if self.worker or not self.session:
            return
        try:
            self.stop_replay()
            eid = next(k for k, v in self.rows.items() if v == row)
            e = self.events[eid]
            self.open_replay(max(0, e["start_time"] - 2), e["end_time"] + 2)
        except Exception as exc:
            self.stop_replay()
            self.error(str(exc))

    def replay_tick(self):
        if self.replay is None:
            return
        ok, frame = self.replay.read()
        if not ok or self.replay.get(cv2.CAP_PROP_POS_MSEC) / 1000 > self.replay_end:
            self.stop_replay()
            return
        self.canvas.set_image(qimage(frame))
        self.update_clock(self.replay.get(cv2.CAP_PROP_POS_MSEC) / 1000)

    def stop_replay(self):
        self.timer.stop()
        self.play_btn.setText("播放原片")
        if self.replay is not None:
            self.replay.release()
            self.replay = None
            self.status.setText("回放已结束 · 事件记录保持不变")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.stop_analysis()
            self.worker.finished.connect(self.close)
            event.ignore()
            return
        self.stop_replay()
        event.accept()


if __name__ == "__main__":
    from src.app import main

    raise SystemExit(main())
