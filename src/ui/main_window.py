"""Offline desktop UI; analysis and replay are separate paths."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import cv2
from time import perf_counter
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QTextEdit,
    QSplitter,
    QHeaderView,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from src.config import ROOT, Settings, load_rois, validate_rois
from src.geometry import image_rect, widget_to_normalized
from src.storage import sha256, write_json
from src.pipeline import AnalysisRunner
from src.ui.ai_worker import AIWorker


class VideoCanvas(QWidget):
    roi_finished = Signal(str, list)
    painted_latency = Signal(float)

    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.rois = {}
        self.points = []
        self.target = None
        self.frame_origin = None
        self.setMinimumSize(480, 270)

    def set_image(self, image, origin=None):
        self.image = image
        self.frame_origin = origin
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0a1018"))
        if self.image.isNull():
            p.setPen(QColor("#a8bbc9"))
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "选择视频并预览，然后设置货架区域"
            )
            return
        x, y, w, h = image_rect(
            self.width(), self.height(), self.image.width(), self.image.height()
        )
        from PySide6.QtCore import QRectF, QPointF

        p.drawImage(QRectF(x, y, w, h), self.image)
        p.setPen(QPen(QColor("#31d7cf"), 2))
        for name, points in self.rois.items():
            pts = [QPointF(x + a * w, y + b * h) for a, b in points]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                p.drawLine(a, b)
            p.drawText(pts[0], name)
        p.setPen(QPen(QColor("#ffc857"), 3))
        for a, b in self.points:
            p.drawEllipse(QPointF(x + a * w, y + b * h), 4, 4)
        p.end()
        if self.frame_origin is not None:
            self.painted_latency.emit((perf_counter() - self.frame_origin) * 1000)
            self.frame_origin = None

    def mousePressEvent(self, event):
        if self.target is None or self.image.isNull():
            return
        pos = event.position()
        pt = widget_to_normalized(
            pos.x(), pos.y(), self.width(), self.height(), self.image.width(), self.image.height()
        )
        if pt is None:
            return
        self.points.append(pt)
        self.update()
        if len(self.points) == 4:
            target, points = self.target, self.points
            self.target = None
            self.points = []
            self.roi_finished.emit(target, points)


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
        self.setWindowTitle("仓储作业姿态风险辅助提醒 · 作品集首版")
        self.resize(1440, 930)
        self.worker = None
        self.session = None
        self.events = {}
        self.rows = {}
        self.video_hash = None
        self.loaded_video = None
        self.rois = {}
        self.roi_confirmed = False
        self.replay = None
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#111b29;color:#dde7ef;} QLineEdit,QComboBox,QDoubleSpinBox,QTextEdit,QTableWidget{background:#1c2c3f;padding:4px;} QPushButton{background:#235a76;padding:8px;border-radius:4px;} QPushButton:disabled{background:#253440;color:#63768a;} QHeaderView::section{background:#23445b;padding:5px;}"
        )
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        header = QLabel("仓储姿态辅助提醒  |  可回放的作业与姿态事件")
        header.setStyleSheet("font-size:22px;font-weight:bold;padding:8px;")
        outer.addWidget(header)
        notice = QLabel(
            "10 秒是可配置演示规则，不是医学、工效或合规标准。姿态候选与提醒均需人工复核。"
        )
        notice.setStyleSheet("color:#ffc857;padding:8px;background:#283447;")
        outer.addWidget(notice)
        paths = QHBoxLayout()
        self.video_edit = QLineEdit(str(ROOT / "data/video_1.mp4"))
        self.model_edit = QLineEdit(str(ROOT / "models/yolo11n-pose.pt"))
        self.browse_video = QPushButton("选择视频")
        self.browse_model = QPushButton("选择模型")
        self.preview_btn = QPushButton("预览 / 加载 ROI")
        paths.addWidget(self.video_edit, 3)
        paths.addWidget(self.browse_video)
        paths.addWidget(self.model_edit, 2)
        paths.addWidget(self.browse_model)
        paths.addWidget(self.preview_btn)
        outer.addLayout(paths)
        self.browse_video.clicked.connect(
            lambda: self.pick(self.video_edit, "视频 (*.mp4 *.avi *.mov *.mkv)")
        )
        self.browse_model.clicked.connect(lambda: self.pick(self.model_edit, "模型 (*.pt)"))
        self.preview_btn.clicked.connect(self.preview)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("会话保存目录"))
        self.output_edit = QLineEdit(str(ROOT / "output/sessions"))
        self.output_btn = QPushButton("选择保存目录")
        output_row.addWidget(self.output_edit)
        output_row.addWidget(self.output_btn)
        outer.addLayout(output_row)
        self.output_btn.clicked.connect(self.pick_output)
        controls = QHBoxLayout()
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda:0"])
        self.alert = QDoubleSpinBox()
        self.alert.setRange(0.1, 3600)
        self.alert.setValue(10)
        self.alert.setSuffix(" 秒提醒")
        controls.addWidget(QLabel("设备"))
        controls.addWidget(self.device)
        controls.addWidget(self.alert)
        self.draw_left = QPushButton("画左货架")
        self.draw_right = QPushButton("画右货架")
        self.confirm_btn = QPushButton("确认当前 ROI")
        self.clear_btn = QPushButton("清除 ROI")
        for b in [self.draw_left, self.draw_right, self.confirm_btn, self.clear_btn]:
            controls.addWidget(b)
        self.draw_left.clicked.connect(lambda: self.begin_roi("left"))
        self.draw_right.clicked.connect(lambda: self.begin_roi("right"))
        self.confirm_btn.clicked.connect(self.confirm_roi)
        self.clear_btn.clicked.connect(self.clear_roi)
        self.skeleton = QCheckBox("骨架")
        self.skeleton.setChecked(True)
        self.angles = QCheckBox("角度")
        self.show_roi = QCheckBox("显示 ROI")
        self.show_roi.setChecked(True)
        for c in [self.skeleton, self.angles, self.show_roi]:
            controls.addWidget(c)
            c.toggled.connect(self.update_settings)
        self.start_btn = QPushButton("开始分析")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        outer.addLayout(controls)
        self.start_btn.clicked.connect(self.start_analysis)
        self.stop_btn.clicked.connect(self.stop_analysis)
        split = QSplitter()
        outer.addWidget(split, 1)
        self.canvas = VideoCanvas()
        self.canvas.roi_finished.connect(self.roi_finished)
        split.addWidget(self.canvas)
        right = QWidget()
        panel = QVBoxLayout(right)
        self.status = QLabel("就绪：先预览并确认 ROI")
        panel.addWidget(self.status)
        self.stats = QLabel("画面人数 0\n作业事件 0\n姿态候选 0\n辅助提醒 0\n无法判断 0")
        self.stats.setStyleSheet("font-size:19px;line-height:160%;padding:12px;")
        panel.addWidget(self.stats)
        fig = Figure(figsize=(4, 2), facecolor="#111b29")
        self.chart = FigureCanvasQTAgg(fig)
        self.ax = fig.add_subplot(111)
        self.chart_time = -1
        panel.addWidget(self.chart)
        self.draw_chart([], 0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(250)
        panel.addWidget(self.log)
        split.addWidget(right)
        split.setSizes([1000, 350])
        row = QHBoxLayout()
        row.addWidget(QLabel("事件列表：双击回放，前后各 2 秒；回放不会新增统计。"))
        self.load_session_btn = QPushButton("打开历史会话")
        self.load_session_btn.clicked.connect(self.load_session_dialog)
        row.addWidget(self.load_session_btn)
        outer.addLayout(row)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["人物 / 区域", "事件", "开始(s)", "结束(s)", "有效时长(s)", "提醒(s)", "结束原因"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(220)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self.replay_event)
        outer.addWidget(self.table)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.replay_tick)

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
            path = Path(self.video_edit.text()).resolve()
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise ValueError("视频无法打开或首帧无法解码")
            self.video_hash = sha256(path)
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
            self.status.setText("请核对货架位置并点击“确认当前 ROI”")
            self.canvas.update()
        except Exception as exc:
            self.error(str(exc))

    def begin_roi(self, name):
        if self.canvas.image.isNull():
            self.error("请先预览视频")
            return
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
            self.status.setText("ROI 已确认" if self.rois else "已确认无 ROI：只分析姿态")
        except Exception as exc:
            self.error(str(exc))

    def update_settings(self):
        self.canvas.rois = self.rois if self.show_roi.isChecked() else {}
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
        ]:
            w.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.table.setEnabled(not busy)

    def start_analysis(self):
        try:
            if not self.roi_confirmed or self.loaded_video != str(
                Path(self.video_edit.text()).resolve()
            ):
                raise ValueError("请先预览所选视频并确认 ROI")
            self.stop_replay()
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
            self.events = {}
            self.rows = {}
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
        self.set_busy(False)
        if worker:
            worker.deleteLater()

    def analysis_result(self, path):
        self.load_session(path)
        self.status.setText("分析已结束；双击事件回放")
        self.log.append(f"记录已保存：{path}")

    def update_frame(self, img, data):
        self.canvas.set_image(img, data.get("processing_started"))
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
        self.canvas.target = None
        self.canvas.points = []
        self.session = Path(path)
        manifest = json.loads((self.session / "manifest.json").read_text(encoding="utf-8"))
        if sha256(manifest["video_path"]) != manifest["video_sha256"]:
            raise ValueError("源视频内容与会话哈希不同，拒绝错配回放")
        self.events = {}
        self.rows = {}
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
            m = json.loads((self.session / "manifest.json").read_text(encoding="utf-8"))
            path = m["video_path"]
            if sha256(path) != m["video_sha256"]:
                raise ValueError("源视频哈希发生变化，拒绝错配回放")
            self.replay = cv2.VideoCapture(path)
            if not self.replay.isOpened():
                raise ValueError("原始视频不可用")
            fps = self.replay.get(cv2.CAP_PROP_FPS)
            self.replay.set(cv2.CAP_PROP_POS_MSEC, max(0, e["start_time"] - 2) * 1000)
            self.replay_end = e["end_time"] + 2
            self.canvas.rois = m["rois"] if self.show_roi.isChecked() else {}
            self.status.setText("仅回放原始视频 · 不运行模型或计数")
            self.timer.start(max(1, round(1000 / fps)))
        except Exception as exc:
            self.stop_replay()
            self.error(str(exc))

    def replay_tick(self):
        ok, frame = self.replay.read()
        if not ok or self.replay.get(cv2.CAP_PROP_POS_MSEC) / 1000 > self.replay_end:
            self.stop_replay()
            return
        self.canvas.set_image(qimage(frame))

    def stop_replay(self):
        self.timer.stop()
        if self.replay:
            self.replay.release()
            self.replay = None

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
