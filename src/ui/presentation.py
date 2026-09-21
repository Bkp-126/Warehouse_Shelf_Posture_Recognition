"""Display-only overlays. Shelf envelopes never enter the analysis runner."""

import json
import math
from time import perf_counter

from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget, QScrollArea, QFrame
from src.config import ROOT
from src.geometry import image_rect, widget_to_normalized


def shelf_profiles(video_hash):
    path = ROOT / "configs/shelf_display.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    profile = data.get("videos", {}).get(video_hash, {})
    result = {}
    for name, points in profile.get("outlines", {}).items():
        if not isinstance(points, list) or not 3 <= len(points) <= 20:
            raise ValueError("货架显示轮廓需要 3 至 20 个点")
        if any(
            len(p) != 2
            or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                for v in p
            )
            for p in points
        ):
            raise ValueError("货架显示轮廓坐标无效")
        result[name] = points
    return result


def time_text(seconds):
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


class VideoCanvas(QWidget):
    roi_finished = Signal(str, list)
    painted_latency = Signal(float)

    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.rois = {}
        self.outlines = {}
        self.show_outlines = True
        self.show_skeleton = True
        self.show_angles = False
        self.people = []
        self.feedback = []
        self.active_regions = set()
        self._levels = {}
        self._transitions = {}
        self._animation = QTimer(self)
        self._animation.setInterval(16)
        self._animation.timeout.connect(self.animate_regions)
        self.points = []
        self.target = None
        self.frame_origin = None
        self.setMinimumSize(480, 270)

    def set_image(self, image, origin=None, people=None, feedback=None):
        self.image = image
        self.people = people or []
        previous = self.feedback
        self.feedback = feedback or []
        active = {f["roi"] for f in self.feedback if f["phase"] == "active" and f["wrists"]}
        unknown = {f["roi"] for f in self.feedback if f["phase"] == "unknown"} - active
        unknown_people = {f["track_id"] for f in self.feedback if f["phase"] == "unknown"}
        present = {d["track_id"] for d in self.people}
        unknown |= {
            f["roi"]
            for f in previous
            if f["track_id"] in unknown_people or f["track_id"] not in present
        } - active
        now = perf_counter()
        if people is None:
            # Raw replay / a new source is never decorated with old live feedback.
            self.clear_feedback()
        else:
            for name in active | self.active_regions | unknown:
                if name in unknown:
                    self._levels[name] = 0
                    self._transitions.pop(name, None)
                elif (name in active) != (name in self.active_regions):
                    self._transitions[name] = (
                        self._levels.get(name, 0),
                        float(name in active),
                        now,
                    )
            self.active_regions = active
            if self._transitions:
                self._animation.start()
        self.frame_origin = origin
        self.update()

    def clear_feedback(self):
        self.feedback = []
        self.active_regions = set()
        self._levels.clear()
        self._transitions.clear()
        self._animation.stop()
        self.update()

    def animate_regions(self):
        now = perf_counter()
        for name, (start, end, stamp) in list(self._transitions.items()):
            fraction = min(1.0, (now - stamp) / 0.18)
            self._levels[name] = start + (end - start) * fraction
            if fraction >= 1:
                del self._transitions[name]
        if not self._transitions:
            self._animation.stop()
        self.update()

    @staticmethod
    def mix(a, b, amount):
        return QColor(*(round(x + (y - x) * amount) for x, y in zip(a, b)))

    @staticmethod
    def pill(p, x, y, text, color, bounds):
        font = QFont(p.font())
        font.setPixelSize(12)
        font.setBold(True)
        p.setFont(font)
        width = p.fontMetrics().horizontalAdvance(text) + 20
        x = max(bounds.left() + 4, min(x, bounds.right() - width - 4))
        y = max(bounds.top() + 4, min(y, bounds.bottom() - 28))
        box = QRectF(x, y, width, 25)
        p.setPen(QPen(QColor(color), 0.8))
        p.setBrush(QColor(15, 25, 35, 225))
        p.drawRoundedRect(box, 5, 5)
        p.setPen(QColor(color))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#080f17"))
        if self.image.isNull():
            p.setPen(QColor("#95a9bb"))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "选择本地视频\n预览并确认作业判定区域后开始分析",
            )
            p.end()
            return
        x, y, w, h = image_rect(
            self.width(), self.height(), self.image.width(), self.image.height()
        )
        bounds = QRectF(x, y, w, h)
        p.drawImage(bounds, self.image)
        p.setClipRect(bounds)
        point = lambda a, b: QPointF(x + a * w, y + b * h)
        if self.show_outlines:
            for name, coords in self.outlines.items():
                poly = QPolygonF([point(a, b) for a, b in coords])
                level = self._levels.get(name, 0)
                p.setPen(
                    QPen(self.mix((123, 148, 158, 180), (56, 225, 195, 255), level), 1.2 + level)
                )
                p.setBrush(self.mix((104, 129, 143, 7), (37, 213, 181, 30), level))
                p.drawPolygon(poly)
                label = "左侧货架 A" if name == "left" else "右侧货架 B"
                lx = x + (0.045 if name == "left" else 0.82) * w
                if name in self.active_regions:
                    label += "  ·  作业中"
                self.pill(
                    p,
                    lx,
                    y + 0.18 * h,
                    label,
                    "#67edd0" if name in self.active_regions else "#bccbd4",
                    bounds,
                )
        for name, coords in self.rois.items():
            poly = QPolygonF([point(a, b) for a, b in coords])
            pen = QPen(QColor("#79cfc5"), 1.2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(poly)
            anchor = min(poly, key=lambda q: q.y())
            self.pill(
                p,
                anchor.x(),
                anchor.y() - 29,
                ("左" if name == "left" else "右") + " · 作业判定区",
                "#a4d5cd",
                bounds,
            )
        for d in self.people:
            # Keypoints and boxes are pixel coordinates in the decoded source frame.
            sx, sy = w / self.image.width(), h / self.image.height()
            x1, y1, x2, y2 = d["bbox"]
            bx, by, ex, ey = x + x1 * sx, y + y1 * sy, x + x2 * sx, y + y2 * sy
            color = "#edbd62" if d["bend"] else ("#a2aeb9" if d["bend"] is None else "#d5e0e6")
            p.setPen(QPen(QColor(color), 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            length = max(5, min(14, (ex - bx) / 5, (ey - by) / 5))
            for px, py, dx, dy in [
                (bx, by, 1, 1),
                (ex, by, -1, 1),
                (bx, ey, 1, -1),
                (ex, ey, -1, -1),
            ]:
                p.drawLine(QPointF(px, py), QPointF(px + dx * length, py))
                p.drawLine(QPointF(px, py), QPointF(px, py + dy * length))
            state = "无法判断" if d["bend"] is None else ("姿态候选" if d["bend"] else "观察中")
            self.pill(p, bx, by - 29, f"T{d['track_id']:02d} · {state}", color, bounds)
            k = d["keypoints"]
            if self.show_skeleton:
                p.setPen(QPen(QColor(105, 209, 207, 200), 1.35))
                links = [
                    (5, 7),
                    (7, 9),
                    (6, 8),
                    (8, 10),
                    (5, 6),
                    (5, 11),
                    (6, 12),
                    (11, 12),
                    (11, 13),
                    (13, 15),
                    (12, 14),
                    (14, 16),
                ]
                for a, b in links:
                    if k[a][2] > 0.5 and k[b][2] > 0.5:
                        p.drawLine(
                            QPointF(x + k[a][0] * sx, y + k[a][1] * sy),
                            QPointF(x + k[b][0] * sx, y + k[b][1] * sy),
                        )
                p.setBrush(QColor("#9de1d8"))
                p.setPen(Qt.PenStyle.NoPen)
                for kp in k[5:]:
                    if kp[2] > 0.5:
                        p.drawEllipse(QPointF(x + kp[0] * sx, y + kp[1] * sy), 2.1, 2.1)
            if self.show_angles and d["angle"] is not None:
                self.pill(p, bx, ey + 3, f"髋角 {d['angle']:.1f}°", color, bounds)
        occupied = []
        body_boxes = [
            QRectF(
                x + d["bbox"][0] * w / self.image.width(),
                y + d["bbox"][1] * h / self.image.height(),
                (d["bbox"][2] - d["bbox"][0]) * w / self.image.width(),
                (d["bbox"][3] - d["bbox"][1]) * h / self.image.height(),
            )
            for d in self.people
        ]
        for feedback in self.feedback:
            if feedback["phase"] not in ("active", "candidate") or not feedback["wrists"]:
                continue
            active = feedback["phase"] == "active"
            color = QColor("#47e2bd" if active else "#a6bbb9")
            for wrist in feedback["wrists"]:
                pos = QPointF(
                    x + wrist["x"] * w / self.image.width(),
                    y + wrist["y"] * h / self.image.height(),
                )
                p.setPen(QPen(color, 2 if active else 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(pos, 10 if active else 6, 10 if active else 6)
                p.setBrush(color)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(pos, 3.5, 3.5)
            wrist = max(feedback["wrists"], key=lambda item: item["confidence"])
            pos = QPointF(
                x + wrist["x"] * w / self.image.width(), y + wrist["y"] * h / self.image.height()
            )
            region = {"left": "左侧货架", "right": "右侧货架"}.get(feedback["roi"], feedback["roi"])
            title = f"T{feedback['track_id']:02d} · " + ("伸手作业" if active else "进入候选")
            subtitle = (
                f"{region} · {feedback['duration']:.1f} s" if active else f"{region} · 待持续确认"
            )
            font = QFont(p.font())
            font.setPixelSize(13)
            font.setBold(True)
            p.setFont(font)
            bw = (
                max(
                    p.fontMetrics().horizontalAdvance(title),
                    p.fontMetrics().horizontalAdvance(subtitle),
                )
                + 24
            )
            bh = 51
            candidates = [
                QRectF(pos.x() + 22, pos.y() + 18, bw, bh),
                QRectF(pos.x() - bw - 22, pos.y() + 18, bw, bh),
                QRectF(pos.x() + 22, pos.y() - bh - 18, bw, bh),
                QRectF(pos.x() - bw - 22, pos.y() - bh - 18, bw, bh),
            ]
            for rect in candidates:
                rect.moveLeft(max(bounds.left() + 5, min(rect.left(), bounds.right() - bw - 5)))
                rect.moveTop(max(bounds.top() + 5, min(rect.top(), bounds.bottom() - bh - 5)))

            def overlap(rect):
                return sum(
                    rect.intersected(b).width() * rect.intersected(b).height()
                    for b in body_boxes + occupied
                    if rect.intersects(b)
                )

            box = min(candidates, key=overlap)
            occupied.append(box)
            endpoint = QPointF(
                box.left() if box.center().x() > pos.x() else box.right(), box.center().y()
            )
            p.setPen(QPen(color, 1.3))
            p.drawLine(pos, endpoint)
            p.setBrush(QColor(12, 27, 33, 242))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(color)
            p.drawText(box.adjusted(12, 4, -8, -25), Qt.AlignmentFlag.AlignVCenter, title)
            font.setBold(False)
            font.setPixelSize(12)
            p.setFont(font)
            p.setPen(QColor("#c7ddd8"))
            p.drawText(box.adjusted(12, 25, -8, -3), Qt.AlignmentFlag.AlignVCenter, subtitle)
        p.setPen(QPen(QColor("#edbd62"), 2))
        p.setBrush(QColor("#edbd62"))
        pts = [point(a, b) for a, b in self.points]
        for a, b in zip(pts, pts[1:]):
            p.drawLine(a, b)
        for pt in pts:
            p.drawEllipse(pt, 4, 4)
        p.end()
        if self.frame_origin is not None:
            self.painted_latency.emit((perf_counter() - self.frame_origin) * 1000)
            self.frame_origin = None

    def mousePressEvent(self, event):
        if (
            self.target is None
            or self.image.isNull()
            or event.button() != Qt.MouseButton.LeftButton
        ):
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
            self.target, self.points = None, []
            self.roi_finished.emit(target, points)


class StatsPanel(QWidget):
    """Compact metric strip retaining the original text/setText interface."""

    def __init__(self):
        super().__init__()
        self.setObjectName("metricsStrip")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(60)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)
        layout.setSpacing(18)
        self.values = {}
        self._text = ""
        for i, label in enumerate(["画面人数", "作业事件", "姿态候选", "辅助提醒", "无法判断"]):
            if i:
                divider = QFrame()
                divider.setFrameShape(QFrame.Shape.VLine)
                divider.setStyleSheet("color:#30424c;")
                layout.addWidget(divider)
            caption = QLabel("姿态待判断" if label == "无法判断" else label)
            caption.setObjectName("muted")
            if label == "无法判断":
                caption.setToolTip("当前帧无法判断弯腰姿态的人数；手腕可见性另见当前作业")
            value = QLabel("0")
            value.setObjectName("stripValue")
            if i in (2, 3):
                value.setStyleSheet("color:#edbd62;font-size:24px;font-weight:600;")
            layout.addWidget(caption)
            layout.addWidget(value)
            layout.addStretch()
            self.values[label] = value

    def setText(self, text):
        self._text = text
        for line in text.splitlines():
            name, value = line.split(" ", 1)
            if name in self.values:
                self.values[name].setText(value)

    def text(self):
        return self._text


class CurrentWorkPanel(QScrollArea):
    """All current person/region states; no arbitrary top-one selection."""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFixedHeight(150)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.content = QWidget()
        self.content.setObjectName("currentArea")
        self.setWidget(self.content)
        self.box = QVBoxLayout(self.content)
        self.box.setContentsMargins(0, 0, 6, 0)
        self.box.setSpacing(8)
        self.rows = {}
        self.empty = QLabel("等待作业\n\n确认伸手后，在这里查看人物、区域和持续时间")
        self.empty.setObjectName("emptyWork")
        self.empty.setWordWrap(True)
        self.box.addWidget(self.empty)
        self.uncertain = QLabel()
        self.uncertain.setObjectName("muted")
        self.uncertain.setWordWrap(True)
        self.uncertain.hide()
        self.box.addWidget(self.uncertain)
        self.box.addStretch()

    def reset(self, message="暂无可确认的伸手作业"):
        for widgets in self.rows.values():
            widgets[0].hide()
            self.box.removeWidget(widgets[0])
            widgets[0].deleteLater()
        self.rows.clear()
        self.uncertain.hide()
        self.setFixedHeight(150)
        self.empty.setText(message)
        self.empty.show()

    def set_feedback(self, image, people, feedback):
        uncertain = [f for f in feedback if f["phase"] == "unknown" and not f["event_id"]]
        feedback = [f for f in feedback if f not in uncertain]
        self.uncertain.setVisible(bool(uncertain))
        self.uncertain.setText(
            "手腕待判断：" + "、".join(f"T{f['track_id']:02d}" for f in uncertain)
        )
        self.setFixedHeight(min(270, max(136, len(feedback) * 126 + (32 if uncertain else 0))))
        phase_order = {"active": 0, "candidate": 1, "unknown": 2, "leaving": 3}
        feedback = sorted(
            feedback, key=lambda f: (phase_order[f["phase"]], f["track_id"], f["roi"])
        )
        keys = {(f["track_id"], f["roi"]) for f in feedback}
        for key in list(self.rows):
            if key not in keys:
                card = self.rows.pop(key)[0]
                card.hide()
                self.box.removeWidget(card)
                card.deleteLater()
        self.empty.setVisible(not feedback)
        self.empty.setText("暂无可确认的伸手作业\n\n有效手腕进入判定区域后显示")
        persons = {d["track_id"]: d for d in people}
        for index, f in enumerate(feedback):
            key = (f["track_id"], f["roi"])
            if key not in self.rows:
                card = QWidget()
                card.setObjectName("workCard")
                row = QHBoxLayout(card)
                row.setContentsMargins(12, 12, 12, 12)
                thumb = QLabel()
                thumb.setFixedSize(72, 94)
                thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                text_box = QVBoxLayout()
                title, state, duration = QLabel(), QLabel(), QLabel()
                for label in (title, state, duration):
                    label.setTextFormat(Qt.TextFormat.PlainText)
                    label.setWordWrap(True)
                    text_box.addWidget(label)
                title.setObjectName("workTitle")
                duration.setObjectName("muted")
                row.addWidget(thumb)
                row.addLayout(text_box, 1)
                self.rows[key] = (card, thumb, title, state, duration)
            card, thumb, title, state, duration = self.rows[key]
            self.box.insertWidget(index, card)
            active = f["phase"] == "active"
            card.setStyleSheet(
                "#workCard{background:"
                + (
                    "#18352f;border:1px solid #3f9d89;"
                    if active
                    else "#192730;border:1px solid #31434e;"
                )
                + "border-radius:8px;}"
            )
            region = {"left": "左侧货架", "right": "右侧货架"}.get(f["roi"], f["roi"] or "手腕观察")
            title.setText(f"T{f['track_id']:02d} · {region}")
            state.setText(
                {
                    "active": "伸手作业中",
                    "candidate": "进入候选 · 待确认",
                    "unknown": "手腕待判断",
                    "leaving": "已离开 · 等待结束确认",
                }[f["phase"]]
            )
            state.setStyleSheet(
                "color:" + ("#55dfbd" if active else "#aebec6") + ";font-weight:600;"
            )
            duration.setText(
                f"有效时长  {f['duration']:.1f} s" if f["event_id"] else "尚未计为作业事件"
            )
            person = persons.get(f["track_id"])
            if person is not None and not image.isNull():
                x1, y1, x2, y2 = person["bbox"]
                crop = (
                    QRectF(x1, y1, max(0, x2 - x1), max(0, y2 - y1))
                    .toRect()
                    .intersected(image.rect())
                )
                if not crop.isEmpty():
                    thumb.setPixmap(
                        QPixmap.fromImage(image.copy(crop)).scaled(
                            thumb.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                    continue
            thumb.clear()
            thumb.setText("暂无画面")


class EventTimeline(QWidget):
    seek = Signal(float)

    def __init__(self):
        super().__init__()
        self.duration = 0.0
        self.position = 0.0
        self.events = []
        self.can_seek = False
        self.setFixedHeight(66)
        self.setToolTip("分析结束后，点击时间轴回放原始视频；青色为作业事件，琥珀色为姿态候选")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = max(1, self.width() - 44)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#2b3e50"))
        p.drawRoundedRect(QRectF(36, 28, width, 4), 2, 2)
        if self.duration > 0:
            for e in self.events:
                start = min(self.duration, max(0, e["start_time"])) / self.duration
                end = min(self.duration, max(e["start_time"], e["end_time"])) / self.duration
                p.setBrush(QColor("#edbd62" if e["event_type"] == "bend" else "#58bdb1"))
                p.drawRoundedRect(
                    QRectF(
                        36 + start * width,
                        53 if e["event_type"] == "bend" else 43,
                        max(2, (end - start) * width),
                        4,
                    ),
                    1,
                    1,
                )
            p.setBrush(QColor("#e0ecef"))
            p.drawEllipse(QPointF(36 + min(1, self.position / self.duration) * width, 30), 5, 5)
        font = QFont(p.font())
        font.setPixelSize(10)
        p.setFont(font)
        p.setPen(QColor("#8da4b0"))
        p.drawText(QRectF(0, 38, 30, 14), Qt.AlignmentFlag.AlignCenter, "作业")
        p.drawText(QRectF(0, 49, 30, 14), Qt.AlignmentFlag.AlignCenter, "姿态")
        if self.duration > 0:
            for fraction in (0, 0.25, 0.5, 0.75, 1):
                tx = 36 + fraction * width
                p.drawText(
                    QRectF(max(28, min(tx - 20, self.width() - 42)), 2, 40, 18),
                    Qt.AlignmentFlag.AlignCenter,
                    time_text(self.duration * fraction),
                )
        p.end()

    def mousePressEvent(self, event):
        if self.can_seek and self.duration > 0 and event.button() == Qt.MouseButton.LeftButton:
            fraction = max(0, min(1, (event.position().x() - 36) / max(1, self.width() - 44)))
            self.seek.emit(fraction * self.duration)
