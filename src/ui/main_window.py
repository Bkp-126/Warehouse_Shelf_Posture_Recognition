import sys
import os
import time
from collections import deque  # 用于存储最近 N 帧的数据

# 路径自适应
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QFrame, QPushButton, QGroupBox,
                               QTextEdit, QGridLayout, QCheckBox, QSizePolicy)
from PySide6.QtCore import Qt, Slot, Signal, QEvent, QPoint, QTimer
from PySide6.QtGui import QFont, QPixmap, QImage, QCursor
from src.ui.ai_worker import AIWorker

# --- 📊 引入 Matplotlib ---
import matplotlib

matplotlib.use('QtAgg')  # 告诉 Matplotlib 使用 Qt 后端
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 设置 Matplotlib 的全局样式 (暗黑风格)
plt.style.use('dark_background')


class RealTimeChart(FigureCanvas):
    """自定义的动态图表组件"""

    def __init__(self, parent=None, width=5, height=2, dpi=100):
        # 创建画布，背景色设为深灰，去掉边框
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e1e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')  # 坐标轴背景

        super().__init__(self.fig)
        self.setParent(parent)

        # 数据容器 (只保留最近 50 个点)
        self.max_len = 50
        self.data_reach = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_bend = deque([0] * self.max_len, maxlen=self.max_len)

        # 初始化两条曲线
        # 黄线: 伸手, 红线: 弯腰
        self.line_reach, = self.ax.plot([], [], 'o-', color='#ffaa00', linewidth=2, markersize=4, label='Reach')
        self.line_bend, = self.ax.plot([], [], 'o-', color='#ff5555', linewidth=2, markersize=4, label='Bend')

        # 设置坐标轴样式 (去刻度，留网格)
        self.ax.grid(True, color='#333333', linestyle='--')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.tick_params(colors='#888')

        # 设置Y轴范围 (自动适应或固定)
        self.ax.set_ylim(-0.5, 5)
        self.ax.legend(loc='upper left', facecolor='#1e1e1e', edgecolor='#333', labelcolor='#ccc', fontsize=8)

    def update_chart(self, new_reach, new_bend):
        """更新数据并重绘"""
        self.data_reach.append(new_reach)
        self.data_bend.append(new_bend)

        x_data = range(len(self.data_reach))

        self.line_reach.set_data(x_data, self.data_reach)
        self.line_bend.set_data(x_data, self.data_bend)

        # 动态调整 Y 轴 (如果数值超过当前范围)
        max_val = max(max(self.data_reach), max(self.data_bend))
        if max_val > self.ax.get_ylim()[1]:
            self.ax.set_ylim(-0.5, max_val + 2)

        self.draw()  # 重绘


class MainWindow(QMainWindow):
    settings_changed = Signal(str, bool)
    roi_updated = Signal(str, list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能仓储行为分析系统 v4.0 (数据可视化版)")
        self.resize(1450, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLabel { color: #e0e0e0; font-family: 'Microsoft YaHei UI'; }
            QLabel#AppHeader { font-size: 20px; font-weight: bold; color: #ffffff; padding: 12px; background-color: #1f1f1f; border-bottom: 1px solid #333; }
            QFrame#Card { background-color: #1e1e1e; border-radius: 8px; border: 1px solid #333; }
            QLabel#CardTitle { color: #888; font-weight: bold; font-size: 14px; margin-bottom: 8px; }
            QLabel#BigNum { font-size: 26px; font-weight: bold; font-family: 'Arial'; }
            QCheckBox { color: #ccc; spacing: 8px; font-size: 14px; }
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 3px; border: 1px solid #555; background: #2b2b2b; }
            QCheckBox::indicator:checked { background-color: #007acc; border-color: #007acc; }
            QTextEdit { background-color: #121212; color: #00ff00; font-family: 'Consolas'; font-size: 12px; border: 1px solid #333; }
            QPushButton { border-radius: 6px; font-weight: bold; font-size: 14px; padding: 8px; background-color: #333; color: #ccc; border: 1px solid #444; }
            QPushButton:hover { background-color: #444; }
            QPushButton#ActionBtn { background-color: #007acc; color: white; border: none; }
            QPushButton#StopBtn { background-color: #d9534f; color: white; border: none; }
            QFrame#HLine { color: #333; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.root_layout = QVBoxLayout()
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        main_widget.setLayout(self.root_layout)

        self.root_layout.addWidget(
            QLabel(" 📦 智能仓储行为分析系统 | Smart Warehouse Analytics", objectName="AppHeader"))

        content = QWidget()
        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(15, 15, 15, 15)
        self.content_layout.setSpacing(15)
        content.setLayout(self.content_layout)
        self.root_layout.addWidget(content)

        self.init_video_area()
        self.init_dashboard_area()
        self.worker = None
        self.drawing_target = None
        self.temp_points = []

        # 记录上一次的统计值，用于计算增量
        self.last_reach = 0
        self.last_bend = 0

    def init_video_area(self):
        video_frame = QFrame(objectName="Card")
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        video_frame.setLayout(layout)

        self.lbl_video = QLabel("等待视频信号...")
        self.lbl_video.setStyleSheet("background-color: #000; color: #555; font-size: 22px;")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_video.setScaledContents(False)
        self.lbl_video.setMouseTracking(True)
        self.lbl_video.installEventFilter(self)

        layout.addWidget(self.lbl_video)
        self.content_layout.addWidget(video_frame, stretch=75)

    def init_dashboard_area(self):
        right_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        right_panel.setLayout(layout)

        # 1. 状态
        status_card = QFrame(objectName="Card")
        s_layout = QVBoxLayout()
        status_card.setLayout(s_layout)
        s_layout.addWidget(QLabel("系统状态", objectName="CardTitle"))
        self.lbl_status = QLabel(" ● 就绪 ")
        self.lbl_status.setStyleSheet(
            "color: #aaa; background-color: #333; border-radius: 14px; padding: 6px 12px; font-weight: bold;")
        s_layout.addWidget(self.lbl_status)
        layout.addWidget(status_card)

        # 2. 统计
        data_card = QFrame(objectName="Card")
        d_layout = QVBoxLayout()
        data_card.setLayout(d_layout)
        d_layout.addWidget(QLabel("实时数据", objectName="CardTitle"))
        grid = QGridLayout()

        self.lbl_worker = QLabel("0", objectName="BigNum", styleSheet="color: #fff")
        self.lbl_reach = QLabel("0", objectName="BigNum", styleSheet="color: #ffaa00")
        self.lbl_bend = QLabel("0", objectName="BigNum", styleSheet="color: #ff5555")

        grid.addWidget(QLabel("在岗人数"), 0, 0);
        grid.addWidget(self.lbl_worker, 0, 1, alignment=Qt.AlignRight)
        grid.addWidget(QLabel("伸手工作"), 1, 0);
        grid.addWidget(self.lbl_reach, 1, 1, alignment=Qt.AlignRight)
        grid.addWidget(QLabel("弯腰工作"), 2, 0);
        grid.addWidget(self.lbl_bend, 2, 1, alignment=Qt.AlignRight)
        d_layout.addLayout(grid)
        layout.addWidget(data_card)

        # 3. 🟢 新增：趋势图表
        chart_card = QFrame(objectName="Card")
        c_layout = QVBoxLayout()
        chart_card.setLayout(c_layout)
        c_layout.addWidget(QLabel("作业趋势 (TRENDS)", objectName="CardTitle"))

        # 实例化图表组件
        self.chart = RealTimeChart(width=5, height=3)  # 高度设为3英寸
        c_layout.addWidget(self.chart)

        layout.addWidget(chart_card)

        # 4. 控制
        viz_card = QFrame(objectName="Card")
        v_layout = QVBoxLayout()
        viz_card.setLayout(v_layout)
        v_layout.addWidget(QLabel("视觉与区域控制", objectName="CardTitle"))

        self.btn_draw_left = QPushButton("✏️ 绘制左侧货架")
        self.btn_draw_left.clicked.connect(lambda: self.start_drawing("left"))
        self.btn_draw_right = QPushButton("✏️ 绘制右侧货架")
        self.btn_draw_right.clicked.connect(lambda: self.start_drawing("right"))
        v_layout.addWidget(self.btn_draw_left);
        v_layout.addWidget(self.btn_draw_right)

        line = QFrame(objectName="HLine");
        line.setFrameShape(QFrame.HLine);
        line.setLineWidth(1)
        v_layout.addWidget(line)

        self.cb_roi = QCheckBox("显示货架区域 (ROI)");
        self.cb_roi.setChecked(True)
        self.cb_skel = QCheckBox("显示骨架与置信度");
        self.cb_skel.setChecked(True)
        self.cb_angle = QCheckBox("显示关节角度");
        self.cb_angle.setChecked(False)

        self.cb_roi.toggled.connect(lambda v: self.send_settings("roi", v))
        self.cb_skel.toggled.connect(lambda v: self.send_settings("skeleton", v))
        self.cb_angle.toggled.connect(lambda v: self.send_settings("angles", v))

        v_layout.addWidget(self.cb_roi);
        v_layout.addWidget(self.cb_skel);
        v_layout.addWidget(self.cb_angle)
        layout.addWidget(viz_card)

        # 5. 日志与按钮
        log_card = QFrame(objectName="Card")
        l_layout = QVBoxLayout()
        log_card.setLayout(l_layout)
        l_layout.addWidget(QLabel("日志", objectName="CardTitle"))
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        l_layout.addWidget(self.log_area)
        layout.addWidget(log_card, stretch=1)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 启动", objectName="ActionBtn")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_stop = QPushButton("⏹ 停止", objectName="StopBtn")
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        btn_layout.addWidget(self.btn_start, stretch=3)
        btn_layout.addWidget(self.btn_stop, stretch=1)
        layout.addLayout(btn_layout)

        self.content_layout.addWidget(right_panel, stretch=25)

    def start_drawing(self, target):
        if not self.worker:
            self.log_area.append("❌ 请先启动分析再绘制！")
            return
        self.drawing_target = target
        self.temp_points = []
        self.log_area.append(f">>> 开始绘制 [{target}]...")
        self.btn_draw_left.setEnabled(False);
        self.btn_draw_right.setEnabled(False)
        self.lbl_video.setCursor(QCursor(Qt.CrossCursor))

    def eventFilter(self, source, event):
        if source == self.lbl_video and event.type() == QEvent.MouseButtonPress:
            if self.drawing_target:
                pos = event.pos()
                w = self.lbl_video.width();
                h = self.lbl_video.height()
                self.temp_points.append((round(pos.x() / w, 3), round(pos.y() / h, 3)))
                self.log_area.append(f"   点 {len(self.temp_points)}")
                if len(self.temp_points) == 4: self.finish_drawing()
                return True
        return super().eventFilter(source, event)

    def finish_drawing(self):
        self.log_area.append(f">>> [{self.drawing_target}] 绘制完成！")
        self.roi_updated.emit(self.drawing_target, self.temp_points)
        self.drawing_target = None;
        self.temp_points = []
        self.lbl_video.setCursor(QCursor(Qt.ArrowCursor))
        self.btn_draw_left.setEnabled(True);
        self.btn_draw_right.setEnabled(True)

    def send_settings(self, key, value):
        self.settings_changed.emit(key, value)

    def start_analysis(self):
        self.log_area.append(">>> 初始化...")
        self.btn_start.setEnabled(False);
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText(" ● 运行中 ")
        self.lbl_status.setStyleSheet(
            "color: #fff; background-color: #009900; border-radius: 14px; padding: 6px 12px; font-weight: bold;")

        # 重置图表计数器
        self.last_reach = 0
        self.last_bend = 0
        self.chart.data_reach.clear()
        self.chart.data_bend.clear()

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model = os.path.join(root_dir, "models", "yolo11n-pose.pt")
        video = os.path.join(root_dir, "data", "video_1.mp4")

        self.worker = AIWorker(model, video)
        self.worker.frame_signal.connect(self.update_image)
        self.worker.stats_signal.connect(self.update_stats)
        self.worker.log_signal.connect(self.update_log)
        self.settings_changed.connect(self.worker.update_settings)
        self.roi_updated.connect(self.worker.update_roi)

        self.send_settings("roi", self.cb_roi.isChecked())
        self.send_settings("skeleton", self.cb_skel.isChecked())
        self.send_settings("angles", self.cb_angle.isChecked())
        self.worker.start()

    def stop_analysis(self):
        if self.worker: self.worker.stop(); self.worker = None
        self.btn_start.setEnabled(True);
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText(" ● 就绪 ");
        self.lbl_status.setStyleSheet(
            "color: #aaa; background-color: #333; border-radius: 14px; padding: 6px 12px; font-weight: bold;")
        self.lbl_video.clear()

    @Slot(QImage)
    def update_image(self, image):
        pix = QPixmap.fromImage(image)
        w, h = self.lbl_video.width(), self.lbl_video.height()
        if w > 10 and h > 10: self.lbl_video.setPixmap(pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(dict)
    def update_stats(self, data):
        r = data.get("reach_count", 0)
        b = data.get("bend_count", 0)
        self.lbl_worker.setText(str(data.get("worker_count", 0)))
        self.lbl_reach.setText(str(r))
        self.lbl_bend.setText(str(b))

        # 🟢 图表更新逻辑：我们不画总数，而是画“当前这一刻是否发生了动作”
        # 或者画总数的增长趋势。为了好看，我们画“总数”。
        # 每隔几帧刷新一次图表，否则太费资源
        self.chart.update_chart(r, b)

    @Slot(str)
    def update_log(self, text):
        self.log_area.append(text)

    def closeEvent(self, event):
        self.stop_analysis()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())