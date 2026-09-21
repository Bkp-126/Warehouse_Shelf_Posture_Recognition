"""Desktop layout for the local video review workflow."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from src.config import ROOT
from src.ui.presentation import EventTimeline, StatsPanel, VideoCanvas, CurrentWorkPanel

STYLE = """
QMainWindow, QDialog { background:#101820; }
QWidget { color:#dce6ee; font-family:'Microsoft YaHei UI'; font-size:13px; }
QLabel { background:transparent; }
QLabel#muted { color:#8fa4b7; }
QLabel#heading { font-size:25px; font-weight:600; }
QLabel#section { font-size:16px; font-weight:600; }
QWidget#metricsStrip { background:#182630; border:1px solid #2c414e; border-radius:8px; }
QLabel#stripValue { font-size:24px; font-weight:600; }
QLabel#workTitle { font-size:14px; font-weight:600; }
QWidget#currentArea { background:#101820; }
QLabel#emptyWork { color:#a0b5c2; background:#17252e; border:1px solid #2b414c; border-radius:8px; padding:20px; }
QWidget#metric { background:#192735; border:1px solid #2b3c4e; border-radius:8px; }
QLabel#metricValue { font-size:28px; font-weight:600; }
QPushButton { background:#203143; border:1px solid #34485b; border-radius:6px; padding:8px 14px; }
QPushButton:hover { background:#2a4053; border-color:#609caa; }
QPushButton:pressed, QPushButton:checked { background:#2c4d5b; }
QPushButton:disabled { color:#61778a; background:#17232f; border-color:#263746; }
QPushButton#primary { background:#276960; border-color:#39877a; color:#f0fffa; }
QPushButton#primary:hover { background:#328073; }
QPushButton#primary:disabled { background:#193d35; border-color:#305247; color:#78978c; }
QLineEdit,QComboBox,QDoubleSpinBox,QTextEdit { background:#152330; border:1px solid #34485b; border-radius:4px; padding:7px; selection-background-color:#326458; }
QCheckBox { spacing:6px; color:#9eb2c2; }
QCheckBox::indicator { width:13px; height:13px; }
QCheckBox::indicator:unchecked { background:#172635; border:1px solid #496074; border-radius:3px; }
QCheckBox::indicator:checked { background:#4dafa0; border:1px solid #90d7c7; border-radius:3px; }
QTabWidget::pane { border:1px solid #26394a; background:#121f2c; border-radius:5px; }
QTabBar::tab { background:#152330; color:#8fa4b7; padding:10px 13px; }
QTabBar::tab:selected { color:#d7f4eb; border-bottom:2px solid #63bfb0; }
QListWidget { background:#121f2c; border:none; outline:none; padding:6px; }
QListWidget::item { background:#192a39; border:1px solid #2b4154; border-radius:7px; padding:10px; margin:4px 0; }
QListWidget::item:selected { background:#234339; border-color:#6dafa0; }
QListWidget::item:hover { background:#233847; }
QTableWidget { background:#121f2c; gridline-color:#2b3c4e; border:none; alternate-background-color:#192a39; selection-background-color:#2c5148; }
QHeaderView::section { background:#1b2e3e; color:#a8bdcd; border:none; padding:8px; }
QScrollBar:vertical { background:#14202b; width:8px; }
QScrollBar::handle:vertical { background:#354f62; border-radius:4px; min-height:25px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QSplitter::handle { background:#263747; width:1px; }
QToolTip { background:#233747; color:#e2ecf3; border:1px solid #567184; }
"""


def build_window(self):
    self.setStyleSheet(STYLE)
    central = QWidget()
    self.setCentralWidget(central)
    outer = QVBoxLayout(central)
    outer.setContentsMargins(20, 16, 20, 12)
    outer.setSpacing(12)
    heading = QHBoxLayout()
    title = QLabel("仓储作业复盘")
    title.setObjectName("heading")
    heading.addWidget(title)
    subtitle = QLabel("本地视频 · 姿态与作业事件")
    subtitle.setObjectName("muted")
    heading.addWidget(subtitle)
    heading.addStretch()
    self.load_session_btn = QPushButton("历史会话")
    self.load_session_btn.clicked.connect(self.load_session_dialog)
    heading.addWidget(self.load_session_btn)
    self.settings_btn = QPushButton("分析设置")
    heading.addWidget(self.settings_btn)
    outer.addLayout(heading)

    self.settings_dialog = QDialog(self)
    self.settings_dialog.setWindowTitle("分析设置")
    self.settings_dialog.resize(700, 400)
    form = QFormLayout(self.settings_dialog)
    form.setSpacing(16)
    self.video_edit = QLineEdit(str(ROOT / "data/video_1.mp4"))
    self.model_edit = QLineEdit(str(ROOT / "models/yolo11n-pose.pt"))
    self.browse_model = QPushButton("选择模型")
    self.browse_model.clicked.connect(lambda: self.pick(self.model_edit, "模型 (*.pt)"))
    form.addRow("视频路径", self.video_edit)
    model_row = QHBoxLayout()
    model_row.addWidget(self.model_edit)
    model_row.addWidget(self.browse_model)
    form.addRow("姿态模型", model_row)
    self.output_edit = QLineEdit(str(ROOT / "output/sessions"))
    self.output_btn = QPushButton("选择目录")
    self.output_btn.clicked.connect(self.pick_output)
    output_row = QHBoxLayout()
    output_row.addWidget(self.output_edit)
    output_row.addWidget(self.output_btn)
    form.addRow("会话保存", output_row)
    self.device = QComboBox()
    self.device.addItems(["auto", "cpu", "cuda:0"])
    form.addRow("计算设备", self.device)
    self.alert = QDoubleSpinBox()
    self.alert.setRange(0.1, 3600)
    self.alert.setValue(10)
    self.alert.setSuffix(" 秒")
    form.addRow("持续提醒时长", self.alert)
    notice = QLabel(
        "时长为演示规则，姿态候选与辅助提醒需人工复核。\n不作为医学、人体工效或安全合规判断。"
    )
    notice.setWordWrap(True)
    notice.setObjectName("muted")
    form.addRow(notice)
    close = QPushButton("完成设置")
    close.clicked.connect(self.settings_dialog.hide)
    form.addRow(close)
    self.settings_btn.clicked.connect(self.settings_dialog.show)

    bar = QHBoxLayout()
    self.browse_video = QPushButton("选择视频")
    self.browse_video.clicked.connect(self.pick_video)
    self.video_name = QLabel("尚未加载视频")
    self.video_name.setObjectName("section")
    bar.addWidget(self.browse_video)
    bar.addWidget(self.video_name)
    bar.addStretch()
    self.preview_btn = QPushButton("预览视频")
    self.preview_btn.clicked.connect(self.preview)
    self.region_btn = QPushButton("区域设置")
    self.region_btn.setCheckable(True)
    self.region_btn.toggled.connect(self.toggle_regions)
    self.start_btn = QPushButton("开始分析")
    self.start_btn.setObjectName("primary")
    self.start_btn.clicked.connect(self.start_analysis)
    self.stop_btn = QPushButton("结束分析")
    self.stop_btn.clicked.connect(self.stop_analysis)
    self.stop_btn.setEnabled(False)
    for btn in [self.preview_btn, self.region_btn, self.start_btn, self.stop_btn]:
        bar.addWidget(btn)
    outer.addLayout(bar)
    self.stats = StatsPanel()
    self.stats.setText("画面人数 0\n作业事件 0\n姿态候选 0\n辅助提醒 0\n无法判断 0")
    outer.addWidget(self.stats)

    self.region_panel = QWidget()
    region_box = QVBoxLayout(self.region_panel)
    region_box.setContentsMargins(0, 0, 0, 0)
    help_text = QLabel("实线是货架外观标识；虚线才是作业判定区域。请核对位置，再确认。")
    help_text.setObjectName("muted")
    region_box.addWidget(help_text)
    row = QHBoxLayout()
    self.draw_left = QPushButton("绘制左侧判定区")
    self.draw_right = QPushButton("绘制右侧判定区")
    self.confirm_btn = QPushButton("确认判定区")
    self.confirm_btn.setObjectName("primary")
    self.clear_btn = QPushButton("清除判定区")
    for btn in [self.draw_left, self.draw_right, self.clear_btn, self.confirm_btn]:
        row.addWidget(btn)
    row.addStretch()
    region_box.addLayout(row)
    self.draw_left.clicked.connect(lambda: self.begin_roi("left"))
    self.draw_right.clicked.connect(lambda: self.begin_roi("right"))
    self.confirm_btn.clicked.connect(self.confirm_roi)
    self.clear_btn.clicked.connect(self.clear_roi)
    outer.addWidget(self.region_panel)
    self.region_panel.hide()

    split = QSplitter()
    self.main_splitter = split
    outer.addWidget(split, 1)
    left = QWidget()
    video_layout = QVBoxLayout(left)
    video_layout.setContentsMargins(0, 0, 14, 0)
    video_layout.setSpacing(6)
    self.canvas = VideoCanvas()
    self.canvas.roi_finished.connect(self.roi_finished)
    video_layout.addWidget(self.canvas, 1)
    progress = QHBoxLayout()
    self.play_btn = QPushButton("播放原片")
    self.play_btn.setEnabled(False)
    self.play_btn.setMinimumHeight(40)
    self.play_btn.clicked.connect(self.toggle_playback)
    self.clock_label = QLabel("00:00 / 00:00")
    self.clock_label.setMinimumWidth(103)
    self.timeline = EventTimeline()
    self.timeline.seek.connect(self.seek_video)
    progress.addWidget(self.play_btn)
    progress.addWidget(self.clock_label)
    progress.addWidget(self.timeline, 1)
    video_layout.addLayout(progress)
    layers = QHBoxLayout()
    self.layer_hint = QLabel("货架轮廓仅用于显示")
    self.layer_hint.setObjectName("muted")
    layers.addWidget(self.layer_hint)
    layers.addStretch()
    self.skeleton = QCheckBox("骨架")
    self.skeleton.setChecked(True)
    self.angles = QCheckBox("关节角度")
    self.show_shelves = QCheckBox("货架轮廓")
    self.show_shelves.setChecked(True)
    self.show_roi = QCheckBox("判定区")
    for box in [self.skeleton, self.angles, self.show_shelves, self.show_roi]:
        box.toggled.connect(self.update_settings)
        layers.addWidget(box)
    video_layout.addLayout(layers)
    split.addWidget(left)

    right = QWidget()
    right.setMinimumWidth(340)
    panel = QVBoxLayout(right)
    panel.setContentsMargins(14, 0, 0, 0)
    panel.setSpacing(12)
    label = QLabel("当前作业")
    label.setObjectName("section")
    panel.addWidget(label)
    self.current_work = CurrentWorkPanel()
    panel.addWidget(self.current_work)
    self.tabs = QTabWidget()
    panel.addWidget(self.tabs, 1)
    card_page = QWidget()
    card_box = QVBoxLayout(card_page)
    card_box.setContentsMargins(0, 0, 0, 0)
    filter_row = QHBoxLayout()
    filter_row.setContentsMargins(8, 4, 8, 0)
    filter_row.addWidget(QLabel("按发生时间排序"))
    filter_row.addStretch()
    self.event_filter = QComboBox()
    self.event_filter.addItems(["全部事件", "伸手作业", "姿态候选"])
    self.event_filter.currentIndexChanged.connect(self.filter_events)
    filter_row.addWidget(self.event_filter)
    card_box.addLayout(filter_row)
    self.event_list = QListWidget()
    self.event_list.setWordWrap(True)
    self.event_list.setIconSize(QSize(64, 48))
    self.event_list.currentItemChanged.connect(self.select_event_card)
    self.event_list.itemDoubleClicked.connect(lambda item: self.replay_selected())
    card_box.addWidget(self.event_list, 1)
    self.replay_btn = QPushButton("回放选中事件  ·  前后各 2 秒")
    self.replay_btn.setEnabled(False)
    self.replay_btn.clicked.connect(self.replay_selected)
    card_box.addWidget(self.replay_btn)
    self.tabs.addTab(card_page, "事件记录")
    self.table = QTableWidget(0, 7)
    self.table.setHorizontalHeaderLabels(
        ["人物 / 区域", "事件", "开始(s)", "结束(s)", "有效时长(s)", "提醒(s)", "结束原因"]
    )
    self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    self.table.setAlternatingRowColors(True)
    self.table.cellDoubleClicked.connect(self.replay_event)
    self.tabs.addTab(self.table, "明细")
    fig = Figure(figsize=(4, 2), facecolor="#121f2c")
    self.chart = FigureCanvasQTAgg(fig)
    self.ax = fig.add_subplot(111)
    self.chart_time = -1
    self.tabs.addTab(self.chart, "趋势")
    self.draw_chart([], 0)
    note = QLabel("姿态候选与辅助提醒均需人工复核")
    note.setObjectName("muted")
    panel.addWidget(note)
    split.addWidget(right)
    split.setSizes([1100, 390])
    split.setStretchFactor(0, 1)
    split.setStretchFactor(1, 0)

    footer = QHBoxLayout()
    self.status = QLabel("就绪 · 预览视频并确认作业判定区")
    self.status.setObjectName("muted")
    footer.addWidget(self.status, 1)
    self.log_btn = QPushButton("运行日志")
    self.log_btn.setCheckable(True)
    footer.addWidget(self.log_btn)
    outer.addLayout(footer)
    self.log = QTextEdit()
    self.log.setReadOnly(True)
    self.log.document().setMaximumBlockCount(250)
    self.log.setMaximumHeight(130)
    outer.addWidget(self.log)
    self.log.hide()
    self.log_btn.toggled.connect(self.log.setVisible)
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.replay_tick)
