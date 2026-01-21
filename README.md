# 🏭 智能仓储行为分析系统  
**Smart Warehouse Analysis System**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)
![YOLO](https://img.shields.io/badge/AI-YOLOv11--Pose-red.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

> 一套面向 **工厂 / 仓储场景** 的工业级 AI 行为分析系统。  
> 基于 **YOLOv11 姿态估计**，实时识别工人 **伸手 / 弯腰** 行为，用于安全合规管理与作业分析。

---

## ✨ 核心功能（Key Features）

### 🎯 高精度行为识别
- 基于 **Ultralytics YOLOv11-pose** 实现人体关键点检测  
- 精准识别 **伸手、弯腰** 等高风险作业行为  
- 独创 **赛博朋克视觉风格**：  
  - 青色光晕（外描边）  
  - 黄色核心（骨架/手臂高亮）  
- 兼顾工业实用性与可视化冲击力

---

### 🧱 交互式电子围栏（ROI）
- 在视频画面中 **鼠标点击 4 个点** 绘制监控区域  
- 支持 **左 / 右货架** 独立 ROI 配置  
- ROI 自动保存至 `data/roi_config.json`  
- 程序重启后 **配置不丢失、即刻生效**

---

### 📈 实时数据可视化
- 右侧集成 **Matplotlib 动态波形图**
  - 实时展示作业频率趋势
- 数据看板实时统计：
  - 👷 在岗人数
  - ⚠️ 违规行为次数
- 帮助管理人员快速掌握现场状态

---

### 📸 证据自动留存
- 行为触发瞬间：
  - 自动抓拍 **高清截图**
  - 保存至本地 `output/images/`
- 自动生成 **带时间戳的 CSV 报表**
  - 便于审计、追溯与合规存档

---

## 📁 项目结构（Project Structure）

```text
Smart-Warehouse-Analysis-System/
├── src/
│   ├── core_inference.py        # YOLO 推理与行为判定核心
│   └── ui/
│       ├── main_window.py       # 主界面（Qt）
│       └── ai_worker.py         # AI 推理工作线程
│
├── data/
│   ├── video_1.mp4              # 示例视频
│   └── roi_config.json          # ROI 配置文件
│
├── models/
│   └── yolo11n-pose.pt          # YOLOv11 姿态模型
│
├── output/
│   ├── images/                  # 行为证据截图
│   └── report.csv               # 行为统计报表
│
├── requirements.txt             # Python 依赖
└── README.md
```

---

## 🚀 快速开始（Quick Start）

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

> 建议使用 **Python 3.10+**，如有 NVIDIA 显卡可自动启用 CUDA 加速。

---

### 2️⃣ 运行主程序
```bash
python src/ui/main_window.py
```

---

## 🧭 操作指南（Usage Guide）

### ▶️ 启动系统
- 运行主程序后，加载默认视频或接入实时视频流  
- 模型与 ROI 配置将自动初始化

---

### 🖱️ 绘制电子围栏（ROI）
1. 在视频画面中点击 **4 个点**
2. 依次绘制监控区域（货架 / 作业区）
3. 配置自动保存至 `data/roi_config.json`

---

### 📊 查看实时分析
- 右侧波形图：作业行为频率趋势
- 数据面板：在岗人数、违规计数
- 视频画面：高亮显示关键动作与骨架

---

### 🗂️ 查看证据与报表
- 行为触发截图：  
  `output/images/`
- 行为统计 CSV：  
  `output/report.csv`

---

## 🛠 技术栈（Tech Stack）

- **Python 3.10+**
- **PySide6 (Qt for Python)** – 工业级桌面 GUI
- **OpenCV** – 视频处理
- **Ultralytics YOLOv11 (Pose)** – 姿态估计
- **Matplotlib** – 实时数据可视化
- **NumPy**
- **NVIDIA CUDA**（可选，加速推理）

---

## 📄 License

本项目采用 **MIT License**  
你可以自由地使用、修改和分发本项目。

---

> 🚧 本项目适用于 **工业视觉 / 安全合规 / 智能制造** 场景  
> 欢迎 Fork、Star 与贡献代码！
