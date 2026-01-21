# 🏭 智能仓储行为分析系统  
**Smart Warehouse Analysis System**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)
![YOLO](https://img.shields.io/badge/AI-YOLOv11--Pose-red.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

> 面向 **工厂 / 仓储场景** 的 AI 行为分析与安全合规监控系统。  
> 基于 **YOLOv11 Pose（姿态估计）** 实时识别 **伸手 / 弯腰** 行为，支持 ROI 电子围栏、实时趋势可视化与证据留存（截图 + CSV）。
>
> 说明：本项目为个人学习与研究用途，素材仅用于演示。

---

## 🎬 演示（Demo）

### 动图预览（UI 录屏）
> 将动图放在仓库：`assets/demo.gif`（建议 < 15MB）

![demo](assets/demo.gif)

### 高清视频（Release）
- UI 演示视频（`ui_demo.mp4`）：https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/ui_demo.mp4  
- 示例输入视频（`video_1.mp4`）：https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/video_1.mp4  

（可选）部分环境支持用 HTML 内嵌视频预览：
```html
<video src="https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/ui_demo.mp4" controls width="100%"></video>
```

---

## ✨ 核心功能（Key Features）

### 🎯 高精度行为识别（YOLOv11 Pose）
- 基于 **Ultralytics YOLOv11-pose** 实现人体关键点检测  
- 实时识别：**伸手**、**弯腰** 等作业行为  
- 可视化增强：**赛博朋克双色高亮**（青色光晕 + 黄色核心），提升现场可读性

### 🧱 交互式电子围栏（ROI）
- 在视频画面中 **点击 4 个点** 绘制监控区域  
- 支持 **左 / 右货架** 监控区域（ROI）  
- ROI 自动保存至 `data/roi_config.json`，重启不丢失

### 📈 实时数据可视化（趋势 + 看板）
- 右侧集成 **Matplotlib 动态波形图**：实时展示作业频率趋势  
- 数据看板：在岗人数、违规计数（伸手/弯腰触发）

### 📸 证据留存（截图 + CSV）
- 触发行为时自动抓拍截图：`output/images/`  
- 自动生成带时间戳的 CSV：`output/report.csv`

---

## 📁 项目结构（Project Structure）

```text
Warehouse_Shelf_Posture_Recognition/
├── src/
│   ├── core_inference.py        # YOLO 推理与行为判定核心
│   └── ui/
│       ├── main_window.py       # 主界面（PySide6 / Qt）
│       └── ai_worker.py         # 推理工作线程
│
├── data/
│   └── roi_config.json          # ROI 配置（自动保存）
│
├── models/                      # 模型权重（不入库，通过 Release 下载）
├── output/
│   ├── images/                  # 行为证据截图（默认忽略追踪）
│   └── .gitkeep
│
├── assets/
│   └── demo.gif                 # UI 演示动图（README 预览）
│
├── scripts/
│   └── download_assets.py       # 一键下载模型/示例视频/UI 演示视频（Release）
│
├── requirements.txt
├── setup_resources.py
├── check_env.py
├── LICENSE
└── README.md
```

---

## 🚀 快速开始（Quick Start）

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

> 建议使用 **Python 3.10+**。CUDA 加速取决于你的 PyTorch 安装版本与驱动环境。

### 2️⃣ 下载模型与示例资源（Release）
本仓库不直接提交大文件（模型权重 / 示例视频 / UI 演示视频），可一键下载：

```bash
python scripts/download_assets.py --all
```

下载完成后将生成：
- `models/yolo11n-pose.pt`（推理权重）
- `data/video_1.mp4`（示例输入视频，用于快速运行）
- `output/ui_demo.mp4`（UI 演示视频，用于预览）

如只需要其中一类资源：
```bash
python scripts/download_assets.py --model
python scripts/download_assets.py --video
python scripts/download_assets.py --ui-demo
```

### 3️⃣ 运行主程序
```bash
python src/ui/main_window.py
```

---

## 🧭 操作指南（Usage Guide）

### ▶️ 启动与加载
- 启动后加载默认视频（`data/video_1.mp4`）或选择自定义视频  
- 模型与 ROI 配置自动初始化

### 🖱️ 绘制电子围栏（ROI）
1. 在视频画面中点击 **4 个点**  
2. 完成区域绘制（左 / 右货架）  
3. 配置自动保存到 `data/roi_config.json`

### 📊 查看趋势与看板
- 右侧波形图：作业频率趋势（实时更新）  
- 看板：在岗人数与违规计数统计  
- 视频画面：关键点骨架与高亮特效叠加

### 🗂️ 查看证据与报表
- 截图输出：`output/images/`  
- 报表输出：`output/report.csv`

---

## 🛠 技术栈（Tech Stack）

- Python 3.10+
- PySide6 (Qt for Python)
- OpenCV
- Ultralytics YOLOv11 (Pose)
- Matplotlib
- NumPy
- NVIDIA CUDA（可选）

---

## 📦 资源（Assets）

- 模型权重（baseline）：https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/yolo11n-pose.pt  
- 示例输入视频（运行用）：https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/video_1.mp4  
- UI 演示视频（预览用）：https://github.com/Bkp-126/Warehouse_Shelf_Posture_Recognition/releases/download/v0.1.0/ui_demo.mp4  

---

## 📄 License

MIT License

> 注：MIT 是开源许可协议名称（GitHub 会显示为“麻省理工学院许可”），不代表项目与任何机构存在隶属或背书关系。
