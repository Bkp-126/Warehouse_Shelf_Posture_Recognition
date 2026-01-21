import cv2
import time
import os
import json
import csv
import torch
import numpy as np
from datetime import datetime
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QImage
from src.core_inference import PoseDetector


class AIWorker(QThread):
    frame_signal = Signal(QImage)
    stats_signal = Signal(dict)
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, model_path, video_path):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.running = True

        self.show_roi = True
        self.show_skeleton = True
        self.show_angles = False

        # ROI 配置
        self.config_path = os.path.join(os.path.dirname(video_path), "roi_config.json")
        self.roi_left = []
        self.roi_right = []
        self.load_config()

        self.counters = {"reach": 0, "bend": 0}
        self.state_memory = {"is_reaching": False, "is_bending": False}

        # --- 📂 修复：绝对路径输出 ---
        # 获取当前运行脚本的根目录 (即 main_window.py 运行的地方)
        # os.getcwd() 通常是项目根目录 D:\AI_Project\Warehouse...
        self.project_root = os.getcwd()
        self.output_dir = os.path.join(self.project_root, "output")
        self.img_dir = os.path.join(self.output_dir, "images")
        self.csv_path = os.path.join(self.output_dir, "report.csv")

        # 自动创建目录
        os.makedirs(self.img_dir, exist_ok=True)

        # 🔴 强制打印路径，让你一眼看到
        print(f"\n[SYSTEM] 证据保存路径已锁定: {self.output_dir}")
        print(f"[SYSTEM] CSV 报表路径: {self.csv_path}")

        # 初始化 CSV 表头
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["时间", "事件类型", "当前计数", "图片文件名"])

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.roi_left = data.get("left", [])
                    self.roi_right = data.get("right", [])
            except:
                pass

    def save_config(self):
        data = {"left": self.roi_left, "right": self.roi_right}
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except:
            pass

    def save_evidence(self, frame, event_type, count):
        """保存证据"""
        try:
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            file_time = now.strftime("%Y%m%d_%H%M%S")

            img_name = f"{event_type}_{file_time}.jpg"
            img_full_path = os.path.join(self.img_dir, img_name)

            # 保存图片
            cv2.imwrite(img_full_path, frame)

            # 追加 CSV
            with open(self.csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([time_str, event_type, count, img_name])

            print(f"[SAVED] {img_full_path}")  # 控制台确认
            self.log_signal.emit(f"💾 已抓拍: {img_name}")

        except Exception as e:
            print(f"[ERROR] 保存失败: {e}")
            self.log_signal.emit(f"❌ 保存失败: {e}")

    @Slot(str, bool)
    def update_settings(self, key, value):
        if key == "roi":
            self.show_roi = value
        elif key == "skeleton":
            self.show_skeleton = value
        elif key == "angles":
            self.show_angles = value

    @Slot(str, list)
    def update_roi(self, side, points):
        if side == "left":
            self.roi_left = points
        elif side == "right":
            self.roi_right = points
        self.save_config()
        self.log_signal.emit(f"✅ {side} ROI 更新")

    def run(self):
        if not os.path.exists(self.video_path): return

        # 显卡选择
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            detector = PoseDetector(self.model_path, device=device)
            self.log_signal.emit(f"✅ 模型加载成功 ({device})")
        except Exception as e:
            self.log_signal.emit(f"❌ {e}")
            return

        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0: video_fps = 30
        frame_interval = 1.0 / video_fps

        self.log_signal.emit(f"🎥 监控已启动 (输出目录: output/)")

        while self.running:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results, _ = detector.process_frame(frame)
            canvas = frame.copy()
            h, w = canvas.shape[:2]

            # 坐标处理
            cnt_left = None;
            cnt_right = None
            if len(self.roi_left) >= 3:
                cnt_left = np.array([(int(nx * w), int(ny * h)) for (nx, ny) in self.roi_left], np.int32)
            if len(self.roi_right) >= 3:
                cnt_right = np.array([(int(nx * w), int(ny * h)) for (nx, ny) in self.roi_right], np.int32)

            trigger_left = False;
            trigger_right = False
            current_worker_count = len(results.boxes) if results.boxes else 0

            # 绘制与检测逻辑
            if results.keypoints is not None:
                for kps in results.keypoints.data:
                    kps = kps.cpu().numpy()

                    # 1. 伸手
                    left_wrist = (int(kps[9][0]), int(kps[9][1]))
                    right_wrist = (int(kps[10][0]), int(kps[10][1]))
                    p_trigger_L = False;
                    p_trigger_R = False

                    if kps[9][2] > 0.5:
                        if (cnt_left is not None and cv2.pointPolygonTest(cnt_left, left_wrist, False) > 0) or \
                                (cnt_right is not None and cv2.pointPolygonTest(cnt_right, left_wrist, False) > 0):
                            trigger_left = True;
                            p_trigger_L = True
                    if kps[10][2] > 0.5:
                        if (cnt_left is not None and cv2.pointPolygonTest(cnt_left, right_wrist, False) > 0) or \
                                (cnt_right is not None and cv2.pointPolygonTest(cnt_right, right_wrist, False) > 0):
                            trigger_right = True;
                            p_trigger_R = True

                    # 2. 弯腰 (显示逻辑)
                    if kps[6][2] > 0.5 and kps[12][2] > 0.5 and kps[14][2] > 0.5:
                        angle = detector.calculate_angle(kps[6][:2], kps[12][:2], kps[14][:2])
                        if angle < 140:
                            cv2.putText(canvas, f"BEND {int(angle)}", (int(kps[12][0]), int(kps[12][1] - 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 3. 高亮 (Reach)
                    color_core = (0, 255, 255);
                    color_glow = (255, 255, 0)
                    if p_trigger_L and kps[7][2] > 0.5:
                        cv2.line(canvas, left_wrist, (int(kps[7][0]), int(kps[7][1])), color_glow, 10)
                        cv2.line(canvas, left_wrist, (int(kps[7][0]), int(kps[7][1])), color_core, 4)
                        cv2.putText(canvas, "REACH", (left_wrist[0], left_wrist[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    color_core, 2)
                    if p_trigger_R and kps[8][2] > 0.5:
                        cv2.line(canvas, right_wrist, (int(kps[8][0]), int(kps[8][1])), color_glow, 10)
                        cv2.line(canvas, right_wrist, (int(kps[8][0]), int(kps[8][1])), color_core, 4)
                        cv2.putText(canvas, "REACH", (right_wrist[0], right_wrist[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    color_core, 2)

                    # 4. 骨架
                    if self.show_skeleton:
                        for x, y, conf in kps:
                            if conf > 0.5: cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
                        links = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
                                 (13, 15), (12, 14), (14, 16)]
                        for p1, p2 in links:
                            if p1 < len(kps) and p2 < len(kps) and kps[p1][2] > 0.5 and kps[p2][2] > 0.5:
                                cv2.line(canvas, (int(kps[p1][0]), int(kps[p1][1])), (int(kps[p2][0]), int(kps[p2][1])),
                                         (255, 0, 255), 2)

            # 状态机与保存
            frame_has_reach = trigger_left or trigger_right
            if frame_has_reach and not self.state_memory["is_reaching"]:
                self.counters["reach"] += 1
                self.log_signal.emit(f"⚠️ 伸手工作 +1")
                self.save_evidence(canvas, "REACH", self.counters["reach"])  # 保存!
                self.state_memory["is_reaching"] = True
            elif not frame_has_reach:
                self.state_memory["is_reaching"] = False

            # 弯腰逻辑 (需要遍历所有人检查是否有弯腰，此处简化为只要画面有弯腰就触发)
            # 在上面的循环里其实已经画了 BEND，这里做状态判定需要更严谨，
            # 为简化代码，我们假设前面检测到 BEND 文字绘制就视作弯腰
            # 但更严谨的是在循环里立 flag
            any_bend = False
            if results.keypoints is not None:
                for kps in results.keypoints.data:
                    kps = kps.cpu().numpy()
                    if kps[6][2] > 0.5 and kps[12][2] > 0.5 and kps[14][2] > 0.5:
                        if detector.calculate_angle(kps[6][:2], kps[12][:2], kps[14][:2]) < 140:
                            any_bend = True
                            break

            if any_bend and not self.state_memory["is_bending"]:
                self.counters["bend"] += 1
                self.log_signal.emit(f"⚠️ 弯腰工作 +1")
                self.save_evidence(canvas, "BEND", self.counters["bend"])  # 保存!
                self.state_memory["is_bending"] = True
            elif not any_bend:
                self.state_memory["is_bending"] = False

            # ROI 绘制
            if self.show_roi:
                if cnt_left is not None:
                    cv2.polylines(canvas, [cnt_left], True, (0, 0, 255) if trigger_left else (0, 255, 255), 2)
                    cv2.putText(canvas, "LEFT", tuple(cnt_left[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if cnt_right is not None:
                    cv2.polylines(canvas, [cnt_right], True, (0, 0, 255) if trigger_right else (0, 255, 255), 2)
                    cv2.putText(canvas, "RIGHT", tuple(cnt_right[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self.frame_signal.emit(qt_img)
            self.stats_signal.emit({"worker_count": current_worker_count, "reach_count": self.counters["reach"],
                                    "bend_count": self.counters["bend"]})

            t_end = time.time()
            if (t_end - t_start) < frame_interval: time.sleep(frame_interval - (t_end - t_start))

        cap.release()
        self.log_signal.emit("⏹ 停止")
        self.finished_signal.emit()

    def stop(self):
        self.running = False
        self.wait()