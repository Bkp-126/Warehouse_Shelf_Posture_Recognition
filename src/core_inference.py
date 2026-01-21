import cv2
import os
import time
import math
import numpy as np
from ultralytics import YOLO


class PoseDetector:
    """
    核心姿态检测类
    包含：模型推理、几何计算
    """

    def __init__(self, model_path, device='cpu'):
        self.device = device
        print(f"[Core] 正在加载模型: {model_path} (设备: {device})...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        try:
            self.model = YOLO(model_path)
            # 预热
            self.model(data=None, verbose=False, device=self.device)
            print("[Core] 模型加载完成")
        except Exception as e:
            print(f"[Core] 模型加载失败: {e}")
            raise e

    def process_frame(self, frame):
        """
        推理单帧
        """
        if frame is None:
            return None, None

        # 推理
        results = self.model(frame, verbose=False, device=self.device, conf=0.5)

        # 获取绘图结果 (这是原图分辨率)
        annotated_frame = results[0].plot()

        return results[0], annotated_frame

    @staticmethod
    def calculate_angle(a, b, c):
        """
        计算三个点之间的夹角 (角度制)
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        degree = np.degrees(angle)

        return degree


# --- 调试代码 (包含视频保存功能) ---
def debug_run():
    # 1. 配置路径
    video_path = "data/video_1.mp4"
    model_path = "models/yolo11n-pose.pt"
    output_path = "output/debug_output.mp4"  # 结果保存路径

    if not os.path.exists(video_path):
        print(f"错误: 找不到视频 {video_path}")
        return

    # 2. 初始化
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = PoseDetector(model_path, device)
    cap = cv2.VideoCapture(video_path)

    # 获取视频属性，用于初始化录制器
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化视频写入器 (mp4v 编码)
    print(f"\n 准备录制视频到: {output_path}")
    print(f"   分辨率: {w_orig}x{h_orig}, FPS: {fps}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w_orig, h_orig))

    print("🚀 开始推理循环... ")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ 视频处理完毕")
            break

        # A. 推理
        result, output_img = detector.process_frame(frame)

        # B. 几何计算 (演示：计算右臂角度)
        if result.keypoints is not None and result.keypoints.data.shape[1] > 0:
            kpts = result.keypoints.data[0].cpu().numpy()
            # 6: Shoulder, 8: Elbow, 10: Wrist
            if len(kpts) > 10:
                pt_s, pt_e, pt_w = kpts[6][:2], kpts[8][:2], kpts[10][:2]
                conf = kpts[8][2]
                if conf > 0.5:
                    angle = detector.calculate_angle(pt_s, pt_e, pt_w)
                    # 绘制角度
                    cv2.putText(output_img, f"Angle: {int(angle)}",
                                (int(pt_e[0]), int(pt_e[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # C. 写入视频文件 (必须是原图大小)
        writer.write(output_img)

        # D. 屏幕显示 (缩放后显示，防止爆屏)
        show_w = 1280
        show_h = int(h_orig * (show_w / w_orig))
        frame_show = cv2.resize(output_img, (show_w, show_h))
        cv2.imshow("Processing... (Press 'q' to stop)", frame_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户手动停止")
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"   已处理 {frame_idx} 帧...")

    # 清理资源
    cap.release()
    writer.release()  # 这一步至关重要，否则视频无法播放
    cv2.destroyAllWindows()

    print(f"\n P1 阶段完成！演示视频已保存至: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    debug_run()