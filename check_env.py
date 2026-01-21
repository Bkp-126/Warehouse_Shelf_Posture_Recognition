import os
import cv2
import torch
from ultralytics import YOLO


def check_environment():
    print("🛡️  开始 P0.4 最终环境自检...")
    print("=" * 60)

    # --- 1. 硬件检测 ---
    print(f"1. [硬件检测]")
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA 可用! 发现 {device_count} 个设备")
            print(f"   🚀 当前显卡: {device_name}")
            print(f"   📊 CUDA 版本: {torch.version.cuda}")
            device = 'cuda'
        else:
            print(f"   ⚠️ CUDA 不可用，系统将使用 CPU 运行。")
            print(f"      (这不影响功能，但帧率会较低，适合调试)")
            device = 'cpu'
    except Exception as e:
        print(f"   ❌ 检测 CUDA 时发生未知错误: {e}")
        device = 'cpu'

    # --- 2. 路径检测 ---
    print(f"\n2. [路径与文件验证]")
    paths = {
        "模型权重": "models/yolo11n-pose.pt",
        "测试视频": "data/video_1.mp4",
        "源码目录": "src"
    }

    all_files_exist = True
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"   ✅ {name.ljust(6)}: 存在 ({path})")
        else:
            print(f"   ❌ {name.ljust(6)}: 缺失! ({path})")
            all_files_exist = False

    if not all_files_exist:
        print("\n❌ 严重错误：关键文件缺失，请不要进入下一步，先修复文件缺失问题。")
        return

    # --- 3. 冒烟测试 (Smoke Test) ---
    print(f"\n3. [冒烟测试 - 模拟运行]")
    try:
        # A. 加载模型
        print("   ... 正在加载 YOLO 模型", end="")
        model = YOLO(paths["模型权重"])
        print(" -> ✅ 模型加载成功")

        # B. 读取视频一帧
        print("   ... 正在读取视频首帧", end="")
        cap = cv2.VideoCapture(paths["测试视频"])
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            print(f" -> ✅ 读取成功 (分辨率: {frame.shape[1]}x{frame.shape[0]})")
        else:
            print(" -> ❌ 视频读取失败 (可能是文件损坏)")
            return

        # C. 尝试推理
        print(f"   ... 尝试在 {device.upper()} 上运行一次推理", end="")
        # verbose=False 不打印多余日志
        results = model(frame, verbose=False, device=device)
        print(" -> ✅ 推理通道畅通")

        # 简单检查结果格式
        if results[0].keypoints is not None:
            print(f"      (检测到关键点数据结构，功能正常)")
        else:
            print(f"      (未检测到人像，但代码运行正常)")

    except Exception as e:
        print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=" * 60)
    print("🎉 P0 地基搭建阶段全部完成！")
    print("   你的开发环境非常健康，可以开始编写核心算法了。")


if __name__ == "__main__":
    check_environment()