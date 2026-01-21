import os
import shutil
from ultralytics import YOLO


def prepare_resources():
    print("🚀 开始准备核心资源...")

    # --- 1. 下载模型 ---
    target_model_path = "models/yolo11n-pose.pt"

    if os.path.exists(target_model_path):
        print(f"   [=] 模型已存在: {target_model_path}")
    else:
        print("   [⬇️] 正在下载 yolo11n-pose.pt (首次运行可能需要几分钟)...")
        try:
            # 加载模型会自动触发下载
            # 注意：ultralytics 默认下载到当前目录
            model = YOLO("yolo11n-pose.pt")

            # 将下载的文件移动到 models 文件夹
            if os.path.exists("yolo11n-pose.pt"):
                shutil.move("yolo11n-pose.pt", target_model_path)
                print(f"   [✅] 模型已移动到: {target_model_path}")
            else:
                # 某些版本可能下载后名字不同，或者已经缓存
                print("   [!] 未在根目录找到模型文件，可能已在缓存中或下载失败。")
                print("       请检查是否有 yolo11n-pose.pt 文件生成。")
        except Exception as e:
            print(f"   [❌] 模型下载出错: {e}")
            print("       解决方案：请手动访问 GitHub (ultralytics/assets) 下载 yolo11n-pose.pt 并放入 models 目录。")

    # --- 2. 检查视频 ---
    target_video_path = "data/video_1.mp4"

    if os.path.exists(target_video_path):
        print(f"   [✅] 测试视频已就绪: {target_video_path}")
    else:
        print(f"   [⚠️] 未检测到测试视频: {target_video_path}")
        print("   👉 行动指南：")
        print("       请找一个包含【人体全身】的视频文件（最好有下蹲、弯腰动作）。")
        print("       将其重命名为 video_1.mp4")
        print("       并放入 data/ 文件夹中。")

    print("\n✅ P0.3 资源检查脚本运行结束。")


if __name__ == "__main__":
    prepare_resources()