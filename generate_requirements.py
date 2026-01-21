import importlib.metadata
import sys


def check_and_generate():
    """
    检测当前环境已安装的库版本，并生成 requirements.txt
    """
    # 定义我们需要用到的库 (PyPI名称 : 导入检查用的名称)
    # 注意：opencv-python 在 pip 中叫 opencv-python，但在 import 时叫 cv2 (这里通过 metadata 检查更准确)
    required_packages = [
        "ultralytics",
        "opencv-python",
        "numpy",
        "PySide6",
        "tqdm",
        "scipy"
    ]

    print(f"🔍 开始检测当前环境依赖状态...")
    print("-" * 40)

    installed_lines = []
    missing_packages = []

    for package in required_packages:
        try:
            # 获取已安装库的版本号
            version = importlib.metadata.version(package)
            line = f"{package}=={version}"
            installed_lines.append(line)
            print(f"   [✅ 已安装] {package.ljust(15)} : {version}")
        except importlib.metadata.PackageNotFoundError:
            # 尝试处理 opencv 的特殊情况 (有时候包名可能是 opencv-python-headless 等)
            if package == "opencv-python":
                try:
                    version = importlib.metadata.version("opencv-python-headless")
                    line = f"opencv-python-headless=={version}"
                    installed_lines.append(line)
                    print(f"   [✅ 已安装] opencv-headless  : {version}")
                    continue
                except:
                    pass

            print(f"   [❌ 未找到] {package}")
            missing_packages.append(package)

    print("-" * 40)

    if missing_packages:
        print("⚠️  检测到以下库缺失，请手动运行 pip install 安装它们：")
        for p in missing_packages:
            print(f"   pip install {p}")
        print("\n❌ requirements.txt 未生成，请先补全环境。")
    else:
        # 全部检测通过，写入文件
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(installed_lines))
        print(f"🎉 成功！已将你当前的库版本写入 requirements.txt")
        print(f"📄 文件内容预览：\n")
        print("\n".join(installed_lines))
        print("\n✅ P0.2 环境依赖配置完成。")


if __name__ == "__main__":
    check_and_generate()