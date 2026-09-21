import sys
from PySide6.QtWidgets import QApplication
from .ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("仓储姿态辅助提醒")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
