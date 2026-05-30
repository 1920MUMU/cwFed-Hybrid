"""
联邦学习实验平台 - 主入口
支持 GUI 和命令行两种模式
"""
import sys
from utils import set_seed
from config import ExperimentConfig
from gui.main_window import MainWindow

def main():
    # 从 config 读取种子，保证一致
    from config import ExperimentConfig
    config = ExperimentConfig()
    set_seed(config.seed)

    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()