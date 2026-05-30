"""
主窗口 - 极简布局
"""
import tkinter as tk
from tkinter import ttk
import threading
import traceback
from config import ExperimentConfig
from foundation import SyntheticDataGenerator, CIFAR10Loader, DataPartitioner
from foundation.model import create_model
from federated import Client
from orchestration.server import Server
from utils import set_seed
from utils import ExperimentRecorder
from .param_panel import ParamPanel
from .monitor_panel import MonitorPanel


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("联邦学习实验平台 - Shapley-CWFedAvg")
        self.geometry("1050x650")
        self.minsize(900, 550)

        self.option_add("*Font", ("Microsoft YaHei", 9))

        # 状态
        self.train_thread = None
        self.is_training = False
        self.server = None
        self.clients = None
        self.total_rounds = 20  # 缓存，避免跨线程访问 GUI 控件

        self._setup_ui()

    def _setup_ui(self):
        """极简布局：左面板 | 右面板"""
        # 左侧参数面板（固定宽度）
        self.param_panel = ParamPanel(self)
        self.param_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.param_panel.bind_start(self._start_experiment)
        self.param_panel.bind_stop(self._stop_experiment)

        # 分隔线
        ttk.Separator(self, orient='vertical').pack(
            side=tk.LEFT, fill=tk.Y, padx=2
        )

        # 右侧监控面板（自适应）
        self.monitor_panel = MonitorPanel(self)
        self.monitor_panel.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        # 底部状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = tk.Label(
            self, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W,
            font=("Microsoft YaHei", 9)
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        tk.Button(btn_frame, text="📊 历史记录", 
                command=self._open_history).pack(side=tk.RIGHT, padx=5)

        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _open_history(self):
        from .history_viewer import HistoryViewer
        HistoryViewer(self)

    # ===== 实验控制 =====

    def _start_experiment(self):
        """启动实验"""
        if self.is_training:
            return

        config = self.param_panel.get_config()
        # 检查是否已有记录
        recorder = ExperimentRecorder()
        if recorder.exists(config):
            if not tk.messagebox.askyesno(
                "已有记录",
                "相同参数下已有实验记录，是否重新训练？\n"
                "选择「否」可查看历史记录。"
            ):
                self._open_history()
                return
        self.total_rounds = config.global_rounds  # 缓存

        self.monitor_panel.configure_metrics(config.display_metrics)
        self.monitor_panel.reset(self.total_rounds)

        self._set_buttons_state(start=False)
        self.is_training = True
        self.status_var.set("正在初始化...")

        self.train_thread = threading.Thread(
            target=self._run_experiment, args=(config,), daemon=True
        )
        self.train_thread.start()

    def _stop_experiment(self):
        """停止实验"""
        self.is_training = False
        self.status_var.set("已停止")
        self._set_buttons_state(start=True)

    def _run_experiment(self, config: ExperimentConfig):
        """在子线程中运行实验"""
        try:
            self._safe_update_status("准备数据...")
            set_seed(config.seed)

            if config.dataset == "synthetic":
                gen = SyntheticDataGenerator(num_classes=config.num_classes, seed=config.seed)
                total = config.num_clients * config.samples_per_client
                X, y = gen.generate(total)
                # 用不同种子生成独立测试集
                test_gen = SyntheticDataGenerator(num_classes=config.num_classes, seed=config.seed + 99999)
                total_test = config.num_clients * config.test_samples_per_client
                X_test, y_test = test_gen.generate_global_test(total_test)
            else:
                loader = CIFAR10Loader(data_dir="data")
                X, y, X_test, y_test = loader.load_raw()

            self._safe_update_status("划分数据...")
            partitioner = DataPartitioner(config)
            client_data = partitioner.partition(X, y, X_test, y_test)

            # 打印噪声信息
            if config.distribution == "noisy":
                noise_info = partitioner.get_client_noise_info()
                for cid, info in noise_info.items():
                    if info["total_samples"] > 0:
                        actual_ratio = info["noisy_samples"] / info["total_samples"]
                        print(f"  Client {cid}: 噪声 {actual_ratio:.2%} "
                            f"({info['noisy_samples']}/{info['total_samples']})")

            self._safe_update_status("创建客户端...")
            base_model = create_model(config)
            self.clients = [
                Client(cid, client_data[cid], base_model, config)
                for cid in range(config.num_clients)
            ]

            self._safe_update_status("训练中...")
            self.server = Server(self.clients, config)

            def on_round_end(stats):
                if self.is_training:
                    self.after(0, self._on_round_end, stats)

            self.server.run(callback=on_round_end)
            self.after(0, self._on_experiment_done)

        except Exception as e:
            traceback.print_exc()
            self.after(0, self._on_error, str(e))


    def _on_experiment_done(self):
        """实验完成"""
        self.is_training = False
        self._set_buttons_state(start=True)
        if self.server and self.server.history:
            best = self.server.get_best_round()
            self.status_var.set(
                f"完成! 最佳准确率: {best['avg_test_acc']:.4f} "
                f"(第{best['round']}轮) | "
                f"增益: {best['personalization_gain']:+.4f}"
            )
        else:
            self.status_var.set("训练完成（无数据）")

    # ===== GUI 回调（主线程）=====

    def _on_round_end(self, stats: dict):
        """每轮结束后更新 GUI"""
        self.monitor_panel.update(stats)
        self.status_var.set(
            f"轮次 {stats['round']}/{self.total_rounds} | "
            f"测试准确率: {stats['avg_test_acc']:.4f}"
        )

    def _on_error(self, msg: str):
        """出错"""
        self.is_training = False
        self._set_buttons_state(start=True)
        self.status_var.set(f"错误: {msg}")

    def _on_close(self):
        """关闭窗口"""
        self._stop_experiment()
        self.destroy()

    # ===== 辅助方法 =====

    def _set_buttons_state(self, start: bool):
        """设置按钮状态"""
        if start:
            self.param_panel.start_btn.config(state=tk.NORMAL)
            self.param_panel.stop_btn.config(state=tk.DISABLED)
        else:
            self.param_panel.start_btn.config(state=tk.DISABLED)
            self.param_panel.stop_btn.config(state=tk.NORMAL)

    def _safe_update_status(self, msg: str):
        """线程安全地更新状态栏"""
        self.after(0, lambda: self.status_var.set(msg))