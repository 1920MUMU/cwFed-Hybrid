"""
参数调节面板 - 使用标签页组织参数
从 ExperimentConfig 读取默认值
"""
import tkinter as tk
from tkinter import ttk
from config import ExperimentConfig


class ParamPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, width=280)
        self.pack_propagate(False)

        # 从 config 获取默认值
        defaults = ExperimentConfig()

        title_font = ("Microsoft YaHei", 12, "bold")
        section_font = ("Microsoft YaHei", 10, "bold")
        label_font = ("Microsoft YaHei", 9)

        title = tk.Label(self, text="实验参数", font=title_font)
        title.pack(pady=8)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 五个标签页（传入 defaults）
        self.data_tab = tk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="数据")
        self._build_data_tab(self.data_tab, section_font, label_font, defaults)

        self.dist_tab = tk.Frame(self.notebook)
        self.notebook.add(self.dist_tab, text="分布")
        self._build_dist_tab(self.dist_tab, section_font, label_font, defaults)

        self.strategy_tab = tk.Frame(self.notebook)
        self.notebook.add(self.strategy_tab, text="策略")
        self._build_strategy_tab(self.strategy_tab, section_font, label_font, defaults)

        self.train_tab = tk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="训练")
        self._build_train_tab(self.train_tab, section_font, label_font, defaults)

        self.display_tab = tk.Frame(self.notebook)
        self.notebook.add(self.display_tab, text="显示")
        self._build_display_tab(self.display_tab, section_font, label_font, defaults)

        self._build_control_buttons()

    # ===== 数据标签页 =====
    def _build_data_tab(self, parent, section_font, label_font, defaults):
        self._add_section(parent, "数据集", section_font)

        self.dataset_var = tk.StringVar(value=defaults.dataset)
        self._add_combobox(parent, "类型:", self.dataset_var,
                          ["synthetic", "cifar10"], label_font)

        self.num_classes_var = tk.IntVar(value=defaults.num_classes)
        self._add_spinbox(parent, "类别数:", self.num_classes_var, 2, 100, label_font)

        self.samples_var = tk.IntVar(value=defaults.samples_per_client)
        self._add_spinbox(parent, "样本数/客户端:", self.samples_var, 10, 5000, label_font)

    # ===== 分布标签页 =====
    def _build_dist_tab(self, parent, section_font, label_font, defaults):
        self._add_section(parent, "数据分布", section_font)

        self.dist_var = tk.StringVar(value=defaults.distribution)
        self._add_combobox(parent, "策略:", self.dist_var,
                          ["uniform", "heterogeneous", "noisy"], label_font)

        self.dirichlet_var = tk.DoubleVar(value=defaults.dirichlet_alpha)
        self._add_scale(parent, "Dirichlet α:", self.dirichlet_var, 0.1, 5.0, 0.1, label_font)

        # 噪声配置
        self._add_section(parent, "噪声配置", section_font)

        self.noise_ratio_var = tk.DoubleVar(value=defaults.noise_ratio)
        self._add_scale(parent, "噪声比例:", self.noise_ratio_var, 0.0, 0.5, 0.05, label_font)

        self.noise_hetero_var = tk.BooleanVar(value=defaults.noise_heterogeneous)
        self._add_checkbox(parent, "客户端间异质", self.noise_hetero_var,
                          "不同客户端噪声不同", label_font)

        self.noise_intra_class_var = tk.BooleanVar(
            value=getattr(defaults, 'noise_intra_class', False)
        )
        self._add_checkbox(parent, "类间异质", self.noise_intra_class_var,
                          "同一客户端不同类噪声不同", label_font)

        self.noise_hetero_alpha_var = tk.DoubleVar(
            value=getattr(defaults, 'noise_hetero_alpha', 0.5)
        )
        self._add_scale(parent, "噪声异质α:", self.noise_hetero_alpha_var, 0.1, 3.0, 0.1, label_font)

    # ===== 策略标签页 =====
    def _build_strategy_tab(self, parent, section_font, label_font, defaults):
        self._add_section(parent, "聚合策略", section_font)

        self.weight_var = tk.StringVar(value=defaults.weight_mode)
        self._add_combobox(parent, "权重计算:", self.weight_var,
                        ["avg", "shapley", "hybrid"], label_font)

        self.agg_var = tk.StringVar(value=defaults.aggregator_mode)
        self._add_combobox(parent, "聚合方式:", self.agg_var,
                          ["fed", "cwfed"], label_font)

        # Shapley 高级配置
        self._add_section(parent, "Shapley 高级", section_font)

        self.ema_enabled_var = tk.BooleanVar(
            value=getattr(defaults, 'ema_enabled', False)
        )
        self._add_checkbox(parent, "EMA 平滑", self.ema_enabled_var,
                          "指数移动平均平滑 Shapley 值", label_font)

        self.ema_base_alpha_var = tk.DoubleVar(
            value=getattr(defaults, 'ema_base_alpha', 0.3)
        )
        self._add_scale(parent, "EMA 基础α:", self.ema_base_alpha_var, 0.1, 0.9, 0.05, label_font)

        self.ema_max_alpha_var = tk.DoubleVar(
            value=getattr(defaults, 'ema_max_alpha', 0.8)
        )
        self._add_scale(parent, "EMA 最大α:", self.ema_max_alpha_var, 0.1, 0.99, 0.05, label_font)

        self._add_section(parent, "WDR 正则化", section_font)

        self.wdr_enabled_var = tk.BooleanVar(
            value=getattr(defaults, 'wdr_enabled', False)
        )
        self._add_checkbox(parent, "启用 WDR", self.wdr_enabled_var,
                        "权重分布正则化", label_font)

        self.wdr_lambda_var = tk.DoubleVar(
            value=getattr(defaults, 'wdr_lambda', 1.0)
        )
        self._add_scale(parent, "WDR 强度:", self.wdr_lambda_var, 0.1, 20.0, 0.1, label_font)

        self.reverse_mix_var = tk.BooleanVar(
            value=getattr(defaults, 'reverse_mix', False)
        )
        self._add_checkbox(parent, "反转混合", self.reverse_mix_var,
                        "高贡献客户端更信任本地模型", label_font)
                        
        self.sliding_window_var = tk.BooleanVar(
            value=getattr(defaults, 'sliding_window_enabled', False)
        )
        self._add_checkbox(parent, "滑动窗口", self.sliding_window_var,
                        "高权重客户端使用更长窗口平滑", label_font)
        
        # 隐私标签页
        self.privacy_tab = tk.Frame(self.notebook)
        self.notebook.add(self.privacy_tab, text="隐私")
        self._build_privacy_tab(self.privacy_tab, section_font, label_font, defaults)

        # 说明
        self._add_section(parent, "说明", section_font)
        desc_text = (
            "avg + fed: 标准联邦平均\n"
            "avg + cwfed: 类级别个性化\n"
            "shapley + fed: Shapley加权+个性化混合\n"
            "shapley + cwfed: 融合策略\n"
            "EMA: 平滑Shapley值，本地轮数越大越信任历史"
        )
        desc = tk.Label(parent, text=desc_text, font=("Microsoft YaHei", 8),
                       fg="#666", justify=tk.LEFT)
        desc.pack(anchor=tk.W, padx=10, pady=5)

    def _build_privacy_tab(self, parent, section_font, label_font, defaults):
        self._add_section(parent, "差分隐私", section_font)
        
        self.dp_enabled_var = tk.BooleanVar(value=getattr(defaults, 'dp_enabled', False))
        self._add_checkbox(parent, "启用 DP", self.dp_enabled_var, "", label_font)
        
        self.dp_mode_var = tk.StringVar(value=getattr(defaults, 'dp_mode', 'classic'))
        self._add_combobox(parent, "模式:", self.dp_mode_var, ["classic", "ladp"], label_font)
        
        self.dp_epsilon_var = tk.DoubleVar(value=getattr(defaults, 'dp_epsilon', 8.0))
        self._add_scale(parent, "ε:", self.dp_epsilon_var, 0.1, 20.0, 0.1, label_font)
        
        # 说明
        tip = tk.Label(parent,
            text="ε 越小隐私越强，噪声越大\n两种模式共用 ε，隐私消耗可对比",
            font=("Microsoft YaHei", 7), fg="#888", justify=tk.LEFT)
        tip.pack(anchor=tk.W, padx=5, pady=5)

    # ===== 训练标签页 =====
    def _build_train_tab(self, parent, section_font, label_font, defaults):
        self._add_section(parent, "客户端", section_font)

        self.clients_var = tk.IntVar(value=defaults.num_clients)
        self._add_spinbox(parent, "数量:", self.clients_var, 2, 50, label_font)

        self.local_epochs_var = tk.IntVar(value=defaults.local_epochs)
        self._add_spinbox(parent, "本地轮数:", self.local_epochs_var, 1, 20, label_font)

        self.batch_size_var = tk.IntVar(value=defaults.batch_size)
        self._add_spinbox(parent, "批大小:", self.batch_size_var, 8, 256, label_font)

        self.lr_var = tk.DoubleVar(value=defaults.learning_rate)
        self._add_combobox(parent, "学习率:", self.lr_var,
                          [0.1, 0.05, 0.01, 0.005, 0.001], label_font)

        self._add_section(parent, "全局", section_font)
    
        self.rounds_var = tk.IntVar(value=defaults.global_rounds)
        self._add_spinbox(parent, "全局轮数:", self.rounds_var, 1, 200, label_font)
        
        # 提前停止
        self._add_section(parent, "提前停止", section_font)
        
        early_stop_default = getattr(defaults, 'early_stop_acc', 1.0)
        self.early_stop_var = tk.DoubleVar(value=early_stop_default)
        self._add_scale(parent, "停止阈值:", self.early_stop_var, 0.5, 1.0, 0.01, label_font)
    
        # 说明
        tip = tk.Label(parent, text="达到阈值后提前停止训练（1.0=不启用）",
                    font=("Microsoft YaHei", 7), fg="#888")
        tip.pack(anchor=tk.W, padx=5)

    def _build_display_tab(self, parent, section_font, label_font, defaults):
        self._add_section(parent, "图表指标", section_font)

        cb_frame = tk.Frame(parent)
        cb_frame.pack(fill=tk.X, padx=5, pady=5)

        # 直接访问属性
        default_metrics = defaults.display_metrics

        self.show_train_acc = tk.BooleanVar(value="train_accuracy" in default_metrics)
        tk.Checkbutton(cb_frame, text="训练准确率", variable=self.show_train_acc,
                    font=label_font).pack(anchor=tk.W, pady=1)

        self.show_test_acc = tk.BooleanVar(value="test_accuracy" in default_metrics)
        tk.Checkbutton(cb_frame, text="测试准确率", variable=self.show_test_acc,
                    font=label_font).pack(anchor=tk.W, pady=1)

        self.show_test_f1 = tk.BooleanVar(value="test_f1" in default_metrics)
        tk.Checkbutton(cb_frame, text="测试F1 (macro)", variable=self.show_test_f1,
                    font=label_font).pack(anchor=tk.W, pady=1)

        self._add_section(parent, "分析指标", section_font)

        cb_frame2 = tk.Frame(parent)
        cb_frame2.pack(fill=tk.X, padx=5, pady=5)

        self.show_gain = tk.BooleanVar(value="personalization_gain" in default_metrics)
        tk.Checkbutton(cb_frame2, text="个性化增益", variable=self.show_gain,
                    font=label_font).pack(anchor=tk.W, pady=1)

        self.show_fairness = tk.BooleanVar(value="fairness_cv" in default_metrics)
        tk.Checkbutton(cb_frame2, text="公平性(CV)", variable=self.show_fairness,
                    font=label_font).pack(anchor=tk.W, pady=1)

    # ===== 控制按钮 =====
    def _build_control_buttons(self):
        frame = tk.Frame(self)
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.start_btn = tk.Button(frame, text="▶ 开始训练",
                                   bg="#4CAF50", fg="white",
                                   font=("Microsoft YaHei", 11, "bold"),
                                   height=2)
        self.start_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = tk.Button(frame, text="■ 停止训练",
                                  bg="#f44336", fg="white",
                                  font=("Microsoft YaHei", 11),
                                  height=2, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

    # ===== 辅助方法（不变）=====
    def _add_section(self, parent, text, font):
        sep = ttk.Separator(parent, orient='horizontal')
        sep.pack(fill=tk.X, pady=(8, 3))
        tk.Label(parent, text=text, font=font, fg="#555").pack(anchor=tk.W, padx=5)

    def _add_spinbox(self, parent, label, var, min_val, max_val, font):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=1, padx=5)
        tk.Label(frame, text=label, width=12, anchor=tk.W, font=font).pack(side=tk.LEFT)
        tk.Spinbox(frame, textvariable=var, from_=min_val, to=max_val,
                   width=8, font=font).pack(side=tk.RIGHT)

    def _add_scale(self, parent, label, var, min_val, max_val, resolution, font):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=1, padx=5)
        tk.Label(frame, text=label, width=12, anchor=tk.W, font=font).pack(side=tk.LEFT)
        tk.Scale(frame, variable=var, from_=min_val, to=max_val,
                resolution=resolution, orient=tk.HORIZONTAL,
                length=100, showvalue=True, font=font).pack(side=tk.RIGHT)

    def _add_combobox(self, parent, label, var, values, font):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=1, padx=5)
        tk.Label(frame, text=label, width=12, anchor=tk.W, font=font).pack(side=tk.LEFT)
        ttk.Combobox(frame, textvariable=var, values=values,
                    state="readonly", width=8, font=font).pack(side=tk.RIGHT)

    def _add_checkbox(self, parent, label, var, tooltip, font):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=1, padx=5)
        tk.Checkbutton(frame, text=label, variable=var, font=font).pack(anchor=tk.W)

    # ===== 获取配置 =====
    def get_config(self) -> ExperimentConfig:
        display_metrics = []
        if self.show_train_acc.get():
            display_metrics.append("train_accuracy")
        if self.show_test_acc.get():
            display_metrics.append("test_accuracy")
        if self.show_test_f1.get():
            display_metrics.append("test_f1")
        if self.show_gain.get():
            display_metrics.append("personalization_gain")
        if self.show_fairness.get():
            display_metrics.append("fairness_cv")

        return ExperimentConfig(
            dataset=self.dataset_var.get(),
            num_classes=self.num_classes_var.get(),
            samples_per_client=self.samples_var.get(),
            distribution=self.dist_var.get(),
            noise_ratio=self.noise_ratio_var.get(),
            noise_heterogeneous=self.noise_hetero_var.get(),
            noise_intra_class=self.noise_intra_class_var.get(),
            noise_hetero_alpha=self.noise_hetero_alpha_var.get(),
            dirichlet_alpha=self.dirichlet_var.get(),
            weight_mode=self.weight_var.get(),
            aggregator_mode=self.agg_var.get(),
            ema_enabled=self.ema_enabled_var.get(),
            ema_base_alpha=self.ema_base_alpha_var.get(),
            ema_max_alpha=self.ema_max_alpha_var.get(),
            num_clients=self.clients_var.get(),
            global_rounds=self.rounds_var.get(),
            local_epochs=self.local_epochs_var.get(),
            batch_size=self.batch_size_var.get(),
            learning_rate=self.lr_var.get(),
            early_stop_acc=self.early_stop_var.get(),
            display_metrics=display_metrics,
            dp_enabled=self.dp_enabled_var.get(),
            dp_mode=self.dp_mode_var.get(),
            dp_epsilon=self.dp_epsilon_var.get(),
            wdr_enabled=self.wdr_enabled_var.get(),
            wdr_lambda=self.wdr_lambda_var.get(),
            reverse_mix=self.reverse_mix_var.get(),
            sliding_window_enabled=self.sliding_window_var.get(),
        )

    def bind_start(self, callback):
        self.start_btn.config(command=callback)

    def bind_stop(self, callback):
        self.stop_btn.config(command=callback)