"""
监控面板 - 修复图线错位问题
只显示测试 macro F1
"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .font_config import configure_chinese_font

configure_chinese_font()

class MonitorPanel(tk.Frame):
    # 可用指标（只保留测试 macro F1）
    METRICS = {
        "train_accuracy":         ("训练准确率",    "#90CAF9", "-",  1.0),
        "test_accuracy":          ("测试准确率",    "#4CAF50", "-",  2.0),
        "test_f1":                ("测试F1(macro)", "#E91E63", "--", 1.5),
        "personalization_gain":   ("个性化增益",    "#9C27B0", "-.", 1.5),
        "fairness_cv":            ("公平性(CV)",    "#F44336", ":",  1.5),
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.rounds = []
        self.data = {}       # {metric_name: [values]}
        self.max_rounds = 20
        self.active = []     # 当前显示的指标

        self._setup_ui()

    def _setup_ui(self):
        # 顶部信息栏
        self.info_frame = tk.Frame(self)
        self.info_frame.pack(fill=tk.X, pady=3)

        self.round_label = tk.Label(self.info_frame, text="轮次: 0/0",
                                     font=("Microsoft YaHei", 11, "bold"))
        self.round_label.pack(side=tk.LEFT, padx=8)
        
        # 动态指标标签容器
        self.info_container = tk.Frame(self.info_frame)
        self.info_container.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.info_labels = {}  # {metric_name: tk.Label}

        # 图表
        self.fig = Figure(figsize=(7, 4.5), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 日志
        log_frame = tk.Frame(self)
        log_frame.pack(fill=tk.X, pady=3)
        
        self.log_text = tk.Text(log_frame, height=4, font=("Consolas", 9),
                               state=tk.DISABLED, wrap=tk.NONE)
        self.log_text.pack(fill=tk.X)
        
        # 水平滚动条
        h_scroll = tk.Scrollbar(log_frame, orient=tk.HORIZONTAL, command=self.log_text.xview)
        h_scroll.pack(fill=tk.X)
        self.log_text.config(xscrollcommand=h_scroll.set)

    def configure_metrics(self, metric_names):
        """设置要显示的指标"""
        self.active = [m for m in metric_names if m in self.METRICS]
        self.data = {m: [] for m in self.active}

        # 重建标签
        for w in self.info_labels.values():
            w.destroy()
        self.info_labels.clear()

        for m in self.active:
            name, color, _, _ = self.METRICS[m]
            lbl = tk.Label(self.info_container, text=f"{name}: ---",
                          font=("Microsoft YaHei", 9), fg=color)
            lbl.pack(side=tk.LEFT, padx=5)
            self.info_labels[m] = lbl

    def reset(self, max_rounds):
        self.rounds = []
        self.data = {m: [] for m in self.active}
        self.max_rounds = max_rounds
        
        self.round_label.config(text=f"轮次: 0/{max_rounds}")
        for m, lbl in self.info_labels.items():
            name, _, _, _ = self.METRICS[m]
            lbl.config(text=f"{name}: ---")
        
        self._draw()
        self._clear_log()

    def update(self, stats: dict):
        """更新显示数据，跳过评估的轮次只更新轮次标签"""
        r = stats["round"]

        # 检查是否有实际评估数据（train_acc 在跳过评估时为上一轮的值）
        # 通过检查是否为新数据来判断：如果 test_acc 与上一轮完全相同，说明是跳过的
        is_evaluated = True
        if self.rounds and r > self.rounds[-1] + 1:
            # 中间跳过了轮次，但当前轮有评估，正常显示
            pass
        if not self.rounds:
            is_evaluated = True
        
        # 更新轮次标签（始终更新）
        self.round_label.config(text=f"轮次: {r}/{self.max_rounds}")

        # 更新指标标签（始终更新，反映最新状态）
        mapping = {
            "train_accuracy": stats.get("avg_train_acc", 0),
            "test_accuracy": stats.get("avg_test_acc", 0),
            "test_f1": stats.get("avg_test_f1", 0),
            "personalization_gain": stats.get("personalization_gain", 0),
            "fairness_cv": stats.get("fairness_cv", 0),
        }

        for m in self.active:
            val = mapping.get(m, 0)
            if m in self.info_labels:
                name, _, _, _ = self.METRICS[m]
                self.info_labels[m].config(text=f"{name}: {val:.4f}")

        # 只在实际评估的轮次添加数据点
        if not self.rounds or r != self.rounds[-1]:
            self.rounds.append(r)
            for m in self.active:
                self.data[m].append(mapping.get(m, 0))

        self._draw()
        self._append_log(stats)

    def _draw(self):
        """稳定绘图：只用一个子图"""
        self.fig.clear()
        
        if not self.active or not self.rounds:
            self.canvas.draw()
            return

        ax = self.fig.add_subplot(111)
        ax.set_xlim(1, self.max_rounds)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("轮次", fontsize=10)

        # 按权重排序，确保重要线在后面绘制（不被遮挡）
        sorted_metrics = sorted(self.active, 
                               key=lambda m: self.METRICS[m][3])

        for m in sorted_metrics:
            if m not in self.data or len(self.data[m]) < len(self.rounds):
                continue
            
            name, color, style, _ = self.METRICS[m]
            x = self.rounds[:len(self.data[m])]
            y = self.data[m]
            
            ax.plot(x, y, color=color, linestyle=style,
                   linewidth=2, label=name, alpha=0.9)

        # 图例
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def _append_log(self, stats):
        r = stats["round"]
        parts = [f"R{r:3d}"]
        for m in self.active:
            if m in self.data and self.data[m]:
                name, _, _, _ = self.METRICS[m]
                parts.append(f"{name}:{self.data[m][-1]:.4f}")
        line = " | ".join(parts) + "\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)