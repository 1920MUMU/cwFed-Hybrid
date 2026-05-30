"""
历史记录查看器
"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.experiment_recorder import ExperimentRecorder


class HistoryViewer(tk.Toplevel):
    """实验记录浏览窗口"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("实验记录")
        self.geometry("700x500")
        
        self.recorder = ExperimentRecorder()
        
        # 记录列表
        list_frame = tk.Frame(self)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        tk.Label(list_frame, text="历史实验", font=("Microsoft YaHei", 11, "bold")).pack()
        
        self.listbox = tk.Listbox(list_frame, width=45, height=25)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)
        
        # 详情面板
        detail_frame = tk.Frame(self)
        detail_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(detail_frame, height=8, font=("Consolas", 9))
        self.info_text.pack(fill=tk.X)
        
        # 图表
        self.fig = Figure(figsize=(5, 3.5), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, detail_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._refresh_list()
    
    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        for r in self.recorder.list_all():
            label = f"[{r['timestamp']}] {r['config']} | Acc:{r['best_acc']:.4f}"
            self.listbox.insert(tk.END, label)
    
    def _on_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
        
        records = self.recorder.list_all()
        if selection[0] >= len(records):
            return
        
        exp_id = records[selection[0]]["exp_id"]
        record = self.recorder.load(exp_id)
        if not record:
            return
        
        # 显示信息
        self.info_text.delete(1.0, tk.END)
        cfg = record["config"]
        info = (
            f"实验ID: {record['exp_id']}\n"
            f"时间: {record['timestamp'][:19]}\n"
            f"策略: {cfg['weight_mode']}+{cfg['aggregator_mode']}\n"
            f"数据: {cfg['dataset']} | {cfg['distribution']}\n"
            f"类别数: {cfg.get('num_classes', '?')}\n"                         # ←
            f"客户端: {cfg['num_clients']} | 轮数: {cfg['global_rounds']}\n"
            f"异质系数 α: {cfg.get('dirichlet_alpha', '-')}\n"                 # ←
            f"最佳准确率: {record['best_acc']:.4f} (第{record['best_round']}轮)\n"
            f"个性化增益: {record['best_gain']:+.4f}\n"
            f"总轮数: {record['total_rounds']} | 耗时: {record['elapsed_seconds']}s\n"
        )
        if cfg.get('dp_enabled'):
            info += (
                f"DP: {cfg['dp_mode']} | ε={cfg.get('dp_epsilon', '?')}\n"     # ←
            )
        self.info_text.insert(1.0, info)

        if cfg.get('wdr_enabled'):
            info += f"WDR: λ={cfg.get('wdr_lambda', '-')}\n"

        if cfg.get('reverse_mix'):
            info += "混合模式: 反转（高贡献信任本地）\n"
        
        # 绘制曲线
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_ylim(0, 1.0)
        rounds = [h["round"] for h in record["history"]]
        accs = [h["test_acc"] for h in record["history"]]
        f1s = [h["test_f1"] for h in record["history"]]
        gains = [h["gain"] for h in record["history"]]
        train_accs = [h.get("train_acc", 0) for h in record["history"]]
        fairness = [h.get("fairness", 0) for h in record["history"]]
        
        ax.plot(rounds, accs, 'g-', label="测试准确率", linewidth=2)
        ax.plot(rounds, f1s, 'r--', label="测试F1", linewidth=1.5)
        ax.plot(rounds, train_accs, 'b:', label="训练准确率", linewidth=1.5, alpha=0.7)
        ax.plot(rounds, gains, 'b-.', label="个性化增益", linewidth=1.5)
        ax.plot(rounds, fairness, 'm:', label="公平性(CV)", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("轮次")
        ax.set_ylabel("指标")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()