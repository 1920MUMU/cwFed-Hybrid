"""
EMA 平滑器
用于 Shapley 值的指数移动平均平滑

公式:
    alpha = base_alpha + (max_alpha - base_alpha) × (1 - e^{-local_epochs/τ})
    smoothed_t = alpha × smoothed_{t-1} + (1 - alpha) × raw_t

alpha 自动随本地轮数调整:
    - local_epochs=1 → alpha ≈ base_alpha（更信任新值）
    - local_epochs=10 → alpha ≈ max_alpha（更信任历史）
"""
import numpy as np


class EMASmoother:
    """
    Shapley 值的 EMA 平滑。
    
    支持两种模式：
    - smooth_gamma: 标量相似度（用于 fed 聚合权重 + 客户端混合比例）
    - smooth_per_class: 每类相似度（用于 cwfed 聚合权重）
    """

    def __init__(self, base_alpha: float = 0.3, max_alpha: float = 0.8,
                 tau: float = 5.0, num_clients: int = 10, num_classes: int = 10):
        self.base_alpha = base_alpha
        self.max_alpha = max_alpha
        self.tau = tau
        self.num_clients = num_clients
        self.num_classes = num_classes

        self.history_gamma = None       # [num_clients]
        self.history_per_class = None   # [num_clients, num_classes]
        self.gamma_round = 0
        self.per_class_round = 0

    def compute_alpha(self, local_epochs: int) -> float:
        """根据本地轮数计算平滑系数"""
        if local_epochs <= 0:
            return self.base_alpha
        decay = np.exp(-local_epochs / self.tau)
        return self.base_alpha + (self.max_alpha - self.base_alpha) * (1.0 - decay)

    def smooth_gamma(self, raw_gammas: list, local_epochs: int) -> list:
        """
        平滑标量相似度。
        
        输入:
            raw_gammas: list[float], 当前轮原始相似度
            local_epochs: 本地训练轮数
        输出:
            list[float], 平滑后的 gamma（clip 到 [0,1]，不做归一化）
        """
        alpha = self.compute_alpha(local_epochs)
        raw = np.array(raw_gammas, dtype=np.float64)

        if self.history_gamma is None or self.gamma_round == 0:
            smoothed = raw
        else:
            smoothed = alpha * self.history_gamma + (1.0 - alpha) * raw

        self.history_gamma = smoothed.copy()
        self.gamma_round += 1

        return np.clip(smoothed, 0.0, 1.0).tolist()

    def smooth_per_class(self, raw_values: list, local_epochs: int) -> list:
        """
        平滑每类相似度（不做归一化，归一化由 CWFedAggregator 负责）。
        
        输入:
            raw_values: list[list[float]], shape [N, K]
            local_epochs: 本地训练轮数
        输出:
            list[list[float]], shape [N, K], 平滑后（非负，不归一化）
        """
        alpha = self.compute_alpha(local_epochs)
        raw = np.array(raw_values, dtype=np.float64)

        if self.history_per_class is None or self.per_class_round == 0:
            smoothed = raw
        else:
            smoothed = alpha * self.history_per_class + (1.0 - alpha) * raw

        self.history_per_class = smoothed.copy()
        self.per_class_round += 1

        return np.maximum(smoothed, 0.0).tolist()

    def reset(self):
        self.history_gamma = None
        self.history_per_class = None
        self.gamma_round = 0
        self.per_class_round = 0

    def get_alpha(self, local_epochs: int) -> float:
        return self.compute_alpha(local_epochs)