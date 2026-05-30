"""
WDR: Weight Distribution Regularization
参考 cwFedAvg 论文，约束分类层权重范数分布接近本地类别分布

论文公式:
    loss = λ × || norm(W) - p ||²
    其中 norm(W)_c = ||W_c|| / Σ_k ||W_k||
    p_c = n_c / Σ_k n_k  (本地类别分布)
"""
import torch


class WDRLoss:
    """权重分布正则化损失"""

    def __init__(self, lambda_reg: float = 1.0):
        """
        参数:
            lambda_reg: 正则化强度 λ
        """
        self.lambda_reg = lambda_reg

    def compute(self, output_layer: torch.nn.Module,
                class_distribution: torch.Tensor) -> torch.Tensor:
        """
        计算 WDR 损失。

        参数:
            output_layer: 分类层（nn.Linear），weight shape [K, D]
            class_distribution: 本地类别分布 p_c，shape [K]，和应为 1
        返回:
            scalar tensor
        """
        weight = output_layer.weight  # [K, D]

        # 每类权重的 L2 范数
        weight_norms = torch.norm(weight, p=2, dim=1)  # [K]

        # 直接归一化（不用 softmax）
        norm_sum = weight_norms.sum()
        if norm_sum < 1e-8:
            return torch.tensor(0.0)

        approx_dist = weight_norms / norm_sum  # [K]

        # 欧氏距离
        wdr_loss = torch.dist(class_distribution, approx_dist, p=2)

        return self.lambda_reg * wdr_loss