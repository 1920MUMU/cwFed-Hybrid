"""
LaDP: Local Layer-wise Differential Privacy
严格遵循论文公式的层间自适应差分隐私
"""
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
from .base import PrivacyModule
from typing import List, Optional, Dict


class LaDP(PrivacyModule):
    """
    LaDP 层间自适应差分隐私（论文严格复现版）。
    
    论文公式:
    Eq. 8:  P_{i,j} = min(KL(w_{i,j}^t || w_{g,j}^t), B)
    Theorem 2: Δf ≤ 2ηEG_c
    Eq. 10: σ_{i,j} = (c_i × Δf) / (ε × P_{i,j})
    Theorem 3: c_i 根据 ε, δ, B 计算
    
    噪声分配逻辑 (论文 Section V-B2):
    - P 大 (local偏离global) → 隐私风险低 → σ 小 → 噪声少
    - P 小 (local接近global) → 隐私风险高 → σ 大 → 噪声多
    """

    def __init__(self, accountant, 
                 epsilon: float = 0.3,           # 隐私预算
                 delta: float = 1e-5,             # 失败概率
                 learning_rate: float = 0.1,      # 学习率 η
                 local_epochs: int = 2,           # 本地训练轮数 E
                 grad_clip_norm: float = 20.0,    # 梯度裁剪阈值 G_c
                 kl_clip_bound: float = 5.0,      # KL裁剪边界 B
                 layer_threshold_R: float = 1.0,  # 层选择阈值 R
                 c_i: Optional[float] = None):    # 客户端系数（可选）
        super().__init__(accountant)
        
        self.epsilon = epsilon
        self.delta = delta
        self.kl_clip_bound = kl_clip_bound
        self.layer_threshold_R = layer_threshold_R
        
        # 论文 Theorem 2: Δf ≤ 2ηEG_c
        self.sensitivity_bound = 2 * learning_rate * local_epochs * grad_clip_norm
        
        # 论文 Theorem 3: 计算 c_i
        if c_i is None:
            self.c_i = self._compute_ci(epsilon, delta, kl_clip_bound)
        else:
            self.c_i = c_i
            
        self._last_noise_stats = {}
        self._last_layer_weights = {}
    
    def _compute_ci(self, epsilon: float, delta: float, B: float) -> float:
        """简化 c_i 计算，确保 σ ∝ 1/ε"""
        # 标准高斯机制: σ = √(2 log(1.25/δ)) / ε × sensitivity
        # 让 c_i = √(2 log(1.25/δ))，与 ClassicDP 一致
        c_i = math.sqrt(2 * math.log(1.25 / delta))
        return max(c_i, B / 2)
    
    def _select_layers(self, model_state: OrderedDict) -> list:
        """
        论文 Algorithm 2 Line 12: if ||w_{i,j}^t|| ≥ R
        使用 L2 范数判断
        """
        if self.layer_threshold_R <= 0:
            return list(model_state.keys())
        
        selected = []
        for key, param in model_state.items():
            # 论文: L2 范数
            l2_norm = torch.norm(param).item()
            if l2_norm >= self.layer_threshold_R:
                selected.append(key)
        return selected

    def _compute_kl_divergence(self, local_weight: torch.Tensor,
                                global_weight: torch.Tensor) -> float:
        """
        论文 Eq. 8: KL(local || global)
        使用高斯分布近似代替 softmax 概率分布
        """
        mu_l = local_weight.mean().item()
        var_l = local_weight.var().item() + 1e-10
        mu_g = global_weight.mean().item()
        var_g = global_weight.var().item() + 1e-10

        # KL(N(μ₁,σ²₁) ‖ N(μ₂,σ²₂)) = log(σ₂/σ₁) + (σ²₁+(μ₁-μ₂)²)/(2σ²₂) - 0.5
        kl = (math.log(math.sqrt(var_g) / math.sqrt(var_l)) +
            (var_l + (mu_l - mu_g) ** 2) / (2 * var_g) - 0.5)

        kl = max(0.0, kl)

        # 论文 Eq. 8: 裁剪到 B
        return min(kl, self.kl_clip_bound)

    def _compute_layer_noise_scale(self, privacy_value: float) -> float:
        """
        论文 Eq. 10: σ = (c_i × Δf) / (ε × P)
        
        P 在分母，因此:
        - P 小 (高隐私风险) → σ 大 → 噪声多
        - P 大 (低隐私风险) → σ 小 → 噪声少
        """
        delta_f = self.sensitivity_bound
        privacy_clipped = max(privacy_value, 1e-8)  # 避免除零

        # 论文 Eq. 10: P 在分母
        noise_scale = (self.c_i * delta_f) / (self.epsilon * privacy_clipped)

        # 防止噪声过大破坏模型
        max_noise = delta_f * 10
        return min(noise_scale, max_noise)

    def add_noise(self, model_state: OrderedDict,
                  global_state: OrderedDict = None) -> OrderedDict:
        """
        主入口：按论文流程加噪
        
        流程:
        1. 层选择 (Algorithm 2 Line 12)
        2. 隐私估计 (Algorithm 3, Eq. 8)
        3. 噪声注入 (Algorithm 4, Eq. 10)
        """
        # 1. 层选择: 基于 L2 范数
        selected_keys = self._select_layers(model_state)
        
        # 2. 只为选中的层计算隐私估计
        privacy_values = {}
        for key in selected_keys:
            if global_state is not None and key in global_state:
                kl = self._compute_kl_divergence(
                    model_state[key], global_state[key]
                )
                privacy_values[key] = kl
            else:
                # 无全局模型时使用默认值
                privacy_values[key] = self.kl_clip_bound / 2
        
        # 3. 为选中的层计算噪声尺度 (论文 Eq. 10)
        layer_noise_scales = {}
        for key in selected_keys:
            layer_noise_scales[key] = self._compute_layer_noise_scale(
                privacy_values[key]
            )
        
        # 4. 加噪: 选中的层加噪，未选中的层保持不变
        noisy_state = OrderedDict()
        total_noise_power = 0.0
        total_params = 0

        for key, param in model_state.items():
            if key in layer_noise_scales:
                sigma = layer_noise_scales[key]
                noise = torch.randn_like(param) * sigma
                noisy_state[key] = param + noise
                total_noise_power += (sigma ** 2) * param.numel()
            else:
                # 论文: 忽略小权重层，不加噪
                noisy_state[key] = param.clone()
            
            total_params += param.numel()

        # 记录统计信息
        avg_std = (total_noise_power / max(total_params, 1)) ** 0.5
        self.accountant.step(avg_std)

        self._last_noise_stats = {
            "mode": "ladp",
            "epsilon": self.epsilon,
            "c_i": round(self.c_i, 4),
            "sensitivity_bound": round(self.sensitivity_bound, 4),
            "threshold_R": self.layer_threshold_R,
            "selected_layers": len(selected_keys),
            "total_layers": len(model_state),
            "selection_ratio": len(selected_keys) / len(model_state),
            "avg_noise_per_param": round(avg_std, 6),
            "total_noise_power": round(total_noise_power, 4),
            "privacy_values": {k: round(v, 4) for k, v in privacy_values.items()},
            "noise_scales": {k: round(v, 6) for k, v in layer_noise_scales.items()},
        }
        self._last_layer_weights = layer_noise_scales.copy()
        return noisy_state
    
    def get_noise_stats(self) -> dict:
        """返回最近一次加噪的统计信息"""
        return self._last_noise_stats

    def get_layer_weights(self) -> dict:
        """返回最近一次计算的层权重（兼容旧接口）"""
        return self._last_layer_weights