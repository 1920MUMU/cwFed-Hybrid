"""
经典差分隐私：由 ε 推导噪声强度
"""
from collections import OrderedDict
import torch
import math
from .base import PrivacyModule


class ClassicDP(PrivacyModule):
    """均匀加噪的经典 DP 模块"""
    
    def __init__(self, accountant, epsilon: float = 8.0, delta: float = 1e-5,
                 clip_norm: float = 1.0, learning_rate: float = 0.1,
                 local_epochs: int = 1):
        super().__init__(accountant)
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

        # 灵敏度 = 2 × η × E × C（与 LaDP 的 Theorem 2 一致）
        sensitivity = 2 * learning_rate * local_epochs * clip_norm

        # σ = sensitivity × √(2 log(1.25/δ)) / ε
        self.noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    
    def add_noise(self, model_state: OrderedDict,
                  global_state: OrderedDict = None) -> OrderedDict:
        noisy_state = OrderedDict()
        total_noise_power = 0.0
        total_params = 0
        
        for key, param in model_state.items():
            noise = torch.randn_like(param) * self.noise_scale
            noisy_state[key] = param + noise
            total_noise_power += noise.norm().item() ** 2
            total_params += param.numel()
        
        avg_std = (total_noise_power / max(total_params, 1)) ** 0.5
        self.accountant.step(avg_std)
        
        self._last_noise_stats = {
            "mode": "classic",
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_scale": round(self.noise_scale, 6),
            "total_noise_power": round(total_noise_power, 4),
            "avg_noise_per_param": round(avg_std, 6),
        }
        
        return noisy_state
    
    def get_noise_stats(self) -> dict:
        return self._last_noise_stats