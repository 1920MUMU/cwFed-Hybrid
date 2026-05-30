import numpy as np


class PrivacyBudgetTracker:
    def __init__(self, epsilon: float = 8.0, delta: float = 1e-5,
                 noise_multiplier: float = 1.0):
        self.epsilon_target = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        
        # 高斯机制常数
        self.gaussian_constant = np.sqrt(2 * np.log(1.25 / self.delta))
        
        self.epsilon_spent = 0.0
        self.steps = 0
        self.total_noise_injected = 0.0

    def compute_noise_std(self, sensitivity: float = 1.0) -> float:
        """
        根据 (ε, δ)-DP 计算所需噪声标准差。
        
        高斯机制: σ = (√(2×ln(1.25/δ)) × Δf) / ε
        
        输入:
            sensitivity: 当前步的敏感度 Δf
        输出:
            float: 噪声标准差 σ
        """
        if self.epsilon_target <= 0:
            return 1e10  # 无隐私预算，极大噪声
        
        sigma = (self.gaussian_constant * sensitivity) / self.epsilon_target
        sigma *= self.noise_multiplier  # 用户可调的乘数
        
        return max(sigma, 1e-8)

    def step(self, sensitivity: float = 1.0) -> dict:
        """
        记录一步隐私消耗。
        
        简化 RDP 计算: ε_step ≈ sensitivity / σ
        """
        self.steps += 1
        sigma = self.compute_noise_std(sensitivity)
        
        # 每步消耗的 ε
        eps_step = sensitivity / sigma
        
        self.epsilon_spent += eps_step
        self.total_noise_injected += sigma

        return {
            "epsilon_step": eps_step,
            "epsilon_spent": self.epsilon_spent,
            "sigma": sigma,
            "steps": self.steps,
            "delta": self.delta,
        }

    def is_depleted(self) -> bool:
        return self.epsilon_spent >= self.epsilon_target

    def remaining_ratio(self) -> float:
        return max(0.0, 1.0 - self.epsilon_spent / self.epsilon_target)

    def get_stats(self) -> dict:
        return {
            "epsilon_target": self.epsilon_target,
            "delta": self.delta,
            "epsilon_spent": self.epsilon_spent,
            "remaining_ratio": self.remaining_ratio(),
            "steps": self.steps,
            "total_noise_injected": self.total_noise_injected,
            "current_sigma": self.compute_noise_std(1.0),
        }