"""
隐私预算追踪器
使用 Moments Accountant 近似累加隐私消耗
"""
import numpy as np
import math


class PrivacyAccountant:
    """Moments Accountant 风格的隐私追踪器"""
    
    def __init__(self):
        self.steps_taken = 0
        self.epsilon_spent = 0.0
        self.history = []
    
    def step(self, noise_multiplier: float):
        """记录一步"""
        if noise_multiplier < 1e-8:
            eps_step = float('inf')
        else:
            sigma = noise_multiplier
            alpha = 2
            eps_step = alpha / (2 * sigma * sigma)
        
        self.epsilon_spent += eps_step
        self.steps_taken += 1
        
        self.history.append({
            "step": self.steps_taken,
            "noise_multiplier": noise_multiplier,
            "epsilon_step": eps_step,
            "epsilon_spent": self.epsilon_spent,
        })
    
    def get_privacy_spent(self) -> dict:
        return {
            "epsilon_spent": self.epsilon_spent,
            "steps_taken": self.steps_taken,
        }
    
    def is_budget_exhausted(self) -> bool:
        return self.epsilon_spent >= self.epsilon_total
    
    def reset(self):
        self.steps_taken = 0
        self.epsilon_spent = 0.0
        self.history = []
    
    def get_history(self) -> list:
        return self.history