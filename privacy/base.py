"""
差分隐私模块基类
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch


class PrivacyModule(ABC):
    """所有 DP 模块的基类"""
    
    def __init__(self, accountant):
        self.accountant = accountant
    
    @abstractmethod
    def add_noise(self, model_state: OrderedDict, 
                  global_state: OrderedDict = None) -> OrderedDict:
        """
        向模型参数添加噪声。
        
        参数:
            model_state: 本地模型参数
            global_state: 全局/基线模型参数（LaDP 需要用于计算KL散度）
        返回:
            加噪后的模型参数
        """
        pass
    
    @abstractmethod
    def get_noise_stats(self) -> dict:
        """返回本轮噪声统计信息"""
        pass