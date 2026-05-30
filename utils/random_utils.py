"""
统一随机种子设置工具
确保所有随机库使用相同种子，保证实验可复现
"""
import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """
    设置所有随机库的种子。
    
    参数:
        seed: 随机种子值
    """
    # Python 内置 random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)          # GPU 种子（如果有）
    torch.cuda.manual_seed_all(seed)      # 多 GPU 种子
    
    # PyTorch 确定性设置（可能略微降低性能，但保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ 全局随机种子已设置: {seed}")