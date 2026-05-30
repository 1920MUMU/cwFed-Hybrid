"""
轻量级模型定义
合成数据使用 MLP，CIFAR-10 使用小型 CNN。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.random_utils import set_seed

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, num_classes=10, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_classification_head(self) -> nn.Module:
        """返回分类层（最后一层）"""
        return self.fc3


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def get_classification_head(self) -> nn.Module:
        """返回分类层"""
        return self.fc2


def create_model(config) -> nn.Module:
    """
    工厂函数：根据配置创建模型。
    
    输入:
        config: ExperimentConfig 对象
    输出:
        nn.Module 实例
    """
    # 设置随机种子，确保模型初始化可复现
    set_seed(config.seed)
    
    if config.dataset == "synthetic":
        return SimpleMLP(input_dim=2, num_classes=config.num_classes)
    elif config.dataset == "cifar10":
        return SimpleCNN(num_classes=config.num_classes)
    else:
        raise ValueError(f"未知数据集: {config.dataset}")