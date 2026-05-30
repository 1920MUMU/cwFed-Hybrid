"""
贡献值统一抽象
所有客户端上传的数据都用 Contribution 封装
服务器和聚合器不需要知道具体含义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import OrderedDict
import torch


@dataclass
class ClientContribution:
    """
    客户端上传的数据。
    
    字段:
        client_id: 客户端ID
        model_state: 本地训练后的模型参数（可能加噪）
        contribution_values: 占位值（保留兼容）
        class_contribution_values: 占位值（保留兼容）
        metadata: 额外信息
            - noisy_k_list: 每类加噪变化量（shapley 模式）
            - total_samples: 本地样本数
            - class_counts: 各类样本数
            - train_acc, train_loss: 训练指标
    """
    client_id: int
    model_state: OrderedDict
    contribution_values: List[float]
    class_contribution_values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    