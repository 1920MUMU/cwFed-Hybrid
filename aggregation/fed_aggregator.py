"""
标准 Fed 聚合器（支持滑动窗口）
"""
from collections import OrderedDict
import torch
import math


class FedAggregator:
    def __init__(self):
        pass

    def aggregate(self, contributions: list, model_states: list,
                  enable_window: bool = False, clients: list = None) -> list:
        """
        参数:
            contributions: List[float], 标量权重（shapley 相似度或数据量比例）
            model_states: List[OrderedDict]
            enable_window: 是否启用滑动窗口
            clients: 客户端列表（滑动窗口需要）
        返回:
            List[OrderedDict], 长度为1
        """
        n = len(model_states)
        if n == 0:
            return []

        # 滑动窗口：替换模型为加权窗口平均
        if enable_window and clients is not None:
            model_states = self._apply_sliding_window(
                contributions, model_states, clients
            )

        first_state = model_states[0]
        avg_state = OrderedDict()
        for key in first_state.keys():
            weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
            for ms, w in zip(model_states, contributions):
                weighted_sum += ms[key].float() * w
            avg_state[key] = weighted_sum

        return [avg_state]

    def _apply_sliding_window(self, contributions, model_states, clients):
        min_w = min(contributions) if contributions else 1.0
        for i, client in enumerate(clients):
            target = contributions[i] / max(min_w, 1e-8)
            model_states[i] = client.get_smoothed_model(target)
        return model_states