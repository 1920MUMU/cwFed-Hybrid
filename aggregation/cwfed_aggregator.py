"""
Class-Wise 聚合器（支持每类独立滑动窗口）
"""
from collections import OrderedDict
import torch
import numpy as np


class CWFedAggregator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.expert_models = None

    def aggregate(self, contributions, model_states,
                  expert_models_old=None,
                  mix_contributions=None,
                  enable_window=False,
                  clients=None):
        N = len(model_states)
        K = self.num_classes
        if N == 0:
            return []

        first_state = model_states[0]
        param_keys = list(first_state.keys())

        # 滑动窗口
        if enable_window and clients is not None:
            model_states = self._apply_per_class_sliding_window(
                contributions, model_states, clients
            )

        # 阶段1: 构建专家模型
        new_expert_models = {}
        for c in range(K):
            class_contribs = [contributions[i][c] for i in range(N)]
            total_c = sum(class_contribs)
            if total_c < 1e-8:
                weights = [1.0/N] * N
            else:
                weights = [v/total_c for v in class_contribs]

            new_expert = OrderedDict()
            for key in param_keys:
                weighted_sum = torch.zeros_like(first_state[key], dtype=torch.float32)
                for i, ms in enumerate(model_states):
                    weighted_sum += ms[key].float() * weights[i]
                new_expert[key] = weighted_sum
            new_expert_models[c] = new_expert

        self.expert_models = new_expert_models

        # 阶段2: 组合个性化模型
        personalized_models = []
        mix_weights = []

        for i in range(N):
            client_mix = mix_contributions[i] if mix_contributions else contributions[i]
            total_i = sum(client_mix)
            if total_i < 1e-8:
                pi = [1.0/K] * K
            else:
                pi = [client_mix[c]/total_i for c in range(K)]

            mix_weights.append(pi)

            personalized_state = OrderedDict()
            for key in param_keys:
                combined = torch.zeros_like(first_state[key], dtype=torch.float32)
                for c in range(K):
                    if pi[c] > 0:
                        combined += new_expert_models[c][key] * pi[c]
                personalized_state[key] = combined
            personalized_models.append(personalized_state)

        self.personalization_weights = mix_weights
        return personalized_models

    def get_expert_models(self):
        return self.expert_models

    def get_personalization_weights(self):
        return self.personalization_weights
    
    def _apply_per_class_sliding_window(self, contributions, model_states, clients):
        K = self.num_classes
        N = len(model_states)

        targets = np.zeros((N, K))
        for c in range(K):
            col = [contributions[i][c] for i in range(N)]
            min_v = min(col) if col and min(col) > 0 else 1.0
            for i in range(N):
                targets[i][c] = col[i] / max(min_v, 1e-8)

        for i, client in enumerate(clients):
            target = float(targets[i].max())
            smoothed = client.get_smoothed_model(target)
            model_states[i] = smoothed

        return model_states