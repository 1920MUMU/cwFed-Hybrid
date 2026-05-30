"""
联邦学习客户端
负责本地训练、贡献值计算、模型更新（含个性化混合）
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import OrderedDict
from federated.contribution import ClientContribution
from utils.metrics import Metrics
import math

class Client:
    def __init__(self, client_id: int, data: dict, model: nn.Module, config):
        self.client_id = client_id
        self.config = config
        self.device = torch.device(config.device)
        self.seed = config.seed
        self.mix_gamma = None

        torch.manual_seed(config.seed + client_id)
        self.model = self._clone_model(model)

        self.x_train = torch.from_numpy(data["x_train"]).float()
        self.y_train = torch.from_numpy(data["y_train"]).long()
        self.x_test = torch.from_numpy(data["x_test"]).float()
        self.y_test = torch.from_numpy(data["y_test"]).long()

        self.train_loader = self._create_loader(self.x_train, self.y_train, shuffle=True)
        self.test_loader = self._create_loader(self.x_test, self.y_test, shuffle=False)

        self.class_counts = self._count_classes(data["y_train"])
        self.total_samples = len(data["y_train"])

        self.model_history = []           # 历史模型队列
        self.max_window = 1               # 最大窗口长度

        self.current_round = 0 

        # DP 模块
        self.privacy_module = None
        if getattr(config, 'dp_enabled', False):
            self._init_privacy_module(config)

        self.wdr_module = None
        if getattr(config, 'wdr_enabled', False):
            from regularization.wdr import WDRLoss
            self.wdr_module = WDRLoss(
                lambda_reg=getattr(config, 'wdr_lambda', 1.0),
            )

    def _init_privacy_module(self, config):
        from privacy.accountant import PrivacyAccountant

        accountant = PrivacyAccountant()
        dp_mode = getattr(config, 'dp_mode', 'classic')
        epsilon = getattr(config, 'dp_epsilon', 8.0)
        delta = getattr(config, 'dp_delta', 1e-5)
        clip_norm = getattr(config, 'dp_grad_clip_norm', 1.0)

        if dp_mode == 'classic':
            from privacy.classic_dp import ClassicDP  # ← 函数内导入
            self.privacy_module = ClassicDP(
                accountant=accountant,
                epsilon=epsilon,
                delta=delta,
                clip_norm=clip_norm,
                learning_rate=config.learning_rate,
                local_epochs=config.local_epochs,
            )
        elif dp_mode == 'ladp':
            from privacy.ladp import LaDP              # ← 函数内导入
            self.privacy_module = LaDP(
                accountant=accountant,
                epsilon=epsilon,
                delta=delta,
                learning_rate=config.learning_rate,
                local_epochs=config.local_epochs,
                grad_clip_norm=clip_norm,
            )

    def _clone_model(self, model: nn.Module) -> nn.Module:
        import copy
        return copy.deepcopy(model)

    def _create_loader(self, x: torch.Tensor, y: torch.Tensor, shuffle: bool):
        if len(y) == 0:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        dataset = TensorDataset(x, y)
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=shuffle,
            generator=generator if shuffle else None,
            worker_init_fn=lambda wid: np.random.seed(self.seed + wid) if shuffle else None
        )

    def _count_classes(self, y: np.ndarray) -> dict:
        if len(y) == 0:
            return {c: 0 for c in range(self.config.num_classes)}
        unique, counts = np.unique(y, return_counts=True)
        counts_dict = {c: 0 for c in range(self.config.num_classes)}
        for u, c in zip(unique, counts):
            counts_dict[int(u)] = int(c)
        return counts_dict
    
    def _get_class_distribution_tensor(self) -> torch.Tensor:
        """获取本地类别分布张量 [K]，和为 1"""
        dist = torch.zeros(self.config.num_classes)
        total = sum(self.class_counts.values())
        if total > 0:
            for c in range(self.config.num_classes):
                dist[c] = self.class_counts.get(c, 0) / total
        else:
            dist = torch.ones(self.config.num_classes) / self.config.num_classes
        return dist

    def train(self) -> ClientContribution:
        """训练 + 上传 noisy_k 和 noisy_model"""
        torch.manual_seed(self.seed + self.client_id * 1000 + self.current_round * 12345)

        if self.train_loader is None or self.total_samples == 0:
            return self._empty_contribution()

        self.model.train()
        self.model.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(self.config.local_epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                # ===== WDR 正则化 =====
                if self.wdr_module is not None:
                    class_dist = self._get_class_distribution_tensor().to(self.device)
                    wdr_loss = self.wdr_module.compute(
                        self.model.get_classification_head(), class_dist
                    )
                    loss = loss + wdr_loss
                loss.backward()

                # ===== 梯度裁剪 =====
                if self.privacy_module is not None:
                    clip_norm = getattr(self.config, 'dp_grad_clip_norm', 1.0)
                    if clip_norm > 0:
                        total_norm = sum(p.grad.norm()**2 for p in self.model.parameters())**0.5
                        if total_norm > clip_norm:
                            scale = clip_norm / total_norm
                            for p in self.model.parameters():
                                p.grad.data.mul_(scale)

                optimizer.step()

                # ===== 记录训练指标 =====
                total_loss += loss.item() * len(batch_y)
                pred = logits.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += len(batch_y)

        # 训练结束后，DP 加噪前：
        if self.wdr_module is not None:
            # 记录分类层每类权重的 L2 范数
            head = self.model.get_classification_head()
            weight_norms = torch.norm(head.weight.data, p=2, dim=1).cpu()  # [K]
            wdr_norms_dict = {c: weight_norms[c].item() for c in range(self.config.num_classes)}
        else:
            wdr_norms_dict = None

        # DP 加噪
        if self.privacy_module is not None:
            noisy_state = self.privacy_module.add_noise(self.model.state_dict())
            self.model.load_state_dict(noisy_state)

        self.model.cpu()

        avg_loss = total_loss / total if total > 0 else 0.0
        train_accuracy = correct / total if total > 0 else 0.0

        if getattr(self.config, 'sliding_window_enabled', False):
            self.push_model()

        # 直接上传模型参数（不再计算变化量）
        return ClientContribution(
            client_id=self.client_id,
            model_state=self.model.state_dict(),
            contribution_values=[float(self.total_samples)],
            class_contribution_values=[
                float(self.class_counts.get(c, 0))
                for c in range(self.config.num_classes)
            ],
            metadata={
                "total_samples": self.total_samples,
                "class_counts": self.class_counts.copy(),
                "train_acc": train_accuracy,
                "train_loss": avg_loss,
                "wdr_weight_norms": wdr_norms_dict,
            }
        )
    
    def compute_similarity(self, expert_models, baseline_model=None) -> dict:
        head = self.model.get_classification_head()
        K = head.weight.shape[0]

        # 一次提取本地分类层参数 [K, D] 和 bias [K]
        local_weight, local_bias = self._get_head_params(self.model.state_dict(), head)

        if baseline_model is not None:
            base_weight, base_bias = self._get_head_params(baseline_model, head)
            per_class = []
            for c in range(K):
                local_c = self._cat_class_vec(local_weight, local_bias, c)
                base_c = self._cat_class_vec(base_weight, base_bias, c)
                per_class.append(max(self._cos_sim(local_c, base_c), 1e-6))
            scalar = sum(per_class) / K
            return {"scalar": scalar, "per_class": per_class}
        else:
            per_class = []
            for c in range(K):
                expert_weight, expert_bias = self._get_head_params(expert_models[c], head)
                local_c = self._cat_class_vec(local_weight, local_bias, c)
                expert_c = self._cat_class_vec(expert_weight, expert_bias, c)
                per_class.append(max(self._cos_sim(local_c, expert_c), 1e-6))
            scalar = sum(per_class) / K
            return {"scalar": scalar, "per_class": per_class}


    def _get_head_params(self, state_dict, head):
        """一次遍历提取分类层 weight 和 bias"""
        weight, bias = None, None
        for key, val in state_dict.items():
            if hasattr(head, 'weight') and val.shape == head.weight.shape:
                weight = val.float()
            elif hasattr(head, 'bias') and head.bias is not None and val.shape == head.bias.shape:
                bias = val.float()
        return weight, bias


    def _cat_class_vec(self, weight, bias, c):
        """拼接类 c 的向量 [W[c], bias[c]]"""
        parts = [weight[c].flatten()]
        if bias is not None:
            parts.append(bias[c].unsqueeze(0))
        return torch.cat(parts)


    def _extract_class_vector(self, state_dict, head, c):
        """保留旧接口兼容"""
        w, b = self._get_head_params(state_dict, head)
        return self._cat_class_vec(w, b, c)

    def _cos_sim(self, a, b):
        a, b = a.float(), b.float()
        eps = 1e-8
        return (torch.dot(a, b) / (torch.norm(a)+eps) / (torch.norm(b)+eps)).item()

    def _empty_contribution(self) -> ClientContribution:
        """空数据客户端的贡献值"""
        return ClientContribution(
            client_id=self.client_id,
            model_state=self.model.state_dict(),
            contribution_values=[0.0],
            class_contribution_values=[0.0] * self.config.num_classes,
            metadata={"total_samples": 0}
        )
    
    def update_model(self, new_model_state: OrderedDict):
        """更新模型，启用个性化混合"""
        if self.config.weight_mode == "shapley" and self.mix_gamma is not None:
            gamma = self.mix_gamma
            mixed_state = OrderedDict()
            local_state = self.model.state_dict()

            if getattr(self.config, 'reverse_mix', False):
                # 反转：高贡献 → 信任本地
                server_weight = 1.0 - gamma
                local_weight = gamma
            else:
                # 默认：高贡献 → 信任全局
                server_weight = gamma
                local_weight = 1.0 - gamma

            server_weight=1
            local_weight =0
            for key in new_model_state.keys():
                mixed_state[key] = (
                    server_weight * new_model_state[key].float() +
                    local_weight * local_state[key].float()
                )
            self.model.load_state_dict(mixed_state)
        else:
            self.model.load_state_dict(new_model_state)
        
    def evaluate(self, model_state: OrderedDict = None) -> dict:
        """评估模型"""
        if model_state is not None:
            self.model.load_state_dict(model_state)

        if self.test_loader is None or len(self.y_test) == 0:
            return {
                "accuracy": 0.0, "f1_macro": 0.0,
                "y_true": np.array([]), "y_pred": np.array([]),
                "correct": 0, "total": 0,
            }

        self.model.eval()
        self.model.to(self.device)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred = self.model(batch_x).argmax(dim=1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        self.model.cpu()

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        return {
            "accuracy": Metrics.accuracy(y_true, y_pred),
            "f1_macro": Metrics.f1_score_macro(y_true, y_pred),
            "y_true": y_true,
            "y_pred": y_pred,
            "correct": int((y_true == y_pred).sum()),
            "total": len(y_true),
        }
    
    def _clone_state(self, state_dict):
        import copy
        return copy.deepcopy(state_dict)

    def push_model(self):
        """将当前模型加入历史队列"""
        import copy
        self.model_history.append(copy.deepcopy(self.model.state_dict()))
        # 限制队列长度
        if len(self.model_history) > self.max_window+1:
            self.model_history.pop(0)

    def get_smoothed_model(self, target_noise_ratio: float) -> OrderedDict:
        """
        返回窗口平滑后的模型，使等效噪声接近目标噪声倍数。
        
        参数:
            target_noise_ratio: 目标噪声倍数 = γ_i / γ_min（≥1）
        返回:
            OrderedDict, 平滑后的模型参数
        
        逻辑:
        1. 计算基础窗口长度 n = ceil(target_noise_ratio²)
        2. 窗口简单平均的等效噪声 = σ / √n
        3. 如果 √n > target_noise_ratio，调高当前轮权重以增加噪声
        目标: σ_actual ≈ σ / target_noise_ratio
        求解: 当前轮权重 β 使得 σ_actual = σ × β / √(n-1+β²) ≈ σ / target_noise_ratio
        """
        target = max(1.0, target_noise_ratio)
        
        # 基础窗口长度（向上取整）
        n = max(1, int(np.ceil(target ** 2)))
        n = min(n, len(self.model_history))
        
        if n <= 1:
            return self.model.state_dict()
        
        recent = self.model_history[-n:]
        current = self.model.state_dict()
        
        # 简单平均的等效噪声倍数
        simple_ratio = np.sqrt(n)
        
        if simple_ratio <= target + 0.01:
            # 简单平均已经满足或略超，直接用等权平均
            smoothed = OrderedDict()
            for key in recent[0].keys():
                stacked = torch.stack([ms[key].float() for ms in recent])
                smoothed[key] = stacked.mean(dim=0)
            return smoothed
        
        # 调高当前轮权重以增加噪声
        # 目标: √(1 + (n-1) × α²) / (1 + (n-1) × α) × σ / √n × adjustment = σ / target
        # 简化：给当前轮更高的权重 β，历史轮权重 α
        # β² + (n-1) × α² = (β + (n-1)α)² / target²
        # 约束: β + (n-1)α = 1
        
        # 数值求解 β
        beta = 1.0 / n  # 初始值（等权）
        for _ in range(20):
            alpha = (1.0 - beta) / (n - 1) if n > 1 else 0
            noise_var = beta ** 2 + (n - 1) * alpha ** 2
            actual_ratio = 1.0 / np.sqrt(noise_var)
            if abs(actual_ratio - target) < 0.001:
                break
            # 梯度调整
            beta += 0.01 * (target - actual_ratio)
            beta = max(0.1 / n, min(1.0, beta))
        
        alpha = (1.0 - beta) / (n - 1) if n > 1 else 0
        
        smoothed = OrderedDict()
        for key in current.keys():
            weighted_sum = beta * current[key].float()
            for ms in recent[1:]:  # 历史模型
                weighted_sum += alpha * ms[key].float()
            smoothed[key] = weighted_sum
        
        return smoothed
    