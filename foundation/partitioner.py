"""
数据划分器
将全局数据按多种分布策略分配给各客户端。
支持：uniform / heterogeneous / noisy
"""
import numpy as np
from config import ExperimentConfig


class DataPartitioner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.num_clients = config.num_clients
        self.num_classes = config.num_classes
        self.distribution = config.distribution
        self.noise_ratio = config.noise_ratio
        self.dirichlet_alpha = config.dirichlet_alpha
        self.seed = config.seed

        # 每客户端测试样本数
        self.test_samples_per_client = getattr(config, 'test_samples_per_client', None)
        if self.test_samples_per_client is None:
            self.test_samples_per_client = config.num_classes * 20

        # 噪声参数
        self.noise_heterogeneous = getattr(config, 'noise_heterogeneous', False)
        self.noise_intra_class = getattr(config, 'noise_intra_class', False)
        self.noise_hetero_alpha = getattr(config, 'noise_hetero_alpha', 0.5)
        self.noise_info = {}

        self.rng = np.random.RandomState(config.seed)

    def partition(self, x_global, y_global, x_test_global=None, y_test_global=None):
        # 1. 划分训练数据
        if self.distribution == "uniform":
            client_data = self._partition_uniform(x_global, y_global)
        elif self.distribution in ["heterogeneous", "noisy"]:
            client_data = self._partition_heterogeneous(x_global, y_global)
        else:
            raise ValueError(f"未知分布策略: {self.distribution}")

        # 2. 从独立测试集抽取（必须有测试集）
        if x_test_global is None or y_test_global is None:
            raise ValueError("必须提供独立测试集")
        result = self._split_train_test_from_global(
            client_data, x_test_global, y_test_global
        )

        # 3. 噪声
        if self.distribution == "noisy":
            result = self._add_label_noise_to_train(result)

        return result

    def _partition_uniform(self, x, y):
        """均匀分布"""
        class_indices = [np.where(y == c)[0] for c in range(self.num_classes)]
        client_data = {i: {"x": [], "y": []} for i in range(self.num_clients)}

        for c in range(self.num_classes):
            indices_c = class_indices[c].copy()
            self.rng.shuffle(indices_c)
            per_client = len(indices_c) // self.num_clients
            for i in range(self.num_clients):
                start = i * per_client
                end = start + per_client if i < self.num_clients - 1 else len(indices_c)
                selected = indices_c[start:end]
                client_data[i]["x"].append(x[selected])
                client_data[i]["y"].append(y[selected])

        for i in range(self.num_clients):
            client_data[i]["x"] = np.concatenate(client_data[i]["x"])
            client_data[i]["y"] = np.concatenate(client_data[i]["y"])

        return client_data

    def _partition_heterogeneous(self, x, y):
        """异构分布"""
        class_indices = [np.where(y == c)[0] for c in range(self.num_classes)]
        client_data = {i: {"x": [], "y": []} for i in range(self.num_clients)}

        for c in range(self.num_classes):
            indices_c = class_indices[c].copy()
            self.rng.shuffle(indices_c)
            n_c = len(indices_c)

            proportions = self.rng.dirichlet([self.dirichlet_alpha] * self.num_clients)
            counts = (proportions * n_c).astype(int)
            diff = n_c - counts.sum()
            counts[0] += diff

            start = 0
            for i in range(self.num_clients):
                cnt = counts[i]
                if cnt > 0:
                    selected = indices_c[start:start + cnt]
                    client_data[i]["x"].append(x[selected])
                    client_data[i]["y"].append(y[selected])
                    start += cnt

        for i in range(self.num_clients):
            if len(client_data[i]["x"]) == 0:
                client_data[i]["x"] = np.array([]).reshape(0, *x.shape[1:])
                client_data[i]["y"] = np.array([], dtype=np.int64)
            else:
                client_data[i]["x"] = np.concatenate(client_data[i]["x"])
                client_data[i]["y"] = np.concatenate(client_data[i]["y"])

        return client_data

    def get_client_noise_info(self) -> dict:
        return self.noise_info

    # ===== 噪声方法（不变）=====
    def _add_label_noise_to_train(self, client_data: dict) -> dict:
        n_clients = len(client_data)
        num_classes = self.num_classes
        max_class_noise = 0.5

        configs = self._generate_noise_configs(n_clients, num_classes)
        configs = self._apply_class_noise_limit(
            client_data, configs, num_classes, max_class_noise
        )
        self._apply_noise_to_clients(client_data, configs, num_classes)
        return client_data

    def _generate_noise_configs(self, n_clients, num_classes):
        configs = []
        if self.noise_intra_class:
            for i in range(n_clients):
                base_noise = self.rng.uniform(
                    self.noise_ratio * 0.1, self.noise_ratio * 3.0
                ) if self.noise_heterogeneous else self.noise_ratio
                raw = self.rng.dirichlet([self.noise_hetero_alpha] * num_classes)
                class_noises = np.clip(raw * base_noise * num_classes, 0.01, 0.5)
                configs.append({c: float(class_noises[c]) for c in range(num_classes)})
        elif self.noise_heterogeneous:
            raw = self.rng.dirichlet([self.dirichlet_alpha] * n_clients)
            client_noises = np.clip(
                raw * self.noise_ratio * n_clients * 2,
                self.noise_ratio * 0.1, self.noise_ratio * 3.0
            )
            configs = [{c: float(client_noises[i]) for c in range(num_classes)} for i in range(n_clients)]
        else:
            configs = [{c: self.noise_ratio for c in range(num_classes)} for _ in range(n_clients)]
        return configs

    def _apply_class_noise_limit(self, client_data, configs, num_classes, max_noise):
        class_total = {c: 0 for c in range(num_classes)}
        class_per_client = {c: {} for c in range(num_classes)}
        for c in range(num_classes):
            for i in range(len(client_data)):
                count = int((client_data[i]["y_train"] == c).sum())
                class_per_client[c][i] = count
                class_total[c] += count
        for c in range(num_classes):
            if class_total[c] == 0:
                continue
            expected_noisy = sum(
                class_per_client[c][i] * configs[i][c]
                for i in range(len(client_data))
            )
            global_ratio = expected_noisy / class_total[c]
            if global_ratio > max_noise:
                scale = max_noise / global_ratio
                for i in range(len(client_data)):
                    configs[i][c] *= scale
        return configs

    def _apply_noise_to_clients(self, client_data, configs, num_classes):
        self.noise_info = {}
        for i in range(len(client_data)):
            y_train = client_data[i]["y_train"]
            n_total = len(y_train)
            if n_total == 0:
                self.noise_info[i] = {"noise_config": {}, "noisy_samples": 0, "total_samples": 0}
                continue
            total_noisy = 0
            for c in range(num_classes):
                class_indices = np.where(y_train == c)[0]
                n_class = len(class_indices)
                if n_class == 0:
                    continue
                n_noisy = max(1, int(n_class * configs[i][c]))
                if n_noisy >= n_class:
                    n_noisy = n_class - 1
                if n_noisy <= 0:
                    continue
                noisy_idx = self.rng.choice(class_indices, size=n_noisy, replace=False)
                for idx in noisy_idx:
                    original = y_train[idx]
                    other_classes = [cl for cl in range(num_classes) if cl != original]
                    y_train[idx] = self.rng.choice(other_classes)
                total_noisy += n_noisy
            self.noise_info[i] = {
                "noise_config": configs[i],
                "noisy_samples": total_noisy,
                "total_samples": n_total,
            }
        
    def _split_train_test_from_global(self, client_data: dict,
                                    x_test_global: np.ndarray,
                                    y_test_global: np.ndarray) -> dict:
        """
        从官方测试集中为每个客户端抽取测试样本，
        各类比例参照该客户端训练集的类别分布，
        每客户端测试样本数 = test_samples_per_client。
        """
        result = {}

        for i in range(self.num_clients):
            x_i = client_data[i]["x"]
            y_i = client_data[i]["y"]
            n_train = len(y_i)

            if n_train == 0:
                result[i] = {
                    "x_train": x_i,
                    "y_train": y_i.copy(),
                    "x_test": np.array([]).reshape(0, *x_i.shape[1:]),
                    "y_test": np.array([], dtype=np.int64),
                }
                continue

            # 计算该客户端训练集的类别分布
            class_counts_train = {}
            for c in range(self.num_classes):
                class_counts_train[c] = int((y_i == c).sum())

            total_train = sum(class_counts_train.values())
            if total_train == 0:
                result[i] = {
                    "x_train": x_i,
                    "y_train": y_i.copy(),
                    "x_test": np.array([]).reshape(0, *x_i.shape[1:]),
                    "y_test": np.array([], dtype=np.int64),
                }
                continue

            # 按训练集比例从官方测试集中抽取
            n_test_target = self.test_samples_per_client
            test_indices = []

            for c in range(self.num_classes):
                proportion = class_counts_train[c] / total_train
                n_c_test = max(1, int(n_test_target * proportion))

                # 从官方测试集中取类别 c 的样本
                c_test_indices = np.where(y_test_global == c)[0]
                if len(c_test_indices) > 0:
                    self.rng.shuffle(c_test_indices)
                    taken = c_test_indices[:n_c_test]
                    test_indices.extend(taken.tolist())

            # 修正总数
            test_indices = test_indices[:n_test_target]
            while len(test_indices) < n_test_target:
                extra = self.rng.choice(len(y_test_global), 1)[0]
                test_indices.append(extra)

            test_indices = np.array(test_indices, dtype=int)

            result[i] = {
                "x_train": x_i,
                "y_train": y_i.copy(),
                "x_test": x_test_global[test_indices],
                "y_test": y_test_global[test_indices].copy(),
            }

        return result