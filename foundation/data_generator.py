"""
合成数据生成器
生成二维高斯分布数据，每个类是一个高斯簇。
用于快速验证联邦学习流程。
"""
import numpy as np


class SyntheticDataGenerator:
    """
    生成简单的二维特征数据用于快速实验。
    每个类别从不同均值的高斯分布采样。
    
    类中心均匀分布在圆周上，各类标准差相同。
    """

    def __init__(self, num_classes: int = 10, input_dim: int = 2,
                 cluster_radius: float = 3.0, cluster_std: float = 0.5,
                 seed: int = 42):
        """
        参数:
            num_classes: 类别数量
            input_dim: 特征维度（默认2，便于可视化）
            cluster_radius: 类中心到原点的距离
            cluster_std: 每个簇的标准差
            seed: 随机种子（仅影响此生成器，不影响全局）
        """
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.cluster_radius = cluster_radius
        self.cluster_std = cluster_std
        self.seed = seed

        # 使用独立的随机状态，不影响全局
        self.rng = np.random.RandomState(seed)

        # 在圆周上均匀分布类中心
        angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        if input_dim == 2:
            self.centers = np.column_stack([
                cluster_radius * np.cos(angles),
                cluster_radius * np.sin(angles)
            ])
        else:
            # 高维：前两维在圆周上，其余维度为0
            self.centers = np.zeros((num_classes, input_dim))
            self.centers[:, 0] = cluster_radius * np.cos(angles)
            self.centers[:, 1] = cluster_radius * np.sin(angles)

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        生成数据，各类别样本数尽量均匀。
        
        输入:
            n_samples: 总样本数
        输出:
            X: np.ndarray, shape (n_samples, input_dim), float32
            y: np.ndarray, shape (n_samples,), int64
        """
        if n_samples < self.num_classes:
            raise ValueError(
                f"样本数({n_samples})不能小于类别数({self.num_classes})"
            )

        # 每类基础样本数 + 余数分配
        per_class = n_samples // self.num_classes
        remainder = n_samples % self.num_classes

        X_list, y_list = [], []

        for c in range(self.num_classes):
            n_c = per_class + (1 if c < remainder else 0)
            if n_c == 0:
                continue

            center = self.centers[c]
            X_c = center + self.rng.randn(n_c, self.input_dim) * self.cluster_std
            y_c = np.full(n_c, c, dtype=np.int64)

            X_list.append(X_c)
            y_list.append(y_c)

        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list)

        # 打乱
        idx = self.rng.permutation(len(y))
        return X[idx], y[idx]

    def generate_global_test(self, n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """
        生成全局测试集。
        
        输入:
            n_samples: 测试集总样本数
        输出:
            X: np.ndarray, shape (n_samples, input_dim)
            y: np.ndarray, shape (n_samples,)
        """
        return self.generate(n_samples)

    def generate_separate_train_test(self, n_train: int, n_test: int) -> tuple:
        """
        生成独立的训练集和测试集（测试集用不同种子保证独立性）。
        
        输入:
            n_train: 训练集样本数
            n_test: 测试集样本数
        输出:
            (X_train, y_train, X_test, y_test)
        """
        X_train, y_train = self.generate(n_train)

        # 用不同种子生成测试集，保证独立性
        test_gen = SyntheticDataGenerator(
            num_classes=self.num_classes,
            input_dim=self.input_dim,
            cluster_radius=self.cluster_radius,
            cluster_std=self.cluster_std,
            seed=self.seed + 99999
        )
        X_test, y_test = test_gen.generate(n_test)

        return X_train, y_train, X_test, y_test