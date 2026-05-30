"""
CIFAR-10 数据加载器
假设数据已存放在 data/cifar-10-batches-py/ 目录下。
"""
import numpy as np
import pickle
import os

class CIFAR10Loader:
    """
    从本地文件加载 CIFAR-10 数据集。
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = os.path.join(data_dir, "cifar-10-batches-py")
        self._validate()

    def _validate(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"CIFAR-10 数据目录未找到: {self.data_dir}\n"
            )

    def _unpickle(self, filepath: str) -> dict:
        """读取单个 batch 文件"""
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        return data_dict


    def load_raw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载完整 CIFAR-10 数据集。
        
        输入: 无
        输出:
            x_train: np.ndarray, shape (50000, 3, 32, 32), float32 归一化到 [0,1]
            y_train: np.ndarray, shape (50000,), int64
            x_test:  np.ndarray, shape (10000, 3, 32, 32), float32 归一化到 [0,1]
            y_test:  np.ndarray, shape (10000,), int64
        """
        # 加载训练数据 (5个batch)
        x_train_list, y_train_list = [], []
        for i in range(1, 6):
            batch = self._unpickle(os.path.join(self.data_dir, f"data_batch_{i}"))
            x_train_list.append(batch[b'data'])
            y_train_list.extend(batch[b'labels'])

        x_train = np.vstack(x_train_list).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_train = np.array(y_train_list, dtype=np.int64)

        # 加载测试数据
        test_batch = self._unpickle(os.path.join(self.data_dir, "test_batch"))
        x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y_test = np.array(test_batch[b'labels'], dtype=np.int64)

        return x_train, y_train, x_test, y_test