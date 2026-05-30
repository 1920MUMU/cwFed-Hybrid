"""
全局配置数据类。所有模块导入此文件获取参数。
"""
from dataclasses import dataclass, field
from typing import Literal, List

@dataclass
class ExperimentConfig:
    # 数据集配置
    dataset: Literal["synthetic", "cifar10"] = "cifar10"
    num_classes: int = 20
    samples_per_client: int = 200

    # 分布配置
    distribution: Literal["uniform", "heterogeneous", "noisy"] = "heterogeneous"
    noise_ratio: float = 0.1
    noise_heterogeneous: bool = True
    noise_intra_class: bool = True
    noise_hetero_alpha: float = 0.5
    dirichlet_alpha: float = 0.5          

    # 客户端配置
    num_clients: int = 20
    local_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.05

    # 全局训练配置
    global_rounds: int = 100
    test_samples_per_client: int = 1000
    early_stop_acc: float = 0.90

    # 策略配置
    weight_mode: Literal["avg", "shapley","hybrid"] = "avg"
    aggregator_mode: Literal["fed", "cwfed"] = "fed"

    # GUI 显示配置
    display_metrics: List[str] = field(default_factory=lambda: [
        "train_accuracy",
        "test_accuracy",
        "test_f1",
        "personalization_gain",
        "fairness_cv"
    ])

    # EMA 平滑配置（仅 shapley 模式生效）
    ema_enabled: bool = False          # 是否启用 EMA 平滑
    ema_base_alpha: float = 0.1        # 基础平滑系数（本地轮数=1时）
    ema_max_alpha: float = 0.8         # 最大平滑系数（本地轮数很大时）
    # alpha 计算公式: alpha = ema_base_alpha + (ema_max_alpha - ema_base_alpha) × (1 - e^{-local_epochs/τ})
    # τ = 5（衰减常数）

    # 差分隐私配置
    dp_enabled: bool = True
    dp_mode: str = "classic"          # "classic" 或 "ladp"
    dp_epsilon: float = 8.0           # 隐私预算 ε
    dp_delta: float = 1e-5            # 失败概率 δ
    dp_grad_clip_norm: float = 1.0

    # WDR 正则化
    wdr_enabled: bool = False
    wdr_lambda: float = 1.0

    reverse_mix: bool = False  # True: 高贡献信任本地, False: 高贡献信任全局

    sliding_window_enabled: bool = False  # 滑动窗口平滑

    eval_every: int = 3  # 每隔多少轮评估一次（1=每轮评估）

    # 设备
    device: str = "cpu"
    seed: int = 42