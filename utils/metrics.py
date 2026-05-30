"""
评估指标计算模块
"""
import numpy as np
from sklearn.metrics import f1_score


class Metrics:
    """联邦学习评估指标计算器"""

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算准确率"""
        if len(y_true) == 0:
            return 0.0
        return (y_true == y_pred).mean()

    @staticmethod
    def global_accuracy(all_y_true: list, all_y_pred: list) -> float:
        """
        全局准确率：将所有客户端的预测结果合并计算。
        
        输入:
            all_y_true: list[np.ndarray] 每个客户端的真实标签
            all_y_pred: list[np.ndarray] 每个客户端的预测标签
        输出:
            float: 全局准确率 = 总正确数 / 总样本数
        """
        total_correct = 0
        total_samples = 0
        for y_true, y_pred in zip(all_y_true, all_y_pred):
            total_correct += (y_true == y_pred).sum()
            total_samples += len(y_true)
        return total_correct / total_samples if total_samples > 0 else 0.0

    @staticmethod
    def global_f1_macro(all_y_true: list, all_y_pred: list) -> float:
        """
        全局宏平均 F1：将所有客户端的预测结果合并后计算。
        
        输入:
            all_y_true: list[np.ndarray]
            all_y_pred: list[np.ndarray]
        输出:
            float: 全局宏平均 F1
        """
        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)
        
        if len(y_true_all) == 0:
            return 0.0
        try:
            return f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
        except Exception:
            return 0.0

    @staticmethod
    def f1_score_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """单客户端 F1（保留兼容）"""
        if len(y_true) == 0:
            return 0.0
        try:
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        except Exception:
            return 0.0

    @staticmethod
    def personalization_gain(personalized_accs: list, global_accs: list) -> float:
        """
        个性化增益（客户端平均版本，保留兼容）。
        """
        if len(personalized_accs) != len(global_accs) or len(personalized_accs) == 0:
            return 0.0
        gains = [p - g for p, g in zip(personalized_accs, global_accs)]
        return np.mean(gains)

    @staticmethod
    def global_personalization_gain(pers_correct: int, pers_total: int,
                                     base_correct: int, base_total: int) -> float:
        """
        全局个性化增益：基于总正确数和总样本数。
        
        输入:
            pers_correct: 个性化模型的总正确预测数
            pers_total: 个性化模型的总样本数
            base_correct: 基线模型的总正确预测数
            base_total: 基线模型的总样本数
        输出:
            float: 全局准确率差值
        """
        if pers_total == 0 or base_total == 0:
            return 0.0
        pers_acc = pers_correct / pers_total
        base_acc = base_correct / base_total
        return pers_acc - base_acc

    @staticmethod
    def fairness_cv(client_accs: list) -> float:
        """公平性指标（变异系数，保留客户端维度）"""
        if len(client_accs) <= 1:
            return 0.0
        mean_acc = np.mean(client_accs)
        if mean_acc < 1e-8:
            return 0.0
        return np.std(client_accs) / mean_acc