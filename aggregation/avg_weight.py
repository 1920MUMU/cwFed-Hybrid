"""
基于样本数的权重计算器
简单平均：权重 = 本地样本数 / 总样本数
"""

class AvgWeightCalculator:
    """基于数据量的权重计算器"""

    def compute(self, client_outputs: list) -> list:
        """
        输入: client_outputs, 每个含 "class_counts"
        输出: weights, 按数据量比例
        """
        total_samples = 0
        sample_counts = []

        for output in client_outputs:
            count = sum(output["class_counts"].values())
            sample_counts.append(count)
            total_samples += count

        if total_samples == 0:
            n = len(client_outputs)
            return [1.0 / n] * n

        return [count / total_samples for count in sample_counts]