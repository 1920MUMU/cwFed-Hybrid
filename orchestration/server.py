"""
联邦学习服务器 - 纯调度
"""
import torch
import numpy as np
from collections import OrderedDict
from foundation.model import create_model
from utils.metrics import Metrics
import time



class Server:
    def __init__(self, clients: list, config):
        self.clients = clients
        self.config = config
        self.num_clients = len(clients)
        self.num_classes = config.num_classes

        self.global_model = create_model(config)
        self.personalized_models = None

        # 两个基线模型
        self.fedavg_baseline_model = None      # 数据量加权 → 评估个性化增益

        # 聚合器
        if config.aggregator_mode == "fed":
            from aggregation.fed_aggregator import FedAggregator
            self.aggregator = FedAggregator()
        else:
            from aggregation.cwfed_aggregator import CWFedAggregator
            self.aggregator = CWFedAggregator(num_classes=self.num_classes)

        self.metrics_calculator = Metrics()
        self.history = []

        # EMA 平滑器
        if config.weight_mode == "shapley" and getattr(config, 'ema_enabled', False):
            from aggregation.ema_smoother import EMASmoother
            self.ema_smoother = EMASmoother(
                base_alpha=getattr(config, 'ema_base_alpha', 0.3),
                max_alpha=getattr(config, 'ema_max_alpha', 0.8),
                num_clients=config.num_clients,
                num_classes=config.num_classes,
            )
        else:
            self.ema_smoother = None

        from aggregation.fed_aggregator import FedAggregator
        self.fedavg_aggregator = FedAggregator()

        self.eval_every = getattr(self.config, 'eval_every', 1)

    def _simple_average(self, model_states: list) -> OrderedDict:
        """简单平均聚合（不加权），用于 Shapley/LaDP 基准"""
        avg_state = OrderedDict()
        n = len(model_states)
        if n == 0:
            return avg_state
        for key in model_states[0].keys():
            stacked = torch.stack([ms[key].float() for ms in model_states])
            avg_state[key] = stacked.mean(dim=0)
        return avg_state

    def _get_dp_epsilon_spent(self) -> float:
        """获取所有客户端的总隐私消耗"""
        total = 0.0
        for client in self.clients:
            if hasattr(client, 'privacy_module') and client.privacy_module is not None:
                total += client.privacy_module.accountant.epsilon_spent
        return total

    def run_round(self, round_num: int, evaluate: bool = True) -> dict:
        K = self.num_classes
        N = self.num_clients

        # ===== 阶段1: 下发模型 =====
        for client in self.clients:
            client.current_round = round_num
            if self.config.aggregator_mode == "cwfed" and self.personalized_models:
                client.update_model(self.personalized_models[client.client_id])
            else:
                client.update_model(self.global_model.state_dict())

        # ===== 阶段2: 训练 + 收集模型 =====
        model_states = []
        contributions = []

        for client in self.clients:
            contrib = client.train()
            contributions.append(contrib)
            model_states.append(contrib.model_state)

        original_model_states = list(model_states)

        # ===== 阶段3: 计算聚合权重 =====
        enable_win = getattr(self.config, 'sliding_window_enabled', False)
        fedavg_contribs = self._compute_fedavg_contribs(contributions)

        if self.config.aggregator_mode == "fed":
            if self.config.weight_mode == "shapley":
                # 纯 shapley：余弦相似度归一化
                baseline = self._simple_average(model_states)
                similarities = []
                for client in self.clients:
                    sim = client.compute_similarity(None, baseline)
                    similarities.append(sim["scalar"])
                total = sum(similarities)
                weights = [s/total for s in similarities] if total > 0 else [1.0/N]*N
            elif self.config.weight_mode == "hybrid":
                # 双重加权：样本量 × 相似度
                avg_w = self._compute_fedavg_contribs(contributions)
                baseline = self._simple_average(model_states)
                similarities = []
                for client in self.clients:
                    sim = client.compute_similarity(None, baseline)
                    similarities.append(sim["scalar"])
                hybrid = [avg_w[i] * similarities[i] for i in range(N)]
                total = sum(hybrid)
                weights = [h/total for h in hybrid] if total > 0 else [1.0/N]*N
            else:
                # avg：数据量权重
                weights = fedavg_contribs

            result = self.aggregator.aggregate(
                weights, model_states,
                enable_window=enable_win,
                clients=self.clients if enable_win else None
            )
            self.global_model.load_state_dict(result[0])

        else:  # cwfed
            avg_contribs = self._compute_per_class_avg(contributions)

            if self.config.weight_mode == "shapley":
                # 纯 shapley：余弦相似度归一化
                self.aggregator.aggregate(avg_contribs, model_states, None, mix_contributions=avg_contribs)
                temp_experts = self.aggregator.get_expert_models()
                temp_experts_list = [temp_experts[c] for c in range(K)]

                similarities_per_class = []
                similarities_scalar = []
                for client in self.clients:
                    sim = client.compute_similarity(temp_experts_list)
                    similarities_scalar.append(sim["scalar"])
                    similarities_per_class.append(sim["per_class"])

                # 纯相似度作为 expert_contribs 和 mix_contribs
                expert_contribs = similarities_per_class
                mix_contribs = similarities_per_class

            elif self.config.weight_mode == "hybrid":
                # 双重加权：样本量 / (1 - 相似度)
                self.aggregator.aggregate(avg_contribs, model_states, None, mix_contributions=avg_contribs)
                temp_experts = self.aggregator.get_expert_models()
                temp_experts_list = [temp_experts[c] for c in range(K)]

                similarities_per_class = []
                similarities_scalar = []
                for client in self.clients:
                    sim = client.compute_similarity(temp_experts_list)
                    similarities_scalar.append(sim["scalar"])
                    similarities_per_class.append(sim["per_class"])

                expert_contribs = []
                for i in range(N):
                    expert_contribs.append([
                        avg_contribs[i][c] / max(1.0 - similarities_per_class[i][c], 1e-8)
                        for c in range(K)
                    ])
                mix_contribs = expert_contribs

            else:
                # avg：样本量/L2范数
                expert_contribs = avg_contribs
                mix_contribs = avg_contribs

            self.personalized_models = self.aggregator.aggregate(
                expert_contribs, model_states, None,
                mix_contributions=mix_contribs,
                enable_window=enable_win,
                clients=self.clients if enable_win else None
            )

        # ===== 阶段4: 评估 =====
        fedavg_contribs = self._compute_fedavg_contribs(contributions)
        self.fedavg_baseline_model = self.fedavg_aggregator.aggregate(
            fedavg_contribs, original_model_states
        )[0]
        self.simple_avg_baseline_model = self._simple_average(original_model_states)
        if evaluate:
            stats = self._evaluate(round_num, contributions)
            self.history.append(stats)
        else:
            stats = {"round": round_num, "avg_test_acc": 0.0, "avg_test_f1": 0.0,
                    "avg_test_acc_fedavg": 0.0, "personalization_gain": 0.0,
                    "fairness_cv": 0.0, "avg_train_acc": 0.0,
                    "client_test_accs": [], "client_test_accs_fedavg": [],
                    "pers_correct_total": 0, "pers_total_samples": 0,
                    "base_correct_total": 0, "base_total_samples": 0,
                    "dp_epsilon_spent": 0.0}

        return stats

    def _evaluate(self, round_num: int, contributions: list) -> dict:
        """统一评估"""
        pers_y_true_all = []
        pers_y_pred_all = []
        pers_correct_total = 0
        pers_total_samples = 0

        base_y_true_all = []
        base_y_pred_all = []
        base_correct_total = 0
        base_total_samples = 0

        client_accs_pers = []
        client_accs_base = []

        for i, client in enumerate(self.clients):
            if self.personalized_models is not None:
                p_state = self.personalized_models[i]
            else:
                p_state = self.global_model.state_dict()

            b_state = self.fedavg_baseline_model

            eval_p = client.evaluate(p_state)
            eval_b = client.evaluate(b_state)

            pers_y_true_all.append(eval_p["y_true"])
            pers_y_pred_all.append(eval_p["y_pred"])
            pers_correct_total += eval_p["correct"]
            pers_total_samples += eval_p["total"]

            base_y_true_all.append(eval_b["y_true"])
            base_y_pred_all.append(eval_b["y_pred"])
            base_correct_total += eval_b["correct"]
            base_total_samples += eval_b["total"]

            client_accs_pers.append(eval_p["accuracy"])
            client_accs_base.append(eval_b["accuracy"])

        avg_test_acc = self.metrics_calculator.global_accuracy(
            pers_y_true_all, pers_y_pred_all
        )
        avg_test_f1 = self.metrics_calculator.global_f1_macro(
            pers_y_true_all, pers_y_pred_all
        )
        avg_test_acc_fedavg = self.metrics_calculator.global_accuracy(
            base_y_true_all, base_y_pred_all
        )
        gain = self.metrics_calculator.global_personalization_gain(
            pers_correct_total, pers_total_samples,
            base_correct_total, base_total_samples
        )

        train_accs = [c.metadata.get("train_acc", 0.0) for c in contributions]
        avg_train_acc = np.mean(train_accs) if train_accs else 0.0

        fairness = self.metrics_calculator.fairness_cv(client_accs_pers)

        return {
            "round": round_num,
            "avg_train_acc": avg_train_acc,
            "avg_test_acc": avg_test_acc,
            "avg_test_f1": avg_test_f1,
            "avg_test_acc_fedavg": avg_test_acc_fedavg,
            "personalization_gain": gain,
            "fairness_cv": fairness,
            "client_test_accs": client_accs_pers,
            "client_test_accs_fedavg": client_accs_base,
            "pers_correct_total": pers_correct_total,
            "pers_total_samples": pers_total_samples,
            "base_correct_total": base_correct_total,
            "base_total_samples": base_total_samples,
            "dp_epsilon_spent": self._get_dp_epsilon_spent(),
        }

    def run(self, callback=None) -> list:
        """执行完整联邦训练"""
        start_time = time.time()
        self.history = []
        self.personalized_models = None
        self.fedavg_baseline_model = None
        self.simple_avg_baseline_model = None

        if self.ema_smoother is not None:
            self.ema_smoother.reset()

        mode_desc = f"{self.config.weight_mode}+{self.config.aggregator_mode}"
        ema_info = " (EMA)" if self.ema_smoother is not None else ""
        early_stop = getattr(self.config, 'early_stop_acc', 1.0)

        print(f"\n{'='*50}")
        print(f"开始联邦训练: {mode_desc}{ema_info}")
        print(f"  客户端数: {self.num_clients}")
        print(f"  全局轮数: {self.config.global_rounds}")
        print(f"  本地轮数: {self.config.local_epochs}")
        if early_stop < 1.0:
            print(f"  提前停止阈值: {early_stop:.1%}")
        if self.ema_smoother is not None:
            alpha = self.ema_smoother.compute_alpha(self.config.local_epochs)
            print(f"  EMA alpha: {alpha:.3f}")
        print(f"{'='*50}\n")

        for r in range(self.config.global_rounds):
            stats = self.run_round(r + 1, evaluate=(r % self.eval_every == 0))
            # stats 始终是 dict（不会为 None），history 已在 run_round 内部添加

            if r % self.eval_every == 0:
                print(f"Round {stats['round']:3d} | "
                    f"Acc: {stats['avg_test_acc']:.4f} | "
                    f"F1: {stats['avg_test_f1']:.4f} | "
                    f"FedAvg: {stats['avg_test_acc_fedavg']:.4f} | "
                    f"Gain: {stats['personalization_gain']:+.4f} | "
                    f"CV: {stats['fairness_cv']:.4f}")

                if callback:
                    callback(stats)

                if early_stop < 1.0 and stats['avg_test_acc'] >= early_stop:
                    print(f"\n  ✓ 准确率 {stats['avg_test_acc']:.4f} 达到阈值 {early_stop:.1%}，提前停止")
                    break
            else:
                print(f"Round {r+1:3d} | (跳过评估)")

        elapsed = time.time() - start_time 
        best = self.get_best_round()
        if best:
            print(f"\n训练完成! (共 {len(self.history)} 轮)")
            print(f"  最佳准确率: {best['avg_test_acc']:.4f} (第{best['round']}轮)")
            print(f"  个性化增益: {best['personalization_gain']:+.4f}\n")
        
            from utils.experiment_recorder import ExperimentRecorder
            recorder = ExperimentRecorder()
            exp_id = recorder.save(self.config, self.history, best, elapsed)
            print(f"  实验已保存: {exp_id}")

        return self.history

    def get_best_round(self) -> dict:
        if not self.history:
            return None
        return max(self.history, key=lambda h: h["avg_test_acc"])
    
    def _compute_fedavg_contribs(self, contributions):
        """计算 FedAvg 数据量权重"""
        n = len(contributions)
        total = sum(c.metadata.get("total_samples", 0) for c in contributions)
        if total == 0:
            return [1.0/n] * n
        return [c.metadata.get("total_samples", 0)/total for c in contributions]
    
    def _compute_per_class_avg(self, contributions):
        """avg 模式：每类贡献值。WDR 启用时用分类层 L2 范数，否则用样本数"""
        per_class = []
        for c in contributions:
            # 优先使用 WDR 的 L2 范数
            wdr_norms = c.metadata.get("wdr_weight_norms", None)
            if wdr_norms is not None:
                per_class.append([float(wdr_norms[cls]) for cls in range(self.num_classes)])
            else:
                class_counts = c.metadata.get("class_counts", {})
                per_class.append([
                    float(class_counts.get(cls, 0))
                    for cls in range(self.num_classes)
                ])
        return per_class