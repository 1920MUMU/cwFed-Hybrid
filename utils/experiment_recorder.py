"""
实验记录器：保存/加载训练历史
"""
import json
import os
import hashlib
import time
from datetime import datetime


class ExperimentRecorder:
    """实验记录管理"""
    
    def __init__(self, save_dir: str = "experiments"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def make_id(self, config) -> str:
        cfg = self._get_config_dict(config)
        raw = json.dumps(cfg, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]
    
    def _get_config_dict(self, config) -> dict:
        """提取关键配置为字典"""
        return {
            "dataset": config.dataset,
            "num_clients": config.num_clients,
            "num_classes": config.num_classes,
            "distribution": config.distribution,
            "dirichlet_alpha": getattr(config, 'dirichlet_alpha', 0.5),
            "weight_mode": config.weight_mode,
            "aggregator_mode": config.aggregator_mode,
            "global_rounds": config.global_rounds,
            "local_epochs": config.local_epochs,
            "noise_ratio": getattr(config, 'noise_ratio', 0),
            "dp_enabled": getattr(config, 'dp_enabled', False),
            "dp_mode": getattr(config, 'dp_mode', ''),
            "dp_epsilon": getattr(config, 'dp_epsilon', 0),
            "early_stop_acc": getattr(config, 'early_stop_acc', 1.0),
            "wdr_enabled": getattr(config, 'wdr_enabled', False),
            "wdr_lambda": getattr(config, 'wdr_lambda', 1.0),
            "reverse_mix": getattr(config, 'reverse_mix', False),
            "sliding_window_enabled": getattr(config, 'sliding_window_enabled', False),
        }
    
    def exists(self, config) -> bool:
        """检查相同参数是否已有记录"""
        exp_id = self.make_id(config)
        path = os.path.join(self.save_dir, f"{exp_id}.json")
        return os.path.exists(path)
    
    def save(self, config, history: list, best_round: dict,
             elapsed_time: float = 0) -> str:
        """保存实验记录"""
        exp_id = self.make_id(config)
        path = os.path.join(self.save_dir, f"{exp_id}.json")
        
        # 提取关键指标
        metrics = []
        for h in history:
            metrics.append({
                "round": h["round"],
                "train_acc": round(h["avg_train_acc"], 4),
                "test_acc": round(h["avg_test_acc"], 4),
                "test_f1": round(h["avg_test_f1"], 4),
                "fedavg_acc": round(h.get("avg_test_acc_fedavg", 0), 4),
                "gain": round(h["personalization_gain"], 4),
                "fairness": round(h["fairness_cv"], 4),
            })
        
        record = {
            "exp_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed_time, 1),
            "config": self._get_config_dict(config),  # ← 复用
            "best_round": best_round["round"] if best_round else 0,
            "best_acc": round(best_round["avg_test_acc"], 4) if best_round else 0,
            "best_gain": round(best_round["personalization_gain"], 4) if best_round else 0,
            "total_rounds": len(history),
            "history": metrics,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        
        return exp_id
    
    def load(self, exp_id: str) -> dict:
        """加载实验记录"""
        path = os.path.join(self.save_dir, f"{exp_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_by_config(self, config) -> dict:
        """根据配置加载已有记录"""
        exp_id = self.make_id(config)
        return self.load(exp_id)
    
    def list_all(self) -> list:
        records = []
        for fname in os.listdir(self.save_dir):
            if fname.endswith('.json'):
                path = os.path.join(self.save_dir, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    r = json.load(f)
                cfg = r['config']
                
                # 构建显示文本
                parts = [
                    f"{cfg['weight_mode']}+{cfg['aggregator_mode']}",
                    f"{cfg['dataset']}",
                    f"C={cfg['num_clients']}",
                    f"K={cfg.get('num_classes', '?')}",                       # ←
                ]
                if cfg.get('distribution') in ['heterogeneous', 'noisy']:
                    parts.append(f"α={cfg.get('dirichlet_alpha', '?')}")      # ←
                if cfg.get('dp_enabled'):
                    parts.append(f"ε={cfg.get('dp_epsilon', '?')}")           # ←
                
                records.append({
                    "exp_id": r["exp_id"],
                    "timestamp": r["timestamp"][:16],
                    "config": " | ".join(parts),
                    "best_acc": r["best_acc"],
                    "rounds": r["total_rounds"],
                })
        return sorted(records, key=lambda r: r["timestamp"], reverse=True)