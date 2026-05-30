# ShapFed

融合 ShapFed 和 cwFedAvg 的联邦学习个性化框架。

## 特性

- 四种聚合策略：avg+fed / shapley+fed / avg+cwfed / shapley+cwfed
- 类级别联邦聚合 (CWFed)
- Shapley 值驱动的自适应加权
- 差分隐私支持 (Classic DP )
- WDR 权重分布正则化
- 滑动窗口噪声平滑
- 图形化实验界面

## 安装

```bash
git clone https://github.com/你的用户名/shapfed.git
cd shapfed
pip install -r requirements.txt
python main.py