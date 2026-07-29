# 6PPD/6PPD-Q 降解预测平台

公开网页版入口：

https://andy123-code.github.io/6PPD-6PPD-Q-DegPred/index.html

如果 GitHub Pages 显示的是本文档，请强制刷新或点击上面的公开网页版入口。

该版本将机理模拟和实验数据驱动模型分开：`degradation_model.py` 用于生成条件化动力学基线，`train.py` 用实验 CSV 训练预测模型，`predict.py` 用已训练模型批量预测。

## 准备数据

`degradation_data_template.csv` 是字段模板，`degradation_data.csv` 是可直接跑通流程的示例训练集。每一行代表一个时间点。至少应包含：试验批次、时间、介质、氧化还原条件、温度、pH，以及 6PPD/6PPD-Q 浓度目标列。推荐同时记录含水率、Eh 或 DO、foc、光照、NOM、硝酸盐。

在 `train.py` 的 `CONFIG` 中设置：

- `data_file`：你的 CSV 路径；
- `target_cols`：需要预测的浓度或动力学指标；
- `group_col`：同一培养瓶或地点的批次列。设置后会按批次分组切分，避免同一时间序列泄漏到训练集和测试集；
- `time_col`：时间列名。默认 `feature_cols` 已包含 `时间_d`，用于学习浓度随时间变化。

## 训练与预测

```bash
cd /Users/andy/Desktop/algorithm_optimized_v1
python3 train.py
python3 predict.py degradation_data_template.csv --save
```

训练使用完整预处理管道、分组交叉验证、`log1p` 目标变换和非负输出约束。模型性能报告必须基于独立批次的测试集和交叉验证，不应只看训练集 R²。

## 机理模型

`TwoSpeciesFateModel` 同时追踪 6PPD 和 6PPD-Q。默认土壤半衰期是文献先验而非通用常数：好氧和厌氧条件分别建模，水体中加入光照、NOM 和硝酸盐的条件项。用于特定土壤/水体前，请根据你的实测时间序列校准参数。
