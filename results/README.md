# 实验结果说明

本目录保存 `train.py` 每次运行的原始结果。

## 文件

```text
results/
  summary.csv
  activation_compare.csv
  activation_compare.png
  <run_id>/
    config.json
    history.csv
    per_class.csv
    confusion_matrix.csv
    confusion_matrix.png
    curve.png
    summary.json
```

- `config.json`：图像尺寸、激活函数、优化器、学习率、epoch、batch size、seed、参数量、设备和 torch 版本。
- `history.csv`：每个 epoch 的 train/test loss 和 accuracy。
- `per_class.csv`：10 个类别的 precision、recall、F1 和样本数。
- `confusion_matrix.csv`：10×10 混淆矩阵，行是真值，列是预测。
- `summary.json`：最后一次 epoch 的汇总和 train/test gap。
- `summary.csv`：所有运行的一行汇总。
- `activation_compare.csv`：ReLU 和 Sigmoid 在相同 SGD 配置下逐 epoch 对比。
- `activation_compare.png`：根据 `activation_compare.csv` 生成的 test acc 和 test loss 曲线。

## 可复现实验

| 配置 | 参数量 | train acc | test acc | gap | test loss |
|---|---:|---:|---:|---:|---:|
| 28×28, ReLU, Adam | 235,146 | 92.20% | 88.70% | 3.50 pt | 0.3300 |
| 32×32, ReLU, Adam | 296,586 | 91.46% | 89.02% | 2.44 pt | 0.3164 |
| 28×28, ReLU, SGD lr=0.5 | 235,146 | 91.51% | 88.52% | 2.99 pt | 0.3355 |
| 28×28, Sigmoid, SGD lr=0.5 | 235,146 | 88.05% | 86.21% | 1.84 pt | 0.3822 |

全部为 seed=42、10 epoch。

## 如何阅读

1. 先看 `summary.csv`，比较最终 test acc、test loss 和 gap。
2. 再看对应目录的 `history.csv` 或 `curve.png`，确认 test loss 是否持续上升。
3. 最后看 `per_class.csv` 和混淆矩阵，定位具体错分。
4. 比较激活函数时，使用 `activation_compare.csv`。这里 ReLU 和 Sigmoid 都使用 SGD lr=0.5，其他条件相同。

## 主要结论

- ReLU 比 Sigmoid 高 2.31 pt（88.52% 对 86.21%）。
- 32×32 比 28×28 高 0.32 pt，只有一个 seed，不能证明 32×32 更好。
- 28×28 ReLU Adam 的 train/test gap 是 3.50 pt，test loss 没有持续上升，属于轻度过拟合。
- 主要弱类是 `shirt`、`coat`、`pullover`，它们经常互相错分。

## 注意

- `summary.csv` 中 `source=README historical` 的两行来自旧 notebook，未用当前 `train.py` 重跑。
- 每组只有 seed=42 一次结果。小于 0.5 pt 的差异需要多 seed 验证。
- 不同硬件和软件版本可能产生小幅数值差异。
