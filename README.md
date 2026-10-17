# Fashion-MNIST 分类实验

用 PyTorch 训练 MLP，比较激活函数、输入尺寸、优化器和隐藏层数。可复现实验由 `train.py` 完成，详细指标保存在 `results/`。`Fashion-MNIST.ipynb` 用于逐步查看数据、模型、训练过程和可视化。

## 运行

```bash
uv sync
python train.py --img-size 28 --activation relu --optimizer adam --epochs 10 --seed 42
```

常用参数：

```text
--img-size   28 或 32
--activation relu 或 sigmoid
--optimizer  adam 或 sgd
--lr         默认 Adam 1e-3，SGD 0.5
--epochs     训练轮数
--batch-size 默认 256
--seed       默认 42
```

每次运行会在 `results/<run_id>/` 下保存配置、逐 epoch 指标、每类指标、混淆矩阵和曲线。
相同参数会写入同一个 `run_id` 目录并覆盖旧文件；改变图像尺寸、激活函数、优化器或 epoch 会生成新目录。

## 数据与模型

Fashion-MNIST 共有 60,000 张训练图、10,000 张测试图，原始尺寸为 28×28，共 10 类。代码使用 `mean=0.2860`、`std=0.3530` 做归一化。

28×28 的模型结构：

```python
nn.Flatten(),
nn.Linear(28 * 28, 256), nn.ReLU(),
nn.Linear(256, 128), nn.ReLU(),
nn.Linear(128, 10)
```

`28 * 28 = 784` 不是固定规则，而是 Fashion-MNIST 原始图片展开后的长度。改成 32×32 后，第一层自动变成 `nn.Linear(32 * 32, 256)`，输入维度是 1024。

`nn.Linear(in_features, out_features)` 的参数量为：

```text
in_features * out_features + out_features
```

28×28 模型的计算：

| 层 | 参数量 |
|---|---:|
| Linear(784, 256) | 200,960 |
| Linear(256, 128) | 32,896 |
| Linear(128, 10) | 1,290 |
| 合计 | 235,146 |

32×32 模型的第一层是 Linear(1024, 256)，总参数量为 296,586。
如果改了 `IMG_SIZE` 但第一层仍写 784，32×32 输入会报维度不匹配。`train.py` 用 `img_size * img_size` 自动计算，不会出现这个问题。

## 实验结果

以下结果全部由 `train.py` 运行，seed=42，训练 10 epoch：

| 配置 | 参数量 | train acc | test acc | train/test gap | test loss |
|---|---:|---:|---:|---:|---:|
| 28×28, ReLU, Adam | 235,146 | 92.20% | 88.70% | 3.50 pt | 0.3300 |
| 32×32, ReLU, Adam | 296,586 | 91.46% | 89.02% | 2.44 pt | 0.3164 |
| 28×28, ReLU, SGD lr=0.5 | 235,146 | 91.51% | 88.52% | 2.99 pt | 0.3355 |
| 28×28, Sigmoid, SGD lr=0.5 | 235,146 | 88.05% | 86.21% | 1.84 pt | 0.3822 |

逐 epoch 数据和每组曲线在 `results/`。`results/summary.csv` 是汇总表，`results/activation_compare.csv` 是 ReLU/Sigmoid 每轮对比，`results/activation_compare.png` 是对应曲线。

旧 notebook 还记录过两组更深的模型，但没有用当前 `train.py` 重跑：

| 配置 | 参数量 | test acc | gap | test loss |
|---|---:|---:|---:|---:|
| 3 隐藏层, ReLU, Adam, 10 epoch | 300,938 | 89.10% | 3.05 pt | 0.3242 |
| 3 隐藏层 + Dropout 0.2, 15 epoch | 300,938 | 89.03% | 2.42 pt | - |

这两行只作为历史记录。下面的结论主要依据前面四组可复现实验。

## 结论

### 1. ReLU 比 Sigmoid 更合适

在相同条件（28×28、SGD lr=0.5、10 epoch、seed=42）下：

```text
ReLU    88.52%
Sigmoid 86.21%
```

ReLU 高 2.31 pt。第 1 个 epoch 的差距也很明显：ReLU 的 train acc 是 74.58%，Sigmoid 是 59.66%。Sigmoid 的 train/test gap 更小（1.84 pt 对 2.99 pt），但它的 train acc 和 test acc 都更低，不能说明它泛化更好。

### 2. 28×28 和 32×32 的差距不足以证明谁更好

32×32 的 test acc 高 0.32 pt，但只跑了一个 seed，差异小于 0.5 pt。Fashion-MNIST 原图就是 28×28，放大到 32×32 只是插值，没有增加新信息，同时第一层参数从 200,960 增加到 262,400。因此当前数据只能说明两者接近，不能说明 32×32 更好。

### 3. 增加隐藏层带来的收益很小

旧记录中，3 隐藏层比 2 隐藏层高 0.40 pt，但参数量从 235,146 增加到 300,938，多了约 28%。这说明当前模型的瓶颈不只是网络容量。

### 4. 当前是轻度过拟合，不是严重过拟合

28×28 ReLU Adam 的最终 train acc 是 92.20%，test acc 是 88.70%，gap 为 3.50 pt。test loss 在第 6 轮附近约为 0.3285，最后一轮是 0.3300，没有明显持续上升，test acc 也还在缓慢提高。

所以目前是轻度过拟合，同时训练也还没有完全收敛。判断过拟合不能只看 train/test 差距，还要看 test loss 是否持续上升、test acc 是否停止提高。若后续 gap 继续扩大且 test loss 上升，再考虑加强正则化。

### 5. 主要错误集中在上衣类

28×28 ReLU Adam 的每类结果：

| 类别 | precision | recall | F1 |
|---|---:|---:|---:|
| shirt | 76.05% | 63.20% | 69.03% |
| coat | 75.93% | 87.40% | 81.26% |
| pullover | 81.72% | 79.60% | 80.65% |
| t-shirt | 83.40% | 84.90% | 84.14% |
| trouser | 98.29% | 97.70% | 97.99% |
| bag | 97.10% | 97.10% | 97.10% |

混淆矩阵中的主要错误：

```text
shirt -> t-shirt      120 张
shirt -> coat         109 张
shirt -> pullover      86 张
pullover -> coat      123 张
coat -> pullover       71 张
ankle boot -> sneaker  56 张
```

`shirt`、`coat`、`pullover`、`t-shirt` 的轮廓和纹理接近，是当前 MLP 的主要失分来源。完整结果见 `results/img28_relu_adam_ep10_seed42/per_class.csv` 和 `confusion_matrix.csv`。

## 注意事项

- 修改 notebook 参数后先重启内核并从头运行，或直接使用 `train.py`。只改中间 cell 再运行，`net`、`optimizer` 和 `DataLoader` 可能还是旧配置。
- 目前每组只跑了一个 seed。小于 0.5 pt 的差异不能当作确定结论，建议用 3 个 seed 取平均值。
- 比较激活函数时必须固定优化器、学习率、层数和 epoch。ReLU Adam 与 Sigmoid SGD 的差异同时包含激活函数和优化器的影响。
- notebook 中的一次运行是 88.66%，`train.py` 同一配置是 88.70%。两者数据加载和随机顺序实现不同，0.04 pt 属于正常波动。正式对照以 `results/` 为准。
- 旧图片 `optimizer_compare.png`、`layer_compare.png`、`dropout_compare.png`、`training_curves.png` 来自之前的 notebook，未纳入本次 `train.py` 复现。

## 文件说明

| 文件 | 内容 |
|---|---|
| `train.py` | 可复现训练和评估脚本 |
| `Fashion-MNIST.ipynb` | 逐步讲解和交互运行 |
| `results/summary.csv` | 所有运行的汇总 |
| `results/activation_compare.csv` | ReLU/Sigmoid 逐 epoch 对比 |
| `results/activation_compare.png` | ReLU/Sigmoid test acc 和 test loss 曲线 |
| `results/<run_id>/history.csv` | 每组逐 epoch 指标 |
| `results/<run_id>/per_class.csv` | 每组每类 precision/recall/F1 |
| `results/<run_id>/confusion_matrix.csv` | 每组 10×10 混淆矩阵 |
| `results/<run_id>/curve.png` | 每组 loss/accuracy 曲线 |
| `results/<run_id>/confusion_matrix.png` | 每组混淆矩阵图 |
