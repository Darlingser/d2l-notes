"""Fashion-MNIST 分类：数据、模型、损失、优化器、评估函数。
训练循环留给你自己写（见文件末尾）。
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "./data"
BATCH_SIZE = 256
NUM_WORKERS = 4
NUM_CLASSES = 10
LR = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
if device.type == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)

# ---------- 1. 数据 ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
])

train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)

train_iter = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
test_iter = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

# ---------- 2. 模型 ----------
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, NUM_CLASSES),
).to(device)

# ---------- 3. 损失与优化器 ----------
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

# ---------- 4. 评估：返回 (平均损失, 准确率) ----------
@torch.no_grad()
def evaluate(net, data_iter):
    net.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in data_iter:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_hat = net(X)
        loss_sum += loss(y_hat, y).item() * y.numel()
        correct += (y_hat.argmax(dim=1) == y).sum().item()
        total += y.numel()
    net.train()
    return loss_sum / total, correct / total

# ---------- 5. 可视化 ----------
CLASSES = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
           'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


def show_images(dataset, n=32, cols=8, start=0):
    """画数据集里的前 n 张图，带类别名。"""
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.7))
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i >= n:
            continue
        idx = (start + i) % len(dataset)
        ax.imshow(dataset.data[idx], cmap='gray')
        ax.set_title(CLASSES[int(dataset.targets[idx])], fontsize=9)
    fig.tight_layout()
    return fig


def _curve(ax, values, label, color, marker):
    if values is None or len(values) == 0:
        return
    epochs = list(range(1, len(values) + 1))
    ax.plot(epochs, values, marker + '-', color=color, label=label)
    if len(epochs) <= 20:
        ax.set_xticks(epochs)


def plot_history(*, train_loss=None, test_loss=None, train_acc=None, test_acc=None,
                 figsize=(11, 4)):
    """左图 loss、右图 acc，训练和测试各一条。

    每个 epoch 各 append 一个数，没记的那条传 None。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    _curve(ax1, train_loss, 'train loss', 'tab:red', 'o')
    _curve(ax1, test_loss, 'test loss', 'tab:orange', 's')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('loss')
    ax1.grid(alpha=0.3)

    _curve(ax2, train_acc, 'train acc', 'tab:blue', 'o')
    _curve(ax2, test_acc, 'test acc', 'tab:green', 's')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('acc')
    ax2.set_title('accuracy')
    ax2.grid(alpha=0.3)

    for ax in (ax1, ax2):
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    fig.tight_layout()
    return fig


@torch.no_grad()
def confusion_matrix(net, data_iter, device=device):
    """返回 10x10 混淆矩阵，行是真值、列是预测。"""
    net.eval()
    cm = torch.zeros(100, dtype=torch.long)
    for X, y in data_iter:
        pred = net(X.to(device)).argmax(dim=1).cpu()
        cm += torch.bincount(y * 10 + pred, minlength=100)
    net.train()
    return cm.reshape(10, 10).numpy()


def plot_confusion(cm, figsize=(7.5, 6)):
    """画混淆矩阵，返回 (fig, 每类准确率)。"""
    cm = np.asarray(cm)
    per_class = np.diag(cm) / cm.sum(axis=1).clip(min=1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(10), labels=CLASSES, rotation=45, ha='right')
    ax.set_yticks(range(10), labels=CLASSES)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')

    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                        color='white' if cm[i, j] > thresh else 'black')

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, per_class


def make_animator(num_epochs, figsize=(6, 4)):
    """d2l 风格的实时曲线，每个 epoch 调一次：

        animator.add(epoch + 1, (train_loss, test_loss, train_acc, test_acc))

    d2l 的 Animator 只画一个坐标轴、最多 4 条线（fmts 就 4 个），多传的会被静默丢掉。
    """
    from d2l import torch as d2l
    # 新版 IPython 删掉了 set_matplotlib_formats，不拦住 d2l 在 __init__ 里就 AttributeError
    d2l.use_svg_display = lambda: None
    return d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'test loss', 'train acc', 'test acc'],
                        figsize=figsize)


# ---------- 6. 训练循环：你自己写 ----------
if __name__ == "__main__":
    num_epochs = 10
    animator = make_animator(num_epochs)
    for epoch in range(num_epochs):
        for X, y in train_iter:
         X=X.to(device)
         y=y.to(device)
         optimizer.zero_grad()
         y_hat=net(X)
         l=loss(y_hat,y)
         l.backward()
         optimizer.step()
    animator.add(epoch + 1, (train_loss, test_loss, train_acc, test_acc))