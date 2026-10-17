import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASSES = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

MEAN = 0.2860
STD = 0.3530


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP on Fashion-MNIST.")
    parser.add_argument("--img-size", type=int, default=28, choices=[28, 32])
    parser.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def build_transforms(img_size):
    ops = []
    if img_size != 28:
        ops.append(transforms.Resize((img_size, img_size)))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(MEAN,), std=(STD,)),
        ]
    )
    return transforms.Compose(ops)


def build_model(img_size, activation, num_classes=10):
    act = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}[activation]
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(img_size * img_size, 256),
        act(),
        nn.Linear(256, 128),
        act(),
        nn.Linear(128, num_classes),
    )


@torch.no_grad()
def evaluate(net, data_iter, loss_fn, device):
    net.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in data_iter:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = net(x)
        loss_sum += loss_fn(logits, y).item() * y.numel()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total


@torch.no_grad()
def confusion_matrix(net, data_iter, device, num_classes=10):
    net.eval()
    cm = torch.zeros(num_classes * num_classes, dtype=torch.long)
    for x, y in data_iter:
        pred = net(x.to(device)).argmax(dim=1).cpu()
        cm += torch.bincount(y * num_classes + pred, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).numpy()


def save_history(path, history):
    fields = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def save_per_class(path, cm):
    true_positive = np.diag(cm)
    predicted = cm.sum(axis=0)
    support = cm.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name, p, r, score, count in zip(CLASSES, precision, recall, f1, support):
            writer.writerow([name, f"{p:.6f}", f"{r:.6f}", f"{score:.6f}", int(count)])


def plot_history(path, history):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [r["train_loss"] for r in history], "o-", label="train loss")
    axes[0].plot(epochs, [r["test_loss"] for r in history], "s--", label="test loss")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [r["train_acc"] for r in history], "o-", label="train acc")
    axes[1].plot(epochs, [r["test_acc"] for r in history], "s--", label="test acc")
    axes[1].set_title("accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(path, cm):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    threshold = cm.max() / 2
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            if cm[row, col] == 0:
                continue
            color = "white" if cm[row, col] > threshold else "black"
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    lr = args.lr if args.lr is not None else (1e-3 if args.optimizer == "adam" else 0.5)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transforms(args.img_size)
    train_set = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(args.seed)
    train_iter = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    test_iter = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    net = build_model(args.img_size, args.activation).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = (
        torch.optim.Adam(net.parameters(), lr=lr)
        if args.optimizer == "adam"
        else torch.optim.SGD(net.parameters(), lr=lr)
    )

    params = sum(p.numel() for p in net.parameters())
    run_id = f"img{args.img_size}_{args.activation}_{args.optimizer}_ep{args.epochs}_seed{args.seed}"
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "img_size": args.img_size,
        "activation": args.activation,
        "optimizer": args.optimizer,
        "lr": lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "params": params,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("config:", config)
    history = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for x, y in train_iter:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = net(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()

        train_loss = loss_sum / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(net, test_iter, loss_fn, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "test_loss": round(test_loss, 6),
                "test_acc": round(test_acc, 6),
            }
        )
        print(
            f"epoch {epoch:2d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| test loss {test_loss:.4f} acc {test_acc:.4f}",
            flush=True,
        )

    cm = confusion_matrix(net, test_iter, device)
    save_history(run_dir / "history.csv", history)
    save_per_class(run_dir / "per_class.csv", cm)
    np.savetxt(run_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_history(run_dir / "curve.png", history)
    plot_confusion(run_dir / "confusion_matrix.png", cm)

    final = history[-1]
    summary = {
        "run_id": run_id,
        "config": config,
        "final_train_loss": final["train_loss"],
        "final_train_acc": final["train_acc"],
        "final_test_loss": final["test_loss"],
        "final_test_acc": final["test_acc"],
        "gap_pt": round((final["train_acc"] - final["test_acc"]) * 100, 2),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("summary:", json.dumps(summary, ensure_ascii=False))
    print("saved to:", run_dir)


if __name__ == "__main__":
    main()
