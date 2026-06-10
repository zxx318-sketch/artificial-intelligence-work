"""
Exp-7: CNN 训练脚本（纯 NumPy / minitorch）
构建一个简单 CNN 并在 MNIST / Fashion-MNIST 上训练，记录完整日志。

增强版支持：
  --dataset     mnist / fashion
  --init_mode   he / xavier
  --activation  relu / sigmoid / tanh / leaky_relu
  --dropout_p   Dropout 概率（放在全连接层前）
  --optimizer   adam / sgd
  --hidden      全连接层隐藏单元数
  --weight_decay L2 系数
  --save_plots  是否保存图片
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from minitorch.nn import (
    Conv2d, MaxPool2d, Flatten, Linear, ReLU, Sigmoid, Tanh, LeakyReLU, Dropout,
    Sequential, CrossEntropyLoss, ACTIVATION_REGISTRY, get_activation
)
from minitorch.optim import Adam, SGD
from minitorch.MNISTIterator import load_mnist, load_fashion_mnist, MNISTIterator
from minitorch.tensor_ops import SimpleBackend
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.autodiff import no_grad


def build_cnn(init_mode="he", activation="relu", dropout_p=0.0,
              hidden=128, num_classes=10, backend=SimpleBackend):
    """
    CNN 架构（输入 1x28x28）：
      Conv(1,32,3x3,p=1) -> Act -> MaxPool(2,2)   # 28->14
      Conv(32,64,3x3,p=0) -> Act -> MaxPool(2,2)  # 12->6
      Flatten -> [Dropout] -> Linear(64*6*6, hidden) -> Act -> Linear(hidden, num_classes)
    """
    act_cls = get_activation(activation)
    layers = [
        Conv2d(1, 32, 3, 3, padding=1),
        act_cls(),
        MaxPool2d((2, 2)),
        Conv2d(32, 64, 3, 3, padding=0),
        act_cls(),
        MaxPool2d((2, 2)),
        Flatten(),
    ]
    if dropout_p > 0.0:
        layers.append(Dropout(p=dropout_p))
    layers.append(Linear(64 * 6 * 6, hidden, bias=True, init_mode=init_mode))
    layers.append(act_cls())
    layers.append(Linear(hidden, num_classes, bias=True, init_mode=init_mode))
    model = Sequential(*layers)

    if backend is not SimpleBackend:
        for param in model.parameters():
            if param.value is not None:
                param_np = param.value.to_numpy()
                new_val = tensor_from_numpy(param_np, backend=backend)
                param.update(new_val)
    return model


def compute_accuracy(logits, labels_onehot):
    pred_np = np.argmax(logits.to_numpy(), axis=1)
    true_np = np.argmax(labels_onehot.to_numpy(), axis=1)
    return np.mean(pred_np == true_np)


def evaluate(model, data_iterator, loss_fn, max_batches=0, return_preds=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0
    all_preds = []
    all_trues = []

    with no_grad():
        for images, labels in data_iterator:
            if max_batches > 0 and batch_count >= max_batches:
                break
            batch_size = images.shape[0]
            logits = model(images)
            logits_np = logits.to_numpy()
            labels_np = labels.to_numpy()
            pred_np = np.argmax(logits_np, axis=1)
            true_np = np.argmax(labels_np, axis=1)
            total_correct += np.sum(pred_np == true_np)

            max_logits = logits_np.max(axis=1, keepdims=True)
            shifted = logits_np - max_logits
            log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
            ce = -np.sum(labels_np * log_probs, axis=1)
            total_loss += np.mean(ce) * batch_size

            total_samples += batch_size
            batch_count += 1
            if return_preds:
                all_preds.extend(pred_np.tolist())
                all_trues.extend(true_np.tolist())

    model.train()
    avg_loss = float(total_loss / total_samples)
    acc = float(total_correct / total_samples)
    if return_preds:
        return avg_loss, acc, np.array(all_preds), np.array(all_trues)
    return avg_loss, acc


def train_cnn(args):
    # 数据
    if args.dataset == "fashion":
        print("Loading Fashion-MNIST...")
        train_images, train_labels = load_fashion_mnist(args.data_dir, kind="train")
        test_images, test_labels = load_fashion_mnist(args.data_dir, kind="test")
    else:
        print("Loading MNIST...")
        train_images, train_labels = load_mnist(args.data_dir, kind="train")
        test_images, test_labels = load_mnist(args.data_dir, kind="test")

    train_iter = MNISTIterator(train_images, train_labels, args.batch_size, shuffle=True, backend="cpu")
    test_iter = MNISTIterator(test_images, test_labels, args.batch_size, shuffle=False, backend="cpu")

    # 模型
    model = build_cnn(
        init_mode=args.init_mode,
        activation=args.activation,
        dropout_p=args.dropout_p,
        hidden=args.hidden,
        backend=SimpleBackend
    )
    print(model)

    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Optimizer: Adam(lr={args.lr}, wd={args.weight_decay})")
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Optimizer: SGD(lr={args.lr}, wd={args.weight_decay})")
    loss_fn = CrossEntropyLoss()

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    epoch_times = []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        batch_count = 0

        for images, labels in train_iter:
            logits = model(images)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.to_numpy().item()
            total_loss += loss_val * images.shape[0]
            pred_np = np.argmax(logits.to_numpy(), axis=1)
            true_np = np.argmax(labels.to_numpy(), axis=1)
            total_correct += np.sum(pred_np == true_np)
            total_samples += images.shape[0]
            batch_count += 1

            if args.log_interval > 0 and batch_count % args.log_interval == 0:
                batch_acc = np.sum(pred_np == true_np) / images.shape[0]
                print(f"  Epoch {epoch+1}/{args.epochs} | Batch {batch_count}/{train_iter.num_batches} | "
                      f"Loss: {loss_val:.4f} | Acc: {batch_acc:.4f}")

        epoch_time = time.time() - t0
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        test_loss, test_acc = evaluate(model, test_iter, loss_fn, max_batches=args.test_batches)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # 最终评估 + 混淆矩阵
    final_test_loss, final_test_acc, preds, trues = evaluate(
        model, test_iter, loss_fn, return_preds=True
    )

    cm = np.zeros((10, 10), dtype=np.int32)
    for p, t in zip(preds, trues):
        cm[t, p] += 1

    # 保存日志
    log_data = {
        "config": {
            "model": "cnn",
            "dataset": args.dataset,
            "init_mode": args.init_mode,
            "activation": args.activation,
            "dropout_p": args.dropout_p,
            "hidden": args.hidden,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "results": {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "epoch_times": epoch_times,
            "final_test_acc": final_test_acc,
            "best_test_acc": max(test_accs),
        }
    }
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "log_cnn.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(args.save_dir, "confusion_matrix.npy"), cm)
    print(f"\n[CNN] Final Test Acc: {final_test_acc:.4f}, Best: {max(test_accs):.4f}")
    print(f"[CNN] Log saved to {log_path}")

    # 绘图
    if args.save_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
        ax1.plot(epochs, test_losses, 'r-s', label='Test Loss')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax1.set_title('CNN Loss Curves')

        ax2.plot(epochs, train_accs, 'b-o', label='Train Acc')
        ax2.plot(epochs, test_accs, 'r-s', label='Test Acc')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        ax2.set_title('CNN Accuracy Curves')
        plt.tight_layout()

        fig_path = os.path.join(os.path.dirname(args.save_dir), "figures", "cnn_curves.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[CNN] Curves saved to {fig_path}")

        # 混淆矩阵图
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap='Blues')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('CNN Confusion Matrix')
        ax.set_xticks(np.arange(10)); ax.set_yticks(np.arange(10))
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='darkred', fontsize=8)
        plt.tight_layout()
        fig_path_cm = os.path.join(os.path.dirname(args.save_dir), "figures", "confusion_matrix.png")
        plt.savefig(fig_path_cm, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[CNN] Confusion matrix saved to {fig_path_cm}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout_p", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion"])
    parser.add_argument("--init_mode", type=str, default="he", choices=["he", "xavier"])
    parser.add_argument("--activation", type=str, default="relu",
                        choices=list(ACTIVATION_REGISTRY.keys()))
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--test_batches", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="experiments/results")
    parser.add_argument("--save_plots", action="store_true", default=True)
    parser.add_argument("--log_interval", type=int, default=0,
                        help="Log every N batches (0 = only epoch logs)")
    args = parser.parse_args()
    train_cnn(args)
