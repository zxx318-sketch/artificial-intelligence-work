"""
MNIST 训练脚本 —— 纯 minitorch 实现

支持超参数：
  --lr          学习率
  --weight_decay L2 正则化系数
  --dropout_p   Dropout 概率
  --hidden      隐藏层大小 (如 "128,64")
  --epochs      训练轮数
  --batch_size  批大小
  --init_mode   权重初始化方式 (he / xavier)

记录量（服务于组员实验报告）：
  train_losses, test_losses, train_accs, test_accs, epoch_times, grad_norms
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minitorch.nn import (
    Linear, ReLU, Sigmoid, Tanh, Dropout, Flatten,
    Sequential, CrossEntropyLoss
)
from minitorch.optim import SGD, Adam
from minitorch.MNISTIterator import load_mnist, MNISTIterator
from minitorch.tensor_ops import SimpleBackend
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.autodiff import no_grad


# ============================================================================
# 超参数配置
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP on MNIST with minitorch")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="L2 regularization coefficient (0 = no L2)")
    parser.add_argument("--dropout_p", type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument("--hidden", type=str, default="128,64",
                        help="Comma-separated hidden layer sizes, e.g. '128,64'")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--init_mode", type=str, default="he",
                        choices=["he", "xavier"], help="Weight initialization mode")
    parser.add_argument("--activation", type=str, default="sigmoid",
                        choices=["relu", "sigmoid", "tanh"], help="Activation function")
    parser.add_argument("--data_dir", type=str, default="./data", help="MNIST data directory")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log every N batches (0 = only epoch logs)")
    parser.add_argument("--save_plots", action="store_true", default=True,
                        help="Save loss/accuracy plots")
    parser.add_argument("--save_log", action="store_true", default=True,
                        help="Save training log as JSON")
    parser.add_argument("--test_batches", type=int, default=0,
                        help="Max test batches per evaluation (0 = all, set 10 for speed)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device to run on (cpu or cuda)")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adam"], help="Optimizer type")
    return parser.parse_args()


# ============================================================================
# 模型构建
# ============================================================================

def get_activation(name: str):
    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    else:
        raise ValueError(f"Unknown activation: {name}")


def build_model(input_dim: int, hidden_sizes, num_classes: int,
                dropout_p: float, init_mode: str, activation_name: str,
                backend=SimpleBackend):
    """
    构建 Sequential 模型:
      Flatten → Linear(input, h1) → Act → Dropout → ... → Linear(hn, num_classes)
    
    Args:
        backend: TensorBackend（SimpleBackend 或 CUDABackend），参数会被转换为对应后端
    """
    layers = [Flatten()]
    in_size = input_dim
    for h in hidden_sizes:
        layers.append(Linear(in_size, h, bias=True, init_mode=init_mode))
        layers.append(get_activation(activation_name))
        if dropout_p > 0.0:
            layers.append(Dropout(p=dropout_p))
        in_size = h
    # 最后一层：输出 logits（不加激活和 dropout）
    layers.append(Linear(in_size, num_classes, bias=True, init_mode=init_mode))
    model = Sequential(*layers)

    # 如果目标 backend 不是 SimpleBackend，转换所有参数
    if backend is not SimpleBackend:
        for param in model.parameters():
            if param.value is not None:
                param_np = param.value.to_numpy()
                new_val = tensor_from_numpy(param_np, backend=backend)
                param.update(new_val)

    return model


# ============================================================================
# 训练 & 评估
# ============================================================================

def compute_accuracy(logits, labels_onehot):
    """
    Args:
        logits: Tensor, shape (batch, num_classes)
        labels_onehot: Tensor, shape (batch, num_classes)
    Returns:
        accuracy: float in [0, 1]
    """
    pred_np = np.argmax(logits.to_numpy(), axis=1)
    true_np = np.argmax(labels_onehot.to_numpy(), axis=1)
    return np.mean(pred_np == true_np)


def compute_grad_norm(parameters):
    """计算所有参数梯度的 L2 范数"""
    total_norm_sq = 0.0
    for p in parameters:
        if p.value is not None and hasattr(p.value, 'grad') and p.value.grad is not None:
            g = p.value.grad.to_numpy()
            total_norm_sq += np.sum(g ** 2)
    return float(np.sqrt(total_norm_sq))


def evaluate(model, data_iterator, loss_fn, max_batches: int = 0):
    """
    在测试集上评估 loss 和 accuracy。
    测试时不构建计算图，直接用 numpy 计算以加速。
    
    Args:
        max_batches: 最多评估的 batch 数（0 = 全部），用于加速
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    with no_grad():
        for images, labels in data_iterator:
            if max_batches > 0 and batch_count >= max_batches:
                break
            batch_size = images.shape[0]
            
            # 前向传播（no_grad 模式下不构建计算图）
            logits = model(images)                # (batch, 10)
            
            # 直接用 numpy 算 accuracy（不走 Tensor 图）
            logits_np = logits.to_numpy()
            labels_np = labels.to_numpy()
            pred_np = np.argmax(logits_np, axis=1)
            true_np = np.argmax(labels_np, axis=1)
            total_correct += np.sum(pred_np == true_np)
            
            # 用 numpy 算 cross-entropy loss（不走 Tensor 图）
            # log_softmax: logits - max - log(sum(exp(logits - max)))
            max_logits = logits_np.max(axis=1, keepdims=True)
            shifted = logits_np - max_logits
            log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
            ce = -np.sum(labels_np * log_probs, axis=1)
            total_loss += np.mean(ce) * batch_size
            
            total_samples += batch_size
            batch_count += 1

    model.train()  # 恢复训练模式
    avg_loss = float(total_loss / total_samples)
    acc = float(total_correct / total_samples)
    return avg_loss, acc


def train(args):
    # ---- 0. 选择后端 ----
    if args.device == "cuda":
        try:
            from minitorch.cuda_kernel_ops import CUDABackend
            backend = CUDABackend
            print("[Device] Using CUDA (GPU) backend")
        except Exception as e:
            print(f"[Warning] CUDA backend not available ({e}), falling back to CPU")
            backend = SimpleBackend
    else:
        backend = SimpleBackend
        print("[Device] Using CPU backend")

    # ---- 1. 数据加载 ----
    print("Loading MNIST data...")
    train_images, train_labels = load_mnist(args.data_dir, kind="train")
    test_images, test_labels = load_mnist(args.data_dir, kind="test")
    print(f"  Train: {train_images.shape[0]} samples")
    print(f"  Test:  {test_images.shape[0]} samples")

    train_iter = MNISTIterator(train_images, train_labels, args.batch_size,
                                shuffle=True, backend=args.device)
    test_iter = MNISTIterator(test_images, test_labels, args.batch_size,
                               shuffle=False, backend=args.device)

    # ---- 2. 模型构建 ----
    input_dim = 28 * 28   # 784
    num_classes = 10
    hidden_sizes = [int(x.strip()) for x in args.hidden.split(",")]

    print(f"\nBuilding model: input={input_dim} → hidden={hidden_sizes} → output={num_classes}")
    print(f"  Activation: {args.activation}, Dropout: {args.dropout_p}, Init: {args.init_mode}")

    model = build_model(input_dim, hidden_sizes, num_classes,
                        args.dropout_p, args.init_mode, args.activation,
                        backend=backend)
    print(f"Model:\n{model}")

    # ---- 3. 优化器 & 损失函数 ----
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"\nOptimizer: Adam(lr={args.lr}, weight_decay={args.weight_decay})")
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"\nOptimizer: SGD(lr={args.lr}, weight_decay={args.weight_decay})")
    loss_fn = CrossEntropyLoss()
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Batches per epoch: {train_iter.num_batches}")

    # ---- 4. 记录容器 ----
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    epoch_times = []
    grad_norms = []  # 每个 epoch 结束时的梯度范数

    # ---- 5. 训练循环 ----
    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0

        for images, labels in train_iter:
            batch_size = images.shape[0]

            # 前向传播
            logits = model(images)                # (batch, 10)
            loss = loss_fn(logits, labels)        # scalar

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            loss_val = loss.to_numpy().item()
            total_loss += loss_val * batch_size

            pred_np = np.argmax(logits.to_numpy(), axis=1)
            true_np = np.argmax(labels.to_numpy(), axis=1)
            total_correct += np.sum(pred_np == true_np)
            total_samples += batch_size
            batch_count += 1

            # 日志
            if args.log_interval > 0 and batch_count % args.log_interval == 0:
                batch_acc = np.sum(pred_np == true_np) / batch_size
                print(f"  Epoch {epoch+1}/{args.epochs} | "
                      f"Batch {batch_count}/{train_iter.num_batches} | "
                      f"Loss: {loss_val:.4f} | Acc: {batch_acc:.4f}")

        # ---- Epoch 结束 ----
        epoch_time = time.time() - epoch_start
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # 梯度范数（取 epoch 结束时的参数梯度）
        gnorm = compute_grad_norm(model.parameters())
        grad_norms.append(gnorm)

        # 测试集评估（默认只测 10 个 batch 加速）
        test_loss, test_acc = evaluate(model, test_iter, loss_fn, 
                                        max_batches=args.test_batches)

        # 记录
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)

        print(f"--- Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) ---")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")
        print(f"  Grad Norm:  {gnorm:.4f}")

    # ---- 6. 最终结果 ----
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Final Test Accuracy: {test_accs[-1]:.4f}")
    print(f"  Best  Test Accuracy: {max(test_accs):.4f} (epoch {np.argmax(test_accs)+1})")
    print(f"  Total Time: {sum(epoch_times):.1f}s")

    # ---- 7. 保存日志 ----
    log_data = {
        "config": {
            "device": args.device,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout_p": args.dropout_p,
            "hidden": args.hidden,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "init_mode": args.init_mode,
            "activation": args.activation,
            "optimizer": args.optimizer,
        },
        "results": {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "epoch_times": epoch_times,
            "grad_norms": grad_norms,
            "final_test_acc": test_accs[-1],
            "best_test_acc": max(test_accs),
        }
    }

    if args.save_log:
        log_path = "training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"  Log saved to {log_path}")

    # ---- 8. 可视化 ----
    if args.save_plots:
        plot_training_curves(train_losses, test_losses, train_accs, test_accs, args)

    return log_data


# ============================================================================
# 可视化
# ============================================================================

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, args):
    """绘制 loss 和 accuracy 曲线，并保存图片"""
    epochs = range(1, len(train_losses) + 1)

    # 构建标题信息
    title_info = (f"LR={args.lr}, WD={args.weight_decay}, "
                  f"Drop={args.dropout_p}, Hidden={args.hidden}, "
                  f"Init={args.init_mode}, Act={args.activation}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Loss 曲线 ----
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-s', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves\n{title_info}', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Accuracy 曲线 ----
    ax2.plot(epochs, train_accs, 'b-o', label='Train Acc', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-s', label='Test Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy Curves\n{title_info}', fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plot_path = "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {plot_path}")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
