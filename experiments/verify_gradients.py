"""
Exp-1: 数值梯度验证 (Gradient Check)
用中心差分法验证手写反向传播的正确性。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from minitorch.nn import Linear, Sigmoid, Sequential, CrossEntropyLoss
from minitorch.optim import SGD
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.tensor_ops import SimpleBackend


def build_toy_model(init_mode="xavier"):
    """构建一个小网络用于梯度验证"""
    model = Sequential(
        Linear(4, 3, bias=True, init_mode=init_mode),
        Sigmoid(),
        Linear(3, 2, bias=True, init_mode=init_mode),
    )
    return model


def numerical_gradient(model, loss_fn, x, y, param_name, eps=1e-5):
    """
    对指定参数计算数值梯度（中心差分）。
    由于我们的 Tensor 系统不方便逐元素修改，这里通过 numpy 直接操作底层数据。
    """
    param = None
    for name, p in model.named_parameters():
        if name == param_name:
            param = p
            break
    if param is None:
        raise ValueError(f"Parameter {param_name} not found")

    w_np = param.value.to_numpy().copy()
    grad = np.zeros_like(w_np)

    it = np.nditer(w_np, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index

        # f(w + eps)
        w_plus = w_np.copy()
        w_plus[idx] += eps
        param.update(tensor_from_numpy(w_plus.astype(np.float32), backend=SimpleBackend))
        logits_plus = model(x)
        loss_plus = loss_fn(logits_plus, y).to_numpy().item()

        # f(w - eps)
        w_minus = w_np.copy()
        w_minus[idx] -= eps
        param.update(tensor_from_numpy(w_minus.astype(np.float32), backend=SimpleBackend))
        logits_minus = model(x)
        loss_minus = loss_fn(logits_minus, y).to_numpy().item()

        # 中心差分
        grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)

        it.iternext()

    # 恢复原值
    param.update(tensor_from_numpy(w_np.astype(np.float32), backend=SimpleBackend))
    return grad


def run_gradient_check(save_dir="experiments/results"):
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(42)

    # 固定输入
    x_np = np.random.randn(2, 4).astype(np.float32)
    y_np = np.zeros((2, 2), dtype=np.float32)
    y_np[0, 1] = 1.0
    y_np[1, 0] = 1.0

    x = tensor_from_numpy(x_np, backend=SimpleBackend)
    y = tensor_from_numpy(y_np, backend=SimpleBackend)

    model = build_toy_model(init_mode="xavier")
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    # 解析梯度
    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    results = []
    for name, param in model.named_parameters():
        analytic_grad = param.value.grad.to_numpy().copy()
        numeric_grad = numerical_gradient(model, loss_fn, x, y, name, eps=1e-5)

        diff = np.abs(analytic_grad - numeric_grad)
        # 避免除以0
        denom = np.abs(numeric_grad)
        denom[denom < 1e-8] = 1e-8
        rel_error = diff / denom

        max_rel_err = float(np.max(rel_error))
        mean_rel_err = float(np.mean(rel_error))

        results.append({
            "param": name,
            "shape": list(analytic_grad.shape),
            "max_rel_error": max_rel_err,
            "mean_rel_error": mean_rel_err,
            "analytic_mean": float(np.mean(analytic_grad)),
            "numeric_mean": float(np.mean(numeric_grad)),
        })

        print(f"[{name}] shape={analytic_grad.shape}, max_rel_error={max_rel_err:.2e}, mean_rel_error={mean_rel_err:.2e}")

    # 保存 JSON
    with open(os.path.join(save_dir, "grad_check.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["param"] for r in results]
    max_errs = [r["max_rel_error"] for r in results]
    mean_errs = [r["mean_rel_error"] for r in results]

    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, max_errs, width, label="Max Relative Error", color="coral")
    ax.bar(x_pos + width/2, mean_errs, width, label="Mean Relative Error", color="steelblue")
    ax.axhline(1e-4, color="red", linestyle="--", linewidth=1, label="Tolerance (1e-4)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Relative Error")
    ax.set_yscale("log")
    ax.set_title("Gradient Check: Analytic vs Numerical Gradient")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    fig_path = os.path.join(os.path.dirname(save_dir), "figures", "gradient_check.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Gradient Check] Figure saved to {fig_path}")


if __name__ == "__main__":
    run_gradient_check()
