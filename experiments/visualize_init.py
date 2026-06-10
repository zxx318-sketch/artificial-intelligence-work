"""
Exp-2: 初始化策略可视化
对比 He / Xavier / Normal(small) / Normal(large) 初始化下，
深层网络各层激活值的分布稳定性。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from minitorch.nn import Linear, ReLU, Sigmoid, Sequential
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.tensor_ops import SimpleBackend
from minitorch.MNISTIterator import load_mnist


def build_deep_mlp(hidden_size=100, num_layers=5, init_mode="he", activation="relu"):
    """构建深层 MLP，用于观察信号传播"""
    act_cls = ReLU if activation == "relu" else Sigmoid
    layers = []
    in_size = 784
    for _ in range(num_layers):
        layers.append(Linear(in_size, hidden_size, bias=True, init_mode=init_mode))
        layers.append(act_cls())
        in_size = hidden_size
    return Sequential(*layers)


def collect_activations(model, images_np):
    """
    逐层前向传播，记录每层的激活值（返回 numpy 列表）。
    注意：由于 Sequential 一次性走到底，我们手动拆分层来记录。
    """
    x = tensor_from_numpy(images_np, backend=SimpleBackend)
    activations = []
    # 手动遍历子模块
    for module in model.modules_list:
        x = module(x)
        # 只记录 Linear 输出后的激活（即经过激活函数后的值）
        if isinstance(module, (ReLU, Sigmoid)):
            activations.append(x.to_numpy().flatten())
    return activations


def run_init_visualization(save_dir="experiments/results"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(save_dir), "figures"), exist_ok=True)
    np.random.seed(42)

    # 加载少量 MNIST 数据
    train_images, _ = load_mnist("./data", kind="train")
    sample_images = train_images[:256].reshape(256, 784).astype(np.float32) / 255.0

    configs = [
        {"name": "He + ReLU", "init_mode": "he", "activation": "relu"},
        {"name": "Xavier + Sigmoid", "init_mode": "xavier", "activation": "sigmoid"},
        {"name": "Normal(0.01)", "init_mode": "he", "activation": "relu", "scale": 0.01},   # 手动缩放
        {"name": "Normal(0.1)", "init_mode": "he", "activation": "relu", "scale": 0.1},     # 手动缩放
    ]

    all_stats = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, cfg in zip(axes, configs):
        model = build_deep_mlp(hidden_size=100, num_layers=5,
                               init_mode=cfg["init_mode"], activation=cfg["activation"])

        # 如果是手动缩放，覆盖权重
        scale = cfg.get("scale")
        if scale is not None:
            for name, param in model.named_parameters():
                if "weight" in name:
                    w = param.value.to_numpy()
                    w[:] = np.random.randn(*w.shape).astype(np.float32) * scale
                    param.update(tensor_from_numpy(w, backend=SimpleBackend))

        acts = collect_activations(model, sample_images)

        # 统计
        stats = []
        for i, a in enumerate(acts):
            stats.append({
                "layer": i + 1,
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
            })
        all_stats.append({"config": cfg["name"], "stats": stats})

        # 箱线图
        bp = ax.boxplot(acts, labels=[f"L{i+1}" for i in range(len(acts))],
                        patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_title(cfg["name"])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Activation Value")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Activation Distribution across Layers (Different Initializations)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = os.path.join(os.path.dirname(save_dir), "figures", "init_distributions.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Init Viz] Figure saved to {fig_path}")

    # 保存统计 JSON
    with open(os.path.join(save_dir, "init_stats.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    # 额外绘制：各层方差随深度变化
    fig, ax = plt.subplots(figsize=(8, 5))
    for entry in all_stats:
        layers = [s["layer"] for s in entry["stats"]]
        stds = [s["std"] for s in entry["stats"]]
        ax.plot(layers, stds, marker='o', label=entry["config"], linewidth=2)
    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("Activation Std Dev")
    ax.set_title("Signal Propagation: Std Dev vs Layer Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path2 = os.path.join(os.path.dirname(save_dir), "figures", "init_std_depth.png")
    plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Init Viz] Std-dev figure saved to {fig_path2}")


if __name__ == "__main__":
    run_init_visualization()
