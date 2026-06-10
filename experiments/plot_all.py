"""
综合绘图脚本：读取 experiments/results/ 下的所有日志，生成对比图片。
用法：
  python experiments/plot_all.py
  python experiments/plot_all.py --only optimizer,cnn
"""
import sys, os, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "figures")


def load_log(name_pattern):
    """按模式查找日志文件并加载"""
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, name_pattern)))
    logs = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_filename"] = os.path.basename(p)
            logs.append(data)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return logs


def _dedup_by_key(logs, key_fn):
    """按 key 去重，保留第一个"""
    seen = set()
    out = []
    for l in logs:
        k = key_fn(l)
        if k not in seen:
            seen.add(k)
            out.append(l)
    return out


def plot_optimizer_comparison():
    logs = load_log("log_mlp_*.json")
    logs = [l for l in logs if "sgd" in l["_filename"] or "adam" in l["_filename"]]
    logs = _dedup_by_key(logs, lambda l: l["config"].get("optimizer", ""))
    if len(logs) < 2:
        print("[SKIP] optimizer comparison: not enough logs")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"sgd": "#e74c3c", "adam": "#3498db"}

    for log in logs:
        cfg = log["config"]
        res = log["results"]
        epochs = np.arange(1, len(res["train_losses"]) + 1)
        label = cfg.get("optimizer", "?").upper()
        color = colors.get(cfg.get("optimizer", ""), "gray")

        axes[0].plot(epochs, res["train_losses"], color=color, linestyle="-", linewidth=2, label=f"{label} Train")
        axes[0].plot(epochs, res["test_losses"], color=color, linestyle="--", linewidth=2, label=f"{label} Test")
        axes[1].plot(epochs, res["test_accs"], color=color, linewidth=2.5, marker="o", label=label)

        if res.get("grad_norms"):
            gnorms = [g for g in res["grad_norms"] if g > 0]
            if gnorms:
                axes[2].plot(epochs[:len(gnorms)], gnorms, color=color, linewidth=2, label=label)

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.05)

    axes[2].set_title("Gradient Norm (per epoch)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("L2 Norm")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    plt.suptitle("Optimizer Comparison: SGD vs Adam", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(FIGURES_DIR, "optimizer_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {save_path}")


def plot_regularization_comparison():
    logs = load_log("log_reg_*.json")
    logs = _dedup_by_key(logs, lambda l: (l["config"].get("weight_decay", 0), l["config"].get("dropout_p", 0)))
    if len(logs) < 2:
        print("[SKIP] regularization comparison: not enough logs")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    name_map = {
        "log_reg_none.json": "None",
        "log_reg_l2.json": "L2",
        "log_reg_dropout.json": "Dropout",
        "log_reg_l2dropout.json": "L2+Dropout",
    }
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for log, color in zip(logs, colors):
        fname = log["_filename"]
        label = name_map.get(fname, fname)
        res = log["results"]
        epochs = np.arange(1, len(res["train_losses"]) + 1)

        axes[0, 0].plot(epochs, res["train_losses"], color=color, label=label)
        axes[0, 1].plot(epochs, res["test_losses"], color=color, label=label)
        axes[1, 0].plot(epochs, res["train_accs"], color=color, linestyle="-", label=f"{label} Train")
        axes[1, 0].plot(epochs, res["test_accs"], color=color, linestyle="--", label=f"{label} Test")
        # 泛化差距
        gap = np.array(res["train_accs"]) - np.array(res["test_accs"])
        axes[1, 1].plot(epochs, gap, color=color, label=label, linewidth=2)

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch")
    axes[0, 0].set_title("Train Loss"); axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_title("Test Loss"); axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_title("Accuracy"); axes[1, 0].set_ylabel("Accuracy")
    axes[1, 1].set_title("Generalization Gap (Train - Test)"); axes[1, 1].set_ylabel("Gap")
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)

    plt.suptitle("Regularization Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(FIGURES_DIR, "regularization_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {save_path}")


def plot_activation_comparison():
    logs = load_log("log_act_*.json")
    logs = _dedup_by_key(logs, lambda l: l["config"].get("activation", ""))
    if len(logs) < 2:
        print("[SKIP] activation comparison: not enough logs")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(logs)))

    for log, color in zip(logs, colors):
        act = log["config"].get("activation", "?")
        res = log["results"]
        epochs = np.arange(1, len(res["train_losses"]) + 1)
        ax1.plot(epochs, res["test_losses"], color=color, label=act, linewidth=2)
        ax2.plot(epochs, res["test_accs"], color=color, label=act, linewidth=2, marker="o", markersize=4)

    ax1.set_title("Test Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.05)

    plt.suptitle("Activation Function Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(FIGURES_DIR, "activation_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {save_path}")


def plot_cnn_vs_mlp():
    mlp_logs = load_log("log_mlp_baseline.json")
    cnn_logs = load_log("log_cnn.json")
    if not mlp_logs or not cnn_logs:
        print("[SKIP] cnn vs mlp: need both log_mlp_baseline.json and log_cnn.json")
        return

    mlp = mlp_logs[0]
    cnn = cnn_logs[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 最终准确率对比
    models = ["MLP", "CNN"]
    accs = [mlp["results"]["final_test_acc"], cnn["results"]["final_test_acc"]]
    bars = axes[0].bar(models, accs, color=["#3498db", "#e74c3c"], width=0.5)
    axes[0].set_ylim(0.8, 1.0)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Final Test Accuracy")
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{acc:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # 2. 训练曲线对比
    for log, name, color in [(mlp, "MLP", "#3498db"), (cnn, "CNN", "#e74c3c")]:
        res = log["results"]
        epochs = np.arange(1, len(res["test_accs"]) + 1)
        axes[1].plot(epochs, res["test_accs"], color=color, label=name, linewidth=2.5, marker="o")
    axes[1].set_title("Test Accuracy Curves")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.05)

    # 3. 每 epoch 耗时对比
    for log, name, color in [(mlp, "MLP", "#3498db"), (cnn, "CNN", "#e74c3c")]:
        res = log["results"]
        epochs = np.arange(1, len(res["epoch_times"]) + 1)
        axes[2].plot(epochs, res["epoch_times"], color=color, label=name, linewidth=2, marker="s")
    axes[2].set_title("Epoch Time")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Time (s)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle("CNN vs MLP Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(FIGURES_DIR, "cnn_vs_mlp.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {save_path}")

    # 混淆矩阵（如果存在）
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.npy")
    if os.path.exists(cm_path):
        cm = np.load(cm_path)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("CNN Confusion Matrix")
        ax.set_xticks(np.arange(10)); ax.set_yticks(np.arange(10))
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="darkred", fontsize=8)
        plt.tight_layout()
        save_path_cm = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        plt.savefig(save_path_cm, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] {save_path_cm}")


def plot_hyperparameter_heatmap():
    logs = load_log("log_grid_*.json")
    if len(logs) < 4:
        print("[SKIP] hyperparameter heatmap: not enough grid logs")
        return

    # 解析参数
    records = []
    for log in logs:
        cfg = log["config"]
        acc = log["results"]["final_test_acc"]
        records.append({
            "lr": cfg.get("lr"),
            "init_mode": cfg.get("init_mode"),
            "dropout_p": cfg.get("dropout_p"),
            "acc": acc,
        })

    # 分 Dropout 两组
    for dropout_val, title_suffix in [(0.0, "No Dropout"), (0.3, "Dropout=0.3")]:
        subset = [r for r in records if abs(r["dropout_p"] - dropout_val) < 1e-6]
        if len(subset) < 4:
            continue

        lrs = sorted(list(set(r["lr"] for r in subset)))
        inits = sorted(list(set(r["init_mode"] for r in subset)))
        mat = np.zeros((len(inits), len(lrs)))
        for r in subset:
            i = inits.index(r["init_mode"])
            j = lrs.index(r["lr"])
            mat[i, j] = r["acc"]

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0.8, vmax=1.0)
        fig.colorbar(im, ax=ax, label="Test Accuracy")
        ax.set_xticks(np.arange(len(lrs)))
        ax.set_yticks(np.arange(len(inits)))
        ax.set_xticklabels([str(l) for l in lrs])
        ax.set_yticklabels(inits)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Init Mode")
        ax.set_title(f"Hyperparameter Heatmap ({title_suffix})")

        for i in range(len(inits)):
            for j in range(len(lrs)):
                text = ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                               color="black" if mat[i, j] < 0.9 else "white", fontsize=10)
        plt.tight_layout()
        fname = f"hyperparameter_heatmap_dropout{int(dropout_val*10)}.png"
        save_path = os.path.join(FIGURES_DIR, fname)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated list: optimizer,regularization,activation,cnn,heatmap")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    only = set(args.only.split(",")) if args.only else set()

    def should_plot(name):
        return not only or name in only

    if should_plot("optimizer"):
        plot_optimizer_comparison()
    if should_plot("regularization"):
        plot_regularization_comparison()
    if should_plot("activation"):
        plot_activation_comparison()
    if should_plot("cnn"):
        plot_cnn_vs_mlp()
    if should_plot("heatmap"):
        plot_hyperparameter_heatmap()

    print(f"\n[Done] All requested plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
