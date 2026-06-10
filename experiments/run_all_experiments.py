"""
一键运行全部实验（原 run_all_experiments + run_zhang_experiments 合并版）。

用法：
  # 运行全部实验（完整版，较慢）
  python experiments/run_all_experiments.py --epochs 10

  # 快速模式（跳过完整网格搜索，Batch Size 只跑关键值）
  python experiments/run_all_experiments.py --epochs 5 --quick

  # 只运行张幸心负责的实验
  python experiments/run_all_experiments.py --only fashion,init,shuffle,batchsize

  # 跳过 CNN（CPU 上极慢）
  python experiments/run_all_experiments.py --skip_cnn
"""
import sys, os, subprocess, json, argparse, time, shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "experiments", "figures")


def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"[RUN] {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    print(f"[DONE] {desc} in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode == 0


def run_mlp_experiment(name, extra_args, epochs, batch_size=64):
    """调用主训练脚本，日志自动保存并归档到 results/ 和 figures/"""
    cmd = [
        sys.executable, "train_mnist.py",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--save_log", "--save_plots",
        "--log_interval", "0",  # 关闭 batch 日志，加速
    ] + extra_args
    ok = run_cmd(cmd, desc=f"MLP experiment: {name}")
    log_src = os.path.join(PROJECT_ROOT, "training_log.json")
    plot_src = os.path.join(PROJECT_ROOT, "training_curves.png")
    if ok and os.path.exists(log_src):
        dst = os.path.join(RESULTS_DIR, f"log_{name}.json")
        shutil.move(log_src, dst)
        print(f"  -> Log moved to {dst}")
    if ok and os.path.exists(plot_src):
        dst_plot = os.path.join(FIGURES_DIR, f"curves_{name}.png")
        shutil.move(plot_src, dst_plot)
        print(f"  -> Plot moved to {dst_plot}")
    return ok


def run_numerical_stability():
    """Exp-6: Softmax 数值稳定性验证"""
    print(f"\n{'='*60}")
    print("[RUN] Numerical Stability Test (Softmax)")
    print(f"{'='*60}")

    import numpy as np
    logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    labels = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    exp_logits = np.exp(logits)
    probs_sep = exp_logits / np.sum(exp_logits)
    log_probs_sep = np.log(probs_sep)
    loss_sep = -np.sum(labels * log_probs_sep)

    max_logit = np.max(logits)
    shifted = logits - max_logit
    log_probs_fused = shifted - np.log(np.sum(np.exp(shifted)))
    loss_fused = -np.sum(labels * log_probs_fused)

    logits_normal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    exp_n = np.exp(logits_normal)
    probs_n_sep = exp_n / np.sum(exp_n)
    loss_n_sep = -np.sum(labels * np.log(probs_n_sep))
    max_n = np.max(logits_normal)
    shifted_n = logits_normal - max_n
    loss_n_fused = -np.sum(labels * (shifted_n - np.log(np.sum(np.exp(shifted_n)))))

    data = {
        "extreme_logits": logits.tolist(),
        "separate_loss": float(loss_sep),
        "fused_loss": float(loss_fused),
        "separate_has_inf": not np.isfinite(loss_sep),
        "fused_has_inf": not np.isfinite(loss_fused),
        "normal_logits": logits_normal.tolist(),
        "normal_separate_loss": float(loss_n_sep),
        "normal_fused_loss": float(loss_n_fused),
        "conclusion": "分开计算在极端值下会 overflow，合并 log-softmax 保持稳定。",
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "numerical_stability.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved to {path}")
    print(f"     Extreme: separate={loss_sep}, fused={loss_fused}")
    print(f"     Normal:  separate={loss_n_sep}, fused={loss_n_fused}")


def main():
    parser = argparse.ArgumentParser(description="Run all experiments (merged)")
    parser.add_argument("--epochs", type=int, default=10, help="Default epochs for training experiments")
    parser.add_argument("--epochs_grid", type=int, default=5, help="Epochs for grid search")
    parser.add_argument("--quick", action="store_true", help="Skip heavy experiments, run minimal set")
    parser.add_argument("--only", type=str, default="", help="Only run specific experiments (comma-separated)")

    # Skip flags for fine-grained control
    parser.add_argument("--skip_gradient", action="store_true", help="Skip gradient check")
    parser.add_argument("--skip_init_viz", action="store_true", help="Skip initialization visualization")
    parser.add_argument("--skip_init_train", action="store_true", help="Skip initialization training comparison")
    parser.add_argument("--skip_optimizer", action="store_true", help="Skip optimizer comparison")
    parser.add_argument("--skip_regularization", action="store_true", help="Skip regularization comparison")
    parser.add_argument("--skip_activation", action="store_true", help="Skip activation comparison")
    parser.add_argument("--skip_shuffle", action="store_true", help="Skip shuffle ablation")
    parser.add_argument("--skip_batchsize", action="store_true", help="Skip batch size comparison")
    parser.add_argument("--skip_stability", action="store_true", help="Skip numerical stability test")
    parser.add_argument("--skip_cnn", action="store_true", help="Skip CNN training")
    parser.add_argument("--skip_fashion", action="store_true", help="Skip Fashion-MNIST experiments")
    parser.add_argument("--skip_grid", action="store_true", help="Skip grid search")

    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    only = set(args.only.split(",")) if args.only else set()

    def should_run(name):
        return not only or name in only

    # ========================================================================
    # Exp-1: 梯度验证
    # ========================================================================
    if should_run("gradient") and not args.skip_gradient:
        run_cmd([sys.executable, "experiments/verify_gradients.py"], "Exp-1 Gradient Check")

    # ========================================================================
    # Exp-2: 初始化可视化（静态分布）
    # ========================================================================
    if should_run("init_viz") and not args.skip_init_viz:
        run_cmd([sys.executable, "experiments/visualize_init.py"], "Exp-2 Initialization Visualization")

    # ========================================================================
    # Exp-3: 优化器对比
    # ========================================================================
    if should_run("optimizer") and not args.skip_optimizer:
        print("\n" + "="*60)
        print("[SECTION] Optimizer Comparison")
        print("="*60)
        run_mlp_experiment("mlp_sgd", ["--optimizer", "sgd", "--lr", "0.1", "--activation", "relu", "--init_mode", "he"], args.epochs)
        run_mlp_experiment("mlp_adam", ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he"], args.epochs)

    # ========================================================================
    # Exp-4: 正则化对比
    # ========================================================================
    if should_run("regularization") and not args.skip_regularization:
        print("\n" + "="*60)
        print("[SECTION] Regularization Comparison")
        print("="*60)
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he"]
        run_mlp_experiment("reg_none", base + ["--weight_decay", "0.0", "--dropout_p", "0.0"], args.epochs)
        run_mlp_experiment("reg_l2", base + ["--weight_decay", "0.001", "--dropout_p", "0.0"], args.epochs)
        run_mlp_experiment("reg_dropout", base + ["--weight_decay", "0.0", "--dropout_p", "0.5"], args.epochs)
        run_mlp_experiment("reg_l2dropout", base + ["--weight_decay", "0.001", "--dropout_p", "0.5"], args.epochs)

    # ========================================================================
    # Exp-5: 激活函数对比
    # ========================================================================
    if should_run("activation") and not args.skip_activation:
        print("\n" + "="*60)
        print("[SECTION] Activation Function Comparison")
        print("="*60)
        run_mlp_experiment("act_relu", ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he"], args.epochs)
        run_mlp_experiment("act_sigmoid", ["--optimizer", "adam", "--lr", "0.001", "--activation", "sigmoid", "--init_mode", "xavier"], args.epochs)
        run_mlp_experiment("act_tanh", ["--optimizer", "adam", "--lr", "0.001", "--activation", "tanh", "--init_mode", "xavier"], args.epochs)
        run_mlp_experiment("act_leakyrelu", ["--optimizer", "adam", "--lr", "0.001", "--activation", "leaky_relu", "--init_mode", "he"], args.epochs)

    # ========================================================================
    # Exp-6: 数值稳定性
    # ========================================================================
    if should_run("stability") and not args.skip_stability:
        run_numerical_stability()

    # ========================================================================
    # Exp-7: 初始化训练对比 (张幸心)
    # ========================================================================
    if should_run("init_train") and not args.skip_init_train:
        print("\n" + "="*60)
        print("[SECTION] Initialization Convergence Comparison")
        print("="*60)
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--dataset", "mnist"]
        run_mlp_experiment("init_he", base + ["--init_mode", "he"], args.epochs)
        run_mlp_experiment("init_xavier", base + ["--init_mode", "xavier"], args.epochs)
        run_mlp_experiment("init_small", base + ["--init_mode", "small"], args.epochs)

    # ========================================================================
    # Exp-8: Shuffle 消融 (张幸心)
    # ========================================================================
    if should_run("shuffle") and not args.skip_shuffle:
        print("\n" + "="*60)
        print("[SECTION] Shuffle Ablation")
        print("="*60)
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he", "--dataset", "mnist"]
        run_mlp_experiment("shuffle_on", base + ["--shuffle", "1"], args.epochs)
        run_mlp_experiment("shuffle_off", base + ["--shuffle", "0"], args.epochs)

    # ========================================================================
    # Exp-9: Batch Size 对比 (张幸心)
    # ========================================================================
    if should_run("batchsize") and not args.skip_batchsize:
        print("\n" + "="*60)
        print("[SECTION] Batch Size Comparison")
        print("="*60)
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he", "--dataset", "mnist"]
        batch_sizes = [32, 64, 128] if args.quick else [16, 32, 64, 128]
        for bs in batch_sizes:
            run_mlp_experiment(f"batchsize_{bs}", base, args.epochs, batch_size=bs)

    # ========================================================================
    # Exp-10: CNN vs MLP
    # ========================================================================
    if should_run("cnn") and not args.skip_cnn:
        print("\n" + "="*60)
        print("[SECTION] CNN Training")
        print("="*60)
        run_mlp_experiment("mlp_baseline", ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he", "--hidden", "128,64"], args.epochs)
        run_cmd([sys.executable, "experiments/cnn_train.py",
                 "--epochs", str(args.epochs), "--lr", "0.001"], "Exp-10 CNN Training")

    # ========================================================================
    # Exp-11: Fashion-MNIST (张幸心)
    # ========================================================================
    if should_run("fashion") and not args.skip_fashion:
        print("\n" + "="*60)
        print("[SECTION] Fashion-MNIST Experiments")
        print("="*60)
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he", "--dataset", "fashion"]
        run_mlp_experiment("fashion_baseline", base + ["--weight_decay", "0.0", "--dropout_p", "0.0"], args.epochs)
        run_mlp_experiment("fashion_l2", base + ["--weight_decay", "0.001", "--dropout_p", "0.0"], args.epochs)
        run_mlp_experiment("fashion_dropout", base + ["--weight_decay", "0.0", "--dropout_p", "0.5"], args.epochs)

    # ========================================================================
    # Exp-12: 网格搜索（超参数 A/B Testing）
    # ========================================================================
    if should_run("grid") and not args.skip_grid:
        print("\n" + "="*60)
        print("[SECTION] Hyperparameter Grid Search")
        print("="*60)
        if args.quick:
            # 快速模式：6 组精简配置
            configs = [
                ("0.001", "he",    "0.0",  "relu"),
                ("0.001", "he",    "0.3",  "relu"),
                ("0.001", "xavier","0.0",  "sigmoid"),
                ("0.001", "xavier","0.3",  "sigmoid"),
                ("0.01",  "he",    "0.0",  "relu"),
                ("0.01",  "xavier","0.0",  "sigmoid"),
            ]
        else:
            # 完整模式：12 组全配置
            configs = []
            for lr in ["0.1", "0.01", "0.001"]:
                for init_mode in ["he", "xavier"]:
                    for dropout_p in ["0.0", "0.3"]:
                        activation = "relu" if init_mode == "he" else "sigmoid"
                        configs.append((lr, init_mode, dropout_p, activation))

        for idx, (lr, init_mode, dropout_p, activation) in enumerate(configs):
            extra = [
                "--optimizer", "adam",
                "--lr", lr,
                "--activation", activation,
                "--init_mode", init_mode,
                "--dropout_p", dropout_p,
                "--weight_decay", "0.0",
                "--dataset", "mnist",
            ]
            run_mlp_experiment(f"grid_{idx:02d}_lr{lr}_init{init_mode}_drop{dropout_p}", extra, max(args.epochs_grid, 3))

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Next step: python experiments/plot_all.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
