"""
一键运行张幸心负责模块的补充实验。
覆盖：Fashion-MNIST、初始化对比、Shuffle消融、Batch Size对比、网格搜索。

用法：
  python experiments/run_zhang_experiments.py
  python experiments/run_zhang_experiments.py --epochs_init 10 --epochs_grid 5
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
    """调用主训练脚本，日志自动保存为 training_log.json，再重命名到 results/"""
    cmd = [
        sys.executable, "train_mnist.py",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--save_log", "--save_plots",
        "--log_interval", "0",  # 关闭batch日志，加速
    ] + extra_args
    ok = run_cmd(cmd, desc=f"MLP experiment: {name}")
    if ok and os.path.exists(os.path.join(PROJECT_ROOT, "training_log.json")):
        dst = os.path.join(RESULTS_DIR, f"log_{name}.json")
        shutil.move(os.path.join(PROJECT_ROOT, "training_log.json"), dst)
        print(f"  -> Log moved to {dst}")
        # 移动图片
        src_plot = os.path.join(PROJECT_ROOT, "training_curves.png")
        if os.path.exists(src_plot):
            dst_plot = os.path.join(FIGURES_DIR, f"curves_{name}.png")
            shutil.move(src_plot, dst_plot)
            print(f"  -> Plot moved to {dst_plot}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Run Zhang Xingxin's supplementary experiments")
    parser.add_argument("--epochs_default", type=int, default=10, help="Default epochs for most experiments")
    parser.add_argument("--epochs_grid", type=int, default=5, help="Epochs for grid search")
    parser.add_argument("--skip_fashion", action="store_true", help="Skip Fashion-MNIST experiments")
    parser.add_argument("--skip_init", action="store_true", help="Skip initialization experiments")
    parser.add_argument("--skip_shuffle", action="store_true", help="Skip shuffle ablation")
    parser.add_argument("--skip_batchsize", action="store_true", help="Skip batch size experiments")
    parser.add_argument("--skip_grid", action="store_true", help="Skip grid search")
    parser.add_argument("--only", type=str, default="", help="Only run specific experiments (comma-separated)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    only = set(args.only.split(",")) if args.only else set()

    def should_run(name):
        return not only or name in only

    # ========================================================================
    # Exp-A: Fashion-MNIST Baseline + Regularization
    # ========================================================================
    if should_run("fashion") and not args.skip_fashion:
        print("\n" + "="*60)
        print("[SECTION] Fashion-MNIST Experiments")
        print("="*60)
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he", "--dataset", "fashion"]
        run_mlp_experiment("fashion_baseline", base + ["--weight_decay", "0.0", "--dropout_p", "0.0"], args.epochs_default)
        run_mlp_experiment("fashion_l2", base + ["--weight_decay", "0.001", "--dropout_p", "0.0"], args.epochs_default)
        run_mlp_experiment("fashion_dropout", base + ["--weight_decay", "0.0", "--dropout_p", "0.5"], args.epochs_default)

    # ========================================================================
    # Exp-B: 初始化策略实际训练对比 (He vs Xavier vs Small)
    # ========================================================================
    if should_run("init") and not args.skip_init:
        print("\n" + "="*60)
        print("[SECTION] Initialization Convergence Comparison")
        print("="*60)
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--dataset", "mnist"]
        # He + ReLU
        run_mlp_experiment("init_he", base + ["--init_mode", "he"], args.epochs_default)
        # Xavier + ReLU (虽非最佳搭配，但可作对比)
        run_mlp_experiment("init_xavier", base + ["--init_mode", "xavier"], args.epochs_default)
        # Small random + ReLU
        run_mlp_experiment("init_small", base + ["--init_mode", "small"], args.epochs_default)

    # ========================================================================
    # Exp-C: Shuffle vs No Shuffle
    # ========================================================================
    if should_run("shuffle") and not args.skip_shuffle:
        print("\n" + "="*60)
        print("[SECTION] Shuffle Ablation")
        print("="*60)
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he", "--dataset", "mnist"]
        run_mlp_experiment("shuffle_on", base + ["--shuffle", "1"], args.epochs_default)
        run_mlp_experiment("shuffle_off", base + ["--shuffle", "0"], args.epochs_default)

    # ========================================================================
    # Exp-D: Batch Size 对比
    # ========================================================================
    if should_run("batchsize") and not args.skip_batchsize:
        print("\n" + "="*60)
        print("[SECTION] Batch Size Comparison")
        print("="*60)
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he", "--dataset", "mnist"]
        for bs in [16, 32, 64, 128]:
            run_mlp_experiment(f"batchsize_{bs}", base, args.epochs_default, batch_size=bs)

    # ========================================================================
    # Exp-E: 精简网格搜索 (A/B Testing)
    # ========================================================================
    if should_run("grid") and not args.skip_grid:
        print("\n" + "="*60)
        print("[SECTION] Hyperparameter Grid Search")
        print("="*60)
        configs = [
            ("0.001", "he",    "0.0",  "relu"),
            ("0.001", "he",    "0.3",  "relu"),
            ("0.001", "xavier","0.0",  "sigmoid"),
            ("0.001", "xavier","0.3",  "sigmoid"),
            ("0.01",  "he",    "0.0",  "relu"),
            ("0.01",  "xavier","0.0",  "sigmoid"),
        ]
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
            run_mlp_experiment(f"grid_{idx:02d}_lr{lr}_init{init_mode}_drop{dropout_p}", extra, args.epochs_grid)

    print(f"\n{'='*60}")
    print("ALL ZHANG'S EXPERIMENTS COMPLETE!")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Next step: python experiments/plot_all.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
