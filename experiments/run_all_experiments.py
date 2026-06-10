"""
一键运行除 im2col 外的全部实验。
用法：
  python experiments/run_all_experiments.py --epochs 10 --quick
  python experiments/run_all_experiments.py --epochs 20           # 完整版
"""
import sys, os, subprocess, json, argparse, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")


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
    ] + extra_args
    ok = run_cmd(cmd, desc=f"MLP experiment: {name}")
    if ok and os.path.exists("training_log.json"):
        dst = os.path.join(RESULTS_DIR, f"log_{name}.json")
        os.replace("training_log.json", dst)
        print(f"  -> Log moved to {dst}")
    return ok


def run_numerical_stability():
    """Exp-6: Softmax 数值稳定性验证"""
    print(f"\n{'='*60}")
    print("[RUN] Numerical Stability Test (Softmax)")
    print(f"{'='*60}")

    import numpy as np
    # 极端 logits
    logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    labels = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 分开计算（softmax -> log -> nll）
    exp_logits = np.exp(logits)
    probs_sep = exp_logits / np.sum(exp_logits)
    log_probs_sep = np.log(probs_sep)
    loss_sep = -np.sum(labels * log_probs_sep)

    # 合并计算（log-softmax 数值稳定版）
    max_logit = np.max(logits)
    shifted = logits - max_logit
    log_probs_fused = shifted - np.log(np.sum(np.exp(shifted)))
    loss_fused = -np.sum(labels * log_probs_fused)

    # 正常 logits
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
    parser = argparse.ArgumentParser(description="Run all AB experiments")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for training experiments")
    parser.add_argument("--quick", action="store_true", help="Skip heavy grid search, run minimal set")
    parser.add_argument("--only", type=str, default="", help="Only run specific experiments (comma-separated)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    only = set(args.only.split(",")) if args.only else set()

    def should_run(name):
        return not only or name in only

    # ========================================================================
    # Exp-1: 梯度验证
    # ========================================================================
    if should_run("gradient"):
        run_cmd([sys.executable, "experiments/verify_gradients.py"], "Exp-1 Gradient Check")

    # ========================================================================
    # Exp-2: 初始化可视化
    # ========================================================================
    if should_run("init"):
        run_cmd([sys.executable, "experiments/visualize_init.py"], "Exp-2 Initialization Visualization")

    # ========================================================================
    # Exp-3: 优化器对比
    # ========================================================================
    if should_run("optimizer"):
        run_mlp_experiment("mlp_sgd", ["--optimizer", "sgd", "--lr", "0.1", "--activation", "relu", "--init_mode", "he"], args.epochs)
        run_mlp_experiment("mlp_adam", ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he"], args.epochs)

    # ========================================================================
    # Exp-4: 正则化对比
    # ========================================================================
    if should_run("regularization"):
        base = ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he"]
        run_mlp_experiment("reg_none", base + ["--weight_decay", "0.0", "--dropout_p", "0.0"], args.epochs)
        run_mlp_experiment("reg_l2", base + ["--weight_decay", "0.001", "--dropout_p", "0.0"], args.epochs)
        run_mlp_experiment("reg_dropout", base + ["--weight_decay", "0.0", "--dropout_p", "0.5"], args.epochs)
        run_mlp_experiment("reg_l2dropout", base + ["--weight_decay", "0.001", "--dropout_p", "0.5"], args.epochs)

    # ========================================================================
    # Exp-5: 激活函数对比
    # ========================================================================
    if should_run("activation"):
        run_mlp_experiment("act_relu", ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he"], args.epochs)
        run_mlp_experiment("act_sigmoid", ["--optimizer", "adam", "--lr", "0.001", "--activation", "sigmoid", "--init_mode", "xavier"], args.epochs)
        run_mlp_experiment("act_tanh", ["--optimizer", "adam", "--lr", "0.001", "--activation", "tanh", "--init_mode", "xavier"], args.epochs)
        run_mlp_experiment("act_leakyrelu", ["--optimizer", "adam", "--lr", "0.001", "--activation", "leaky_relu", "--init_mode", "he"], args.epochs)

    # ========================================================================
    # Exp-6: 数值稳定性
    # ========================================================================
    if should_run("stability"):
        run_numerical_stability()

    # ========================================================================
    # Exp-7: CNN vs MLP
    # ========================================================================
    if should_run("cnn"):
        # 先跑 MLP baseline（与激活函数实验共用参数即可）
        run_mlp_experiment("mlp_baseline", ["--optimizer", "adam", "--lr", "0.001", "--activation", "relu", "--init_mode", "he", "--hidden", "128,64"], args.epochs)
        # 跑 CNN
        run_cmd([sys.executable, "experiments/cnn_train.py",
                 "--epochs", str(args.epochs), "--lr", "0.001"], "Exp-7 CNN Training")

    # ========================================================================
    # Exp-8: 网格搜索（超参数 A/B Testing）
    # ========================================================================
    if should_run("grid") and not args.quick:
        print(f"\n{'='*60}")
        print("[RUN] Exp-8 Hyperparameter Grid Search")
        print(f"{'='*60}")
        grid_id = 0
        for lr in [0.1, 0.01, 0.001]:
            for init_mode in ["he", "xavier"]:
                for dropout_p in [0.0, 0.3]:
                    # 简单跳过明显不兼容的组合
                    activation = "relu" if init_mode == "he" else "sigmoid"
                    extra = [
                        "--optimizer", "adam",
                        "--lr", str(lr),
                        "--activation", activation,
                        "--init_mode", init_mode,
                        "--dropout_p", str(dropout_p),
                        "--weight_decay", "0.0",
                    ]
                    run_mlp_experiment(f"grid_{grid_id:02d}_lr{lr}_init{init_mode}_drop{dropout_p}", extra, max(args.epochs, 5))
                    grid_id += 1
    elif should_run("grid") and args.quick:
        print("[SKIP] Grid search skipped in quick mode. Use --epochs 5 without --quick to run.")

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Next step: python experiments/plot_all.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
