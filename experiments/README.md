# 实验方案与数据展示指南

本目录包含完整的一键实验运行脚本和绘图脚本，覆盖开题报告中除 im2col 外的所有研究要求。

---

## 快速开始

```bash
# 1. 运行所有实验（会根据你的硬件自动选择 epoch 数，CPU 建议用少量 epoch）
python experiments/run_all_experiments.py --epochs 10 --quick

# 2. 生成所有图片
python experiments/plot_all.py
```

实验日志会自动保存到 `experiments/results/`，图片保存到 `experiments/figures/`。

---

## 实验清单与对应研究要求

| 实验编号 | 研究要求 | 实验内容 | 记录数据 | 生成图片 |
|:---|:---|:---|:---|:---|
| **Exp-1** | ② 反向传播链式求导 | 数值梯度验证（Finite Difference） | `results/grad_check.json` | `figures/gradient_check.png` |
| **Exp-2** | ⑤ He & Xavier 初始化 | 不同初始化下各层激活值分布 | `results/init_stats.json` | `figures/init_distributions.png` |
| **Exp-3** | ③ 优化器（SGD vs Adam） | 相同网络下两种优化器收敛对比 | `results/log_*.json` | `figures/optimizer_comparison.png` |
| **Exp-4** | ⑥ 正则化（L2 / Dropout） | 四组正则化策略对比 | `results/log_*.json` | `figures/regularization_comparison.png` |
| **Exp-5** | ① 激活函数 | ReLU/Sigmoid/Tanh/LeakyReLU 对比 | `results/log_*.json` | `figures/activation_comparison.png` |
| **Exp-6** | ④ Softmax 数值稳定性 | 极端 logits 下数值溢出测试 | `results/numerical_stability.json` | 报告中直接放表格 |
| **Exp-7** | 加分项1 CNN | MLP vs CNN 准确率与混淆矩阵 | `results/log_cnn.json` | `figures/cnn_vs_mlp.png` + `figures/confusion_matrix.png` |
| **Exp-8** | 加分项3 A/B Testing | 学习率 × 初始化网格搜索 | `results/log_grid_*.json` | `figures/hyperparameter_heatmap.png` |

---

## 各实验详细说明

### Exp-1 梯度数值验证
- **目的**：证明手写的反向传播与链式求导公式正确。
- **方法**：构建小型网络（输入4维 → 隐藏3维 → 输出2维），对每一组参数用中心差分计算数值梯度，与 `backward()` 得到的解析梯度对比。
- **关键指标**：相对误差 `| analytic - numeric | / |numeric|` 应 `< 1e-4`。
- **报告展示**：柱状图展示各参数相对误差 + 一张汇总表格。

### Exp-2 初始化分布可视化
- **目的**：验证 He / Xavier 初始化能稳定各层激活值方差。
- **方法**：构建 5 层 MLP（每层 100 神经元），分别用 He、Xavier、Normal(0.01)、Normal(0.1) 初始化；输入 256 张 MNIST 图像，记录每层激活值。
- **关键指标**：每层激活值的均值、方差、最小/最大值。
- **报告展示**：
  - 箱线图 / 小提琴图：对比四种初始化在各层的输出分布
  - 折线图：各层激活值方差随深度的变化趋势

### Exp-3 优化器对比
- **命令示例**：
  ```bash
  python train_mnist.py --optimizer sgd --lr 0.1 --epochs 20 --activation relu --init_mode he --hidden "128,64" --save_log
  python train_mnist.py --optimizer adam --lr 0.001 --epochs 20 --activation relu --init_mode he --hidden "128,64" --save_log
  ```
- **记录数据**：`train_losses`, `test_losses`, `train_accs`, `test_accs`, `grad_norms`, `epoch_times`
- **报告展示**：
  - 左图：Loss 曲线（Train/Test 双线）
  - 右图：Accuracy 曲线
  - 子图：梯度范数变化（验证梯度是否稳定）
  - 表格：达到 95%/98% 测试准确率所需的 epoch、最终准确率、总耗时

### Exp-4 正则化效果对比
- **命令示例**：
  ```bash
  python train_mnist.py --optimizer adam --lr 0.001 --epochs 20 --activation relu --init_mode he --weight_decay 0   --dropout_p 0
  python train_mnist.py --optimizer adam --lr 0.001 --epochs 20 --activation relu --init_mode he --weight_decay 0.001 --dropout_p 0
  python train_mnist.py --optimizer adam --lr 0.001 --epochs 20 --activation relu --init_mode he --weight_decay 0   --dropout_p 0.5
  python train_mnist.py --optimizer adam --lr 0.001 --epochs 20 --activation relu --init_mode he --weight_decay 0.001 --dropout_p 0.5
  ```
- **关键指标**：
  - 泛化差距 = Train Acc - Test Acc（差距越小越好）
  - 最终测试准确率
- **报告展示**：
  - 四组 Train/Test Loss 曲线（2×2 子图）
  - 泛化差距折线图
  - 对比表格

### Exp-5 激活函数对比
- **命令示例**：
  ```bash
  python train_mnist.py --activation relu       --init_mode he     --optimizer adam --lr 0.001 --epochs 20
  python train_mnist.py --activation sigmoid    --init_mode xavier --optimizer adam --lr 0.001 --epochs 20
  python train_mnist.py --activation tanh       --init_mode xavier --optimizer adam --lr 0.001 --epochs 20
  python train_mnist.py --activation leaky_relu --init_mode he     --optimizer adam --lr 0.001 --epochs 20
  ```
- **报告展示**：四种激活函数的 Loss/Accuracy 曲线（四线同图），便于直接对比收敛速度。

### Exp-6 Softmax 数值稳定性
- **方法**：不跑完整训练，直接构造极端 logits（如 `[1000, 1001, 1002]`），对比：
  - 分开计算：`softmax` → `log` → `nll_loss`
  - 合并计算：`log_softmax`（减最大值）→ `nll_loss`
- **报告展示**：直接放一张三列表格（输入、分开计算结果、合并计算结果），说明分开计算会 `inf/overflow`。

### Exp-7 CNN vs MLP
- **命令示例**：
  ```bash
  # MLP（已存在于主脚本）
  python train_mnist.py --hidden "128,64" --epochs 10 --optimizer adam --lr 0.001
  
  # CNN（使用 experiments/cnn_train.py）
  python experiments/cnn_train.py --epochs 10 --lr 0.001
  ```
- **记录数据**：最终测试准确率、训练总时间、每 epoch 时间、参数量。
- **报告展示**：
  - 柱状图：MLP vs CNN 的测试准确率、参数量、训练耗时
  - 混淆矩阵热力图（10×10，针对 CNN 结果）
  - 训练曲线对比

### Exp-8 超参数 A/B Testing（网格搜索）
- **搜索空间**（建议快速版，CPU 友好）：
  - 学习率：`[0.1, 0.01, 0.001]`
  - 初始化：`["he", "xavier"]`
  - Dropout：`[0, 0.3]`
  共 12 组实验。
- **报告展示**：
  - 热力图：横轴学习率，纵轴初始化，颜色深浅 = 最终测试准确率（分 Dropout=0 和 Dropout=0.3 两张）
  - 汇总表格：Top-5 最佳组合的详细指标

---

## 日志文件命名规范

运行 `run_all_experiments.py` 后会自动按以下命名保存：

```
experiments/results/
  grad_check.json              # Exp-1
  init_stats.json              # Exp-2
  log_mlp_sgd.json             # Exp-3
  log_mlp_adam.json
  log_reg_none.json            # Exp-4
  log_reg_l2.json
  log_reg_dropout.json
  log_reg_l2dropout.json
  log_act_relu.json            # Exp-5
  log_act_sigmoid.json
  log_act_tanh.json
  log_act_leakyrelu.json
  numerical_stability.json     # Exp-6
  log_mlp_baseline.json        # Exp-7 MLP
  log_cnn.json                 # Exp-7 CNN
  log_grid_*.json              # Exp-8
```

---

## 绘图脚本使用说明

```bash
# 绘制全部
python experiments/plot_all.py --results_dir experiments/results --output_dir experiments/figures

# 只绘制指定实验
python experiments/plot_all.py --only optimizer,regularization
```

---

## 注意事项

1. **CPU 训练较慢**：若用 CPU，建议 `--epochs 5~10` 做快速对比，关键实验再跑 20 epoch。
2. **CUDA 加速**：若安装了 CUDA 后端，所有脚本支持 `--device cuda`。
3. **grad_norm 修复**：主脚本 `train_mnist.py` 已在 `optimizer.step()` 之前记录梯度范数，确保数值非零。
4. **混淆矩阵**：Exp-7 的 CNN 脚本会在测试阶段额外保存 `confusion_matrix.npy`，供 `plot_all.py` 绘制。
