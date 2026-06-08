"""
最小调试脚本 —— 跑一个 batch 看哪里卡住
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
print("[1] Imports done")

from minitorch.nn import Linear, Sigmoid, Flatten, Sequential, CrossEntropyLoss
from minitorch.optim import SGD
from minitorch.MNISTIterator import load_mnist, MNISTIterator
from minitorch.tensor_ops import SimpleBackend
print("[2] Minitorch imports done")

# 加载数据
print("[3] Loading MNIST...")
train_images, train_labels = load_mnist("./data", kind="train")
print(f"[3] Loaded: {train_images.shape[0]} samples")

# 构建一个 batch
train_iter = MNISTIterator(train_images, train_labels, batch_size=8, shuffle=True, backend="cpu")
images, labels = next(iter(train_iter))
print(f"[4] Got batch: images={images.shape}, labels={labels.shape}")

# 构建小模型
model = Sequential(
    Flatten(),
    Linear(784, 32, bias=True, init_mode="xavier"),
    Sigmoid(),
    Linear(32, 10, bias=True, init_mode="xavier"),
)
optimizer = SGD(model.parameters(), lr=0.1, weight_decay=0.0)
loss_fn = CrossEntropyLoss()
print("[5] Model built")

# 前向传播
print("[6] Forward...")
t0 = time.time()
logits = model(images)
print(f"[6] Forward done in {time.time()-t0:.2f}s, logits shape={logits.shape}")

# Loss
print("[7] Loss...")
t0 = time.time()
loss = loss_fn(logits, labels)
print(f"[7] Loss done in {time.time()-t0:.2f}s, loss={loss.to_numpy().item():.4f}")

# 反向传播
print("[8] Backward...")
t0 = time.time()
optimizer.zero_grad()
loss.backward()
print(f"[8] Backward done in {time.time()-t0:.2f}s")

# 优化器更新
print("[9] Optimizer step...")
t0 = time.time()
optimizer.step()
print(f"[9] Step done in {time.time()-t0:.2f}s")

print("[10] ALL DONE!")
