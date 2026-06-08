import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minitorch.nn import Linear, Sigmoid, Flatten, Sequential, CrossEntropyLoss
from minitorch.optim import SGD
from minitorch.MNISTIterator import load_mnist, MNISTIterator
from minitorch.tensor_ops import SimpleBackend
import numpy as np

train_images, train_labels = load_mnist('./data', kind='train')

def test_config(batch_size, hidden):
    train_iter = MNISTIterator(train_images, train_labels, batch_size=batch_size, shuffle=True, backend='cpu')
    images, labels = next(iter(train_iter))
    layers = [Flatten()]
    in_size = 784
    for h in hidden:
        layers.append(Linear(in_size, h, bias=True, init_mode='xavier'))
        layers.append(Sigmoid())
        in_size = h
    layers.append(Linear(in_size, 10, bias=True, init_mode='xavier'))
    model = Sequential(*layers)
    optimizer = SGD(model.parameters(), lr=0.1, weight_decay=0.0)
    loss_fn = CrossEntropyLoss()
    
    # Warmup
    logits = model(images)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Timed run
    t0 = time.time()
    logits = model(images)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - t0
    print(f'batch={batch_size:3d}, hidden={str(hidden):20s} -> {elapsed:.4f}s')
    return elapsed

print('=== Default-like configs ===')
test_config(64, [128, 64])
test_config(64, [128])
test_config(32, [128, 64])
test_config(16, [128, 64])
test_config(8, [128, 64])
test_config(4, [128, 64])

print('\n=== Small configs ===')
test_config(8, [32])
test_config(8, [16])
test_config(8, [8])
