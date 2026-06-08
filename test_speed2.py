import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minitorch.nn import Linear, Sigmoid, Flatten, Sequential, CrossEntropyLoss
from minitorch.optim import SGD
from minitorch.MNISTIterator import load_mnist, MNISTIterator
from minitorch.tensor_ops import SimpleBackend
import numpy as np

train_images, train_labels = load_mnist('./data', kind='train')

def test_config(batch_size, hidden, act='Sigmoid'):
    train_iter = MNISTIterator(train_images, train_labels, batch_size=batch_size, shuffle=True, backend='cpu')
    images, labels = next(iter(train_iter))
    layers = [Flatten()]
    in_size = 784
    for h in hidden:
        layers.append(Linear(in_size, h, bias=True, init_mode='xavier'))
        if act == 'Sigmoid':
            from minitorch.nn import Sigmoid as Act
        elif act == 'ReLU':
            from minitorch.nn import ReLU as Act
        else:
            from minitorch.nn import Tanh as Act
        layers.append(Act())
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
    
    # Timed run: forward + loss + backward + step (full training batch)
    t0 = time.time()
    logits = model(images)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - t0
    
    # Also test just forward
    t1 = time.time()
    _ = model(images)
    fwd_only = time.time() - t1
    
    print(f'batch={batch_size:3d}, hidden={str(hidden):20s}, act={act:7s} -> full_batch={elapsed:.2f}s, forward_only={fwd_only:.2f}s')
    return elapsed

print('=== Ultra-small configs ===')
test_config(1, [4])
test_config(1, [2])
test_config(1, [1])
test_config(2, [2])
test_config(2, [4])
test_config(4, [2])
test_config(4, [4])

print('\n=== No hidden layer (logistic regression) ===')
test_config(1, [])
test_config(4, [])
test_config(8, [])
test_config(16, [])
test_config(32, [])

print('\n=== Compare activations (batch=1, hidden=4) ===')
test_config(1, [4], 'Sigmoid')
test_config(1, [4], 'ReLU')
test_config(1, [4], 'Tanh')
