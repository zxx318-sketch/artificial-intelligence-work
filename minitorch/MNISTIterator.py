import os
import urllib.request
import gzip
import struct
import math
import numpy as np
from typing import Tuple
from .tensor import Tensor
from .tensor_functions import tensor_from_numpy
from .tensor_ops import SimpleBackend
from .cuda_kernel_ops import CUDABackend

# Begin Task 2.4
class MNISTIterator:
    """
    A simple dataloader for the MNIST dataset.
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True, backend=None):
        back = backend if backend is not None else "cpu"
        if back == "cpu":
            self.backend = SimpleBackend
        else:
            self.backend = CUDABackend
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = images.shape[0]
        self.num_batches = math.ceil(self.num_samples / batch_size)
        
        self.current_batch = 0
        self.indices = np.arange(self.num_samples)

        # Preprocess data
        images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

        # Reshape to (N, 1, 28, 28) and normalize to [0, 1]
        self.images = images.reshape(-1, 1, 28, 28) / 255.0


    def __iter__(self):
        self.current_batch = 0
        # Shuffle indices if required
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.current_batch >= self.num_batches:
            raise StopIteration
        num_classes = 10

        # 1. Calculate start and end indices for the current batch
        start = self.current_batch * self.batch_size
        end = min(start + self.batch_size, self.num_samples)

        # 2. Slice images and labels using shuffled indices
        batch_indices = self.indices[start:end]
        batch_images = self.images[batch_indices]  # shape: (batch, 1, 28, 28)
        batch_labels = self.labels[batch_indices]  # shape: (batch,)

        # 3. One-hot encode labels: (batch,) -> (batch, 10)
        one_hot = np.zeros((len(batch_indices), num_classes), dtype=np.float32)
        one_hot[np.arange(len(batch_indices)), batch_labels] = 1.0

        # 4. Convert to Tensors
        image_tensor = tensor_from_numpy(batch_images, backend=self.backend)
        label_tensor = tensor_from_numpy(one_hot, backend=self.backend)

        self.current_batch += 1
        return image_tensor, label_tensor
# End Task 2.4

# Helper functions(only for example)


def download_mnist(data_dir: str = "./data"):
    """
    Download the MNIST dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    for key, filename in files.items():
        url = base_url + filename
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {filename} to {filepath}")
        else:
            print(f"{filepath} already exists, skipping.")


def load_mnist(data_dir: str = "./data", kind: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset from specified directory.
    Returns:
        images: np.ndarray of shape (N, 28, 28) with pixel values in [0, 255].
        labels: np.ndarray of shape (N,) with integer labels in [0, 9].
    """
    download_mnist(data_dir)

    if kind == "train":
        labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
        images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    else:
        labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
        images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")

    # Read Labels
    with gzip.open(labels_path, 'rb') as lbpath:
        # Magic number and number of items are 2 integers (8 bytes)
        struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    # Read Images
    with gzip.open(images_path, 'rb') as imgpath:
        # Magic number, num_images, rows, cols are 4 integers (16 bytes)
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num, rows, cols)

    return images, labels
    