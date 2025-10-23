import numpy as np
import struct


# Parse IDX format (uncompressed)
def load_mnist_images(filename):
    with open(filename, "rb") as f:
        f.read(16)  # Header: magic (4), n (4), rows (4), cols (4)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64) / 255.0
        data = data.reshape(-1, 28 * 28)  # n x 784
    return data


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        f.read(8)  # Header: magic (4), n (4)
        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)  # float64 for C
        labels = labels.reshape(-1, 1)  # n x 1 (class indices)
    return labels


# Train set (subset to 10k for speed; remove [:10000] for full 60k)
X_train = load_mnist_images("train-images.idx3-ubyte")[:10000]
Y_train = load_mnist_labels("train-labels.idx1-ubyte")[:10000]

# Write train binary
data_train = np.hstack([X_train, Y_train])  # n x (784 + 1)
with open("mnist_train.bin", "wb") as f:
    f.write(struct.pack("iii", len(X_train), 784, 1))  # out_dim=1 (indices)
    f.write(data_train.tobytes())

print("Generated mnist_train.bin")

# Test set (full 10k)
X_test = load_mnist_images("t10k-images.idx3-ubyte")
Y_test = load_mnist_labels("t10k-labels.idx1-ubyte")

# Write test binary
data_test = np.hstack([X_test, Y_test])
with open("mnist_test.bin", "wb") as f:
    f.write(struct.pack("iii", len(X_test), 784, 1))
    f.write(data_test.tobytes())

print("Generated mnist_test.bin")
