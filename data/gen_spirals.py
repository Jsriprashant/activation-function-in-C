import numpy as np
import struct


def gen_spirals(n_points=100, n_arms=2):
    t = np.linspace(0, 4 * np.pi, n_points // n_arms)
    x1 = np.concatenate([t * np.cos(t), -t * np.cos(t)])
    y1 = np.concatenate([t * np.sin(t), -t * np.sin(t)])
    X = np.stack([x1, y1], axis=1).astype(np.float64)
    Y = np.tile([0, 1], (n_points // 2, 1)).astype(
        np.float64
    )  # One-hot-ish, but for CE use idx
    Y = np.zeros((n_points, 2))  # One-hot
    Y[: n_points // 2, 0] = 1
    Y[n_points // 2 :, 1] = 1
    return X, Y


X, Y = gen_spirals(200)
data = np.concatenate([X, Y], axis=1)

with open("spirals.bin", "wb") as f:
    f.write(struct.pack("iii", len(X), X.shape[1], Y.shape[1]))
    f.write(data.tobytes())

print("Generated spirals.bin")
