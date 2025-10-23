from sklearn.datasets import make_moons  # Assume OK, or impl
import numpy as np
X, Y = make_moons(200, noise=0.1, random_state=42)
Y = np.eye(2)[Y]  # One-hot
# Same pack to moons.bin
