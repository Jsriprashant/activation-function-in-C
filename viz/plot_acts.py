import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assume CSV has cols for params, e.g., param0,param1,... per epoch
def plot_act_evol(csv_file, act_type):
    df = pd.read_csv(csv_file)
    z = np.linspace(-5, 5, 100)
    fig, ax = plt.subplots()
    for epoch in [0, 50, 100]:  # Sample
        row = df[df['epoch'] == epoch].iloc[0]
        if act_type == 'poly':
            a = [row[f'param{i}'] for i in range(4)]
            fz = a[0] + a[1]*z + a[2]*z**2 + a[3]*z**3
        elif act_type == 'prelu':
            alpha = row['param0']
            fz = np.maximum(0, z) + alpha * np.minimum(0, z)
        # Add others
        ax.plot(z, fz, label=f'Epoch {epoch}')
    ax.set_title(f'{act_type.upper()} Evolution')
    ax.legend()
    plt.savefig(f"results/{act_type}_evol.png")
    plt.show()

# Usage: python plot_acts.py xor_poly.csv poly