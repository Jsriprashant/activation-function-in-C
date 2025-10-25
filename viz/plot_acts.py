import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assume CSV has cols for params, e.g., param0,param1,... per epoch
def plot_act_evol(csv_file, act_type):
    df = pd.read_csv(csv_file)
    # param columns are assumed to be the columns after 'acc'
    cols = list(df.columns)
    if 'epoch' not in cols or 'loss' not in cols or 'acc' not in cols:
        raise ValueError('CSV must contain epoch,loss,acc columns')
    param_cols = cols[3:]
    z = np.linspace(-5, 5, 200)
    fig, ax = plt.subplots()
    sample_epochs = [df['epoch'].iloc[0]]
    # choose up to 3 sample epochs (start, mid, end)
    sample_epochs.append(df['epoch'].iloc[len(df)//2])
    sample_epochs.append(df['epoch'].iloc[-1])
    for epoch in sample_epochs:
        row = df[df['epoch'] == epoch].iloc[0]
        params = [row[c] for c in param_cols]
        if act_type == 'poly':
            if len(params) < 4:
                raise ValueError('Not enough params for poly')
            a0, a1, a2, a3 = params[0:4]
            fz = a0 + a1 * z + a2 * z**2 + a3 * z**3
        elif act_type == 'prelu':
            alpha = params[0]
            fz = np.maximum(0, z) + alpha * np.minimum(0, z)
        elif act_type == 'swish':
            beta = params[0]
            fz = z * (1.0 / (1.0 + np.exp(-beta * z)))
        elif act_type == 'piecewise':
            # assume params layout: tau0_raw, log(d1), log(d2), s0,s1,s2,s3
            p0 = params[0]; p1 = params[1]; p2 = params[2]
            tau0 = p0; tau1 = p0 + np.exp(p1); tau2 = tau1 + np.exp(p2)
            s = params[3:7]
            fz = np.zeros_like(z)
            for i, zi in enumerate(z):
                seg = 0
                if zi > tau0: seg = 1
                if zi > tau1: seg = 2
                if zi > tau2: seg = 3
                c = 0.0
                for m in range(seg):
                    c += (s[m] - s[m+1]) * ([tau0, tau1, tau2][m])
                fz[i] = s[seg] * zi + c
        else:
            # fallback: plot first param scaled
            fz = params[0] * np.tanh(z)
        ax.plot(z, fz, label=f'Epoch {epoch}')
    ax.set_title(f'{act_type.upper()} Evolution')
    ax.legend()
    outname = f"viz/{act_type}_evol.png"
    plt.savefig(outname)
    print(f"Saved activation evolution plot to {outname}")
    plt.show()

# Usage: python plot_acts.py xor_poly.csv poly
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python plot_acts.py <csv_file> <act_type>')
        sys.exit(1)
    csv_file = sys.argv[1]
    act_type = sys.argv[2]
    plot_act_evol(csv_file, act_type)