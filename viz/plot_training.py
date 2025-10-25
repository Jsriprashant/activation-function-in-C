import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

if __name__ == "__main__":
    fname = sys.argv[1]  # e.g., xor_poly_42.csv
    df = pd.read_csv(fname)
    plt.figure()
    plt.plot(df["epoch"], df["loss"], label="Loss")
    plt.plot(df["epoch"], df["acc"], label="Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    # Save into viz/results directory
    outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(fname)
    outpath = os.path.join(outdir, base.replace('.csv', '_curves.png'))
    plt.savefig(outpath)
    print(f"Saved training curves to {outpath}")
    plt.show()
