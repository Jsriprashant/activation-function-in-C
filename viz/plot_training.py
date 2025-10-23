import pandas as pd
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    fname = sys.argv[1]  # e.g., xor_poly_42.csv
    df = pd.read_csv(fname)
    plt.figure()
    plt.plot(df["epoch"], df["loss"], label="Loss")
    plt.plot(df["epoch"], df["acc"], label="Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"results/{fname.replace('.csv','_curves.png')}")
    plt.show()
