import os
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
except Exception as e:
    print("Required packages missing. Please install pandas and matplotlib:")
    print("    pip install pandas matplotlib")
    raise

OUT_DIR = os.path.join('viz', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

ABL = os.path.join('experiments', 'ablations.csv')
if not os.path.exists(ABL):
    print(f"Ablations file not found: {ABL}")
    sys.exit(1)

df = pd.read_csv(ABL, header=None, names=['dataset','act_hidden','act_init','seed','final_loss','final_acc','logfile'])
# coerce final_acc to numeric (some entries may be strings); invalid -> NaN
df['final_acc'] = pd.to_numeric(df['final_acc'], errors='coerce')
if df['final_acc'].isna().any():
    print('Warning: some final_acc values could not be parsed as numbers; they will be ignored in averages')

# Normalize act_init strings (some files use ACT_INIT_* naming)

INIT_MAP = {
    'ACT_INIT_NOISY': 'noisy',
    'ACT_INIT_RANDOM_SMALL': 'random_small',
    'ACT_INIT_DEFAULT': 'default'
}

df['act_init_norm'] = df['act_init'].map(INIT_MAP).fillna(df['act_init'])

datasets = ['xor', 'spirals', 'mnist']
init_strategies = ['noisy', 'random_small']

for init in init_strategies:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[0, i]
        sub = df[(df['dataset'] == ds) & (df['act_init_norm'] == init)]
        if sub.empty:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center')
            ax.set_title(f"{ds} — {init}")
            ax.set_xticks([])
            ax.set_ylim(0, 1)
            continue
        # group by activation and take mean final_acc (in case multiple seeds)
        grp = sub.groupby('act_hidden')['final_acc'].mean().reset_index()
        # sort by acc desc for better visual
        grp = grp.sort_values('final_acc', ascending=False)
        ax.bar(grp['act_hidden'], grp['final_acc'], color=matplotlib.cm.tab10.colors[:len(grp)])
        ax.set_title(f"{ds} — {init}")
        ax.set_ylim(0, 1)
        ax.set_ylabel('final_acc')
        ax.set_xticklabels(grp['act_hidden'], rotation=30, ha='right')
        for j, v in enumerate(grp['final_acc']):
            ax.text(j, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'activations_best_by_init_{init}.png')
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

print('Done')
