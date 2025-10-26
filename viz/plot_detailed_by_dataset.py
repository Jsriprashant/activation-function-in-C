import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

ROOT = 'experiments/results'
OUT = os.path.join('viz', 'results')
os.makedirs(OUT, exist_ok=True)

datasets = ['xor', 'spirals', 'mnist']
# colors for init strategies
INIT_COLORS = {'default': '#1f77b4', 'noisy': '#ff7f0e', 'random_small': '#2ca02c'}

# list files
files = [f for f in os.listdir(ROOT) if f.endswith('.csv')]

# helper to extract activation token after dataset_
def extract_act_token(fname, dataset):
    m = re.match(rf'^{re.escape(dataset)}_(.+?)(?:_act|_results|_42|\.csv|$)', fname)
    if m:
        return m.group(1)
    return None

# helper to detect init from filename
def detect_init(fname):
    if 'act_init_noisy' in fname or '_noisy' in fname:
        return 'noisy'
    if 'act_init_random_small' in fname or 'random_small' in fname:
        return 'random_small'
    if 'act_init_default' in fname or '_default' in fname:
        return 'default'
    # fallback: if file contains 'act_init' but unknown label, return the token
    m = re.search(r'act_init_([a-zA-Z0-9_]+)', fname)
    if m:
        return m.group(1)
    return 'default'

for ds in datasets:
    # discover activations present for this dataset
    acts = {}
    for f in files:
        if not f.startswith(ds + '_'):
            continue
        token = extract_act_token(f, ds)
        if not token:
            continue
        # canonicalize token (strip trailing _results/_42 handled above)
        acts.setdefault(token, []).append(f)

    if not acts:
        print(f'No result CSVs found for dataset {ds} in {ROOT}')
        continue

    # sort activations alphabetically but keep stable order
    act_keys = sorted(acts.keys())
    n = len(act_keys)
    # create figure: 2 rows (loss, acc), n columns
    fig, axes = plt.subplots(2, n, figsize=(4*n if n>0 else 6, 6), squeeze=False)
    for col, act in enumerate(act_keys):
        col_files = acts[act]
        # for each init strategy, try to find the corresponding file
        inits = {}
        for f in col_files:
            init = detect_init(f)
            inits[init] = f
        # ensure consistent order
        for init in ['default', 'noisy', 'random_small']:
            if init not in inits:
                # maybe file name lacks act_init but is a poly baseline (e.g., *_poly_42.csv)
                pass
        ax_loss = axes[0, col]
        ax_acc = axes[1, col]
        plotted = 0
        for init, fname in inits.items():
            path = os.path.join(ROOT, fname)
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f'Could not read {path}: {e}')
                continue
            # coerce numeric
            df['loss'] = pd.to_numeric(df.get('loss', pd.Series()), errors='coerce')
            df['acc'] = pd.to_numeric(df.get('acc', pd.Series()), errors='coerce')
            # prefer epoch column if present
            if 'epoch' in df.columns:
                x = pd.to_numeric(df['epoch'], errors='coerce')
            else:
                x = pd.RangeIndex(start=0, stop=len(df))
            if df['loss'].notna().any():
                ax_loss.plot(x, df['loss'], label=init, color=INIT_COLORS.get(init, None))
            if df['acc'].notna().any():
                ax_acc.plot(x, df['acc'], label=init, color=INIT_COLORS.get(init, None))
            plotted += 1
        # axis formatting
        act_label = act.replace('_', ' ').upper()
        ax_loss.set_title(act_label)
        ax_loss.set_xlabel('epoch')
        ax_loss.set_ylabel('loss')
        ax_acc.set_xlabel('epoch')
        ax_acc.set_ylabel('accuracy')
        # legends (only once per column)
        if plotted > 0:
            ax_loss.legend(fontsize=8)
            ax_acc.legend(fontsize=8)
        else:
            ax_loss.text(0.5, 0.5, 'no data', ha='center', va='center')
            ax_acc.text(0.5, 0.5, 'no data', ha='center', va='center')

    plt.suptitle(f'{ds.upper()}: loss and accuracy by activation (inits compared)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(OUT, f'{ds}_detailed_activations.png')
    plt.savefig(out_path, dpi=150)
    print(f'Wrote {out_path}')

print('Done')
