#!/usr/bin/env python3
"""Generate summary visualizations from experiments/results CSVs.

Outputs (saved to viz/results/):
- {dataset}_act_compare_default.png  : loss & acc comparison (default init)
- {dataset}_<act>_params_default.png: activation parameter trajectories for activations that log params
- {dataset}_final_acc_by_act.png    : bar chart of best final accuracy per activation (from experiments/ablations.csv)

Run from project root:
    python viz/make_plots.py

Requires: pandas, matplotlib, seaborn
"""
import os
import re
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "experiments" / "results"
OUT_DIR = ROOT / "viz" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["xor", "spirals", "mnist"]
ACTS = ["fixed_relu", "prelu", "poly_cubic", "piecewise", "swish"]


def find_run_csv(dataset, act, init="ACT_INIT_DEFAULT"):
    pattern = f"{dataset}_{act}_act_init_default_results_*.csv"
    files = sorted(RES_DIR.glob(pattern))
    return files[0] if files else None


def load_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def plot_loss_acc_comparison(dataset):
    rows = []
    for act in ACTS:
        p = find_run_csv(dataset, act)
        if p is None:
            continue
        df = load_csv(p)
        if df is None or 'epoch' not in df.columns:
            continue
        rows.append((act, df))

    if not rows:
        print(f"No default-init runs found for {dataset}")
        return

    sns.set(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for act, df in rows:
        axes[0].plot(df['epoch'], df['loss'], label=act)
        axes[1].plot(df['epoch'], df['acc'], label=act)

    axes[0].set_title(f"{dataset} — Loss vs Epoch (default init)")
    axes[1].set_title(f"{dataset} — Accuracy vs Epoch (default init)")
    axes[0].set_xlabel('epoch')
    axes[1].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()
    axes[0].legend()

    out = OUT_DIR / f"{dataset}_act_compare_default.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("Wrote", out)


def plot_activation_params(dataset):
    # For each activation run present for dataset, if CSV contains param columns, plot them
    for act in ACTS:
        p = find_run_csv(dataset, act)
        if p is None:
            continue
        df = load_csv(p)
        if df is None:
            continue
        # detect parameter columns beyond epoch,loss,acc
        param_cols = [c for c in df.columns if re.search(r'_p\d+$', c)]
        if not param_cols:
            continue
        sns.set(style='whitegrid')
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in param_cols:
            ax.plot(df['epoch'], df[c], label=c)
        ax.set_title(f"{dataset} — {act} params (default init)")
        ax.set_xlabel('epoch')
        ax.set_ylabel('param value')
        ax.legend(fontsize='small')
        out = OUT_DIR / f"{dataset}_{act}_params_default.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print("Wrote", out)


def plot_final_acc_bars():
    # Read experiments/ablations.csv and aggregate best final_acc per act_hidden for each dataset
    abla = ROOT / 'experiments' / 'ablations.csv'
    if not abla.exists():
        print('No ablations.csv found; skipping final-acc bar charts')
        return
    df = pd.read_csv(abla)
    # find best final_acc per (dataset,act_hidden)
    best = df.groupby(['dataset','act_hidden'])['final_acc'].max().reset_index()
    for ds in best['dataset'].unique():
        sub = best[best['dataset']==ds]
        sns.set(style='whitegrid')
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x='act_hidden', y='final_acc', data=sub, ax=ax)
        ax.set_title(f"{ds} — best final accuracy by activation (best init)")
        ax.set_ylim(0,1.0)
        plt.xticks(rotation=45)
        out = OUT_DIR / f"{ds}_final_acc_by_act.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print("Wrote", out)


def main():
    for ds in DATASETS:
        plot_loss_acc_comparison(ds)
        plot_activation_params(ds)
    plot_final_acc_bars()


if __name__ == '__main__':
    main()
