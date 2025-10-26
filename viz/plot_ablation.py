import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This script reads the project's experiments/ablations.csv which uses the
# header: dataset,act_hidden,act_init,seed,final_loss,final_acc,logfile
ABL = Path('experiments') / 'ablations.csv'
OUT = Path('viz') / 'results'
OUT.mkdir(parents=True, exist_ok=True)

if not ABL.exists():
	print(f"Ablations CSV not found at {ABL}; skipping plot_ablation")
	raise SystemExit(0)

df = pd.read_csv(ABL)

# Normalize act_init for readability
INIT_MAP = {
	'ACT_INIT_NOISY': 'noisy',
	'ACT_INIT_RANDOM_SMALL': 'random_small',
	'ACT_INIT_DEFAULT': 'default'
}
df['act_init_norm'] = df['act_init'].map(INIT_MAP).fillna(df['act_init'])

# For each dataset, create either a heatmap (if multiple seeds) or a bar plot
for ds in df['dataset'].unique():
	sub = df[df['dataset'] == ds]
	# pivot table: rows=act_hidden, cols=seed, values=final_acc
	try:
		pivot = sub.pivot(index='act_hidden', columns='seed', values='final_acc')
	except Exception:
		pivot = None

	if pivot is not None and pivot.shape[1] > 1:
		plt.figure(figsize=(8, max(3, pivot.shape[0]*0.5)))
		sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
		plt.title(f'{ds} — final accuracy by activation (seed columns)')
		out = OUT / f'{ds}_ablation_heatmap.png'
		plt.tight_layout()
		plt.savefig(out)
		plt.close()
		print('Wrote', out)
	else:
		# Aggregate best final_acc per activation (across inits) and plot bar chart
		best = sub.groupby('act_hidden')['final_acc'].max().reset_index()
		best = best.sort_values('final_acc', ascending=False)
		plt.figure(figsize=(8,4))
		sns.barplot(x='act_hidden', y='final_acc', data=best)
		plt.title(f'{ds} — best final accuracy by activation (best init)')
		plt.ylim(0,1.0)
		plt.xticks(rotation=45)
		out = OUT / f'{ds}_ablation_best_acc.png'
		plt.tight_layout()
		plt.savefig(out)
		plt.close()
		print('Wrote', out)

print('plot_ablation: done')