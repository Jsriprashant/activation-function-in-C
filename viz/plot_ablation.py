import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Assume ablations CSV: act_type, seed, final_acc, conv_epochs
df = pd.read_csv('experiments/ablations.csv')  # Gen manually or script
pivot = df.pivot(index='act_type', columns='seed', values='final_acc')
sns.heatmap(pivot, annot=True, cmap='viridis')
plt.title('Acc Heatmap by Act Type & Seed')
plt.savefig('results/ablation_acc.png')
plt.show()