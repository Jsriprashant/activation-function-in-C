import csv
import sys
p='experiments/ablations.csv'
bad=[]
with open(p, newline='') as f:
    r=csv.reader(f)
    header=next(r, None)
    for i,row in enumerate(r, start=2):
        if len(row) < 6:
            bad.append((i, 'short row', row))
            continue
        fa=row[5]
        try:
            float(fa)
        except Exception as e:
            bad.append((i, fa, row))
if not bad:
    print('All final_acc entries look numeric')
else:
    print('Found non-numeric final_acc entries:')
    for b in bad[:20]:
        print(b)
    if len(bad) > 20:
        print('... (truncated)')
