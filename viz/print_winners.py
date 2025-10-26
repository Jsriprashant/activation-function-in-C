import pandas as pd
import os
p='experiments/ablations.csv'
df=pd.read_csv(p, header=0)
INIT_MAP={'ACT_INIT_NOISY':'noisy','ACT_INIT_RANDOM_SMALL':'random_small','ACT_INIT_DEFAULT':'default'}
df['act_init_norm']=df['act_init'].map(INIT_MAP).fillna(df['act_init'])
for init in ['noisy','random_small']:
    print('\nInit:',init)
    for ds in ['xor','spirals','mnist']:
        sub=df[(df['dataset']==ds)&(df['act_init_norm']==init)]
        if sub.empty:
            print(' ',ds,': no data')
            continue
        # convert final_acc
        sub['final_acc']=pd.to_numeric(sub['final_acc'], errors='coerce')
        best=sub.loc[sub['final_acc'].idxmax()]
        print(' ',ds,':',best['act_hidden'],'->',best['final_acc'])
