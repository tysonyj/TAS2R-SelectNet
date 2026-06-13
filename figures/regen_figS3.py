"""figS3 재생성: scaffold split 특성 (282 scaffolds, 607 compounds, train 526/test 81)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

P_BLUE='#A8CCE5'; P_RED='#E8A7A0'
def PL(ax,l):
    ax.annotate(l, xy=(0,1), xycoords='axes fraction', xytext=(-30,12),
                textcoords='offset points', fontsize=15, fontweight='bold',
                va='bottom', ha='left', annotation_clip=False)

desc=pd.read_csv('compounds_with_descriptors_v2.csv')
split=pd.read_csv('scaffold_split_v2.csv')
cids=set(split['bdb_cid'])
# scaffold별 화합물 수
scaff_count=Counter()
cid_scaff={}
for _,r in desc.iterrows():
    if r['bdb_cid'] not in cids: continue
    s=r.get('smiles')
    if not isinstance(s,str): continue
    m=Chem.MolFromSmiles(s)
    if m is None: continue
    try:
        sc=MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        scaff_count[sc]+=1; cid_scaff[r['bdb_cid']]=sc
    except: pass

sizes=list(scaff_count.values())
n_scaff=len(scaff_count); n_comp=len(cid_scaff)
singletons=sum(1 for v in sizes if v==1)// 1
sing_pct=singletons/n_scaff*100

# per-subtype train/test 양성 (association)
import json
s3=json.load(open('s3_counts.json'))
order=['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10','TAS2R16','TAS2R43','TAS2R7','TAS2R38','TAS2R5','TAS2R47','TAS2R40','TAS2R8','TAS2R44','TAS2R49','TAS2R41','TAS2R2','TAS2R13','TAS2R50','TAS2R9','TAS2R3','TAS2R42']
tr_train=int(sum(s3[r]['train'] for r in s3)); tr_test=int(sum(s3[r]['test'] for r in s3))

fig,ax=plt.subplots(1,2,figsize=(13,4.8))
# A: scaffold size 분포
maxs=max(sizes)
ax[0].hist(sizes,bins=range(1,min(maxs,20)+2),color=P_BLUE,edgecolor='#555',align='left')
ax[0].set_xlabel('Compounds per scaffold'); ax[0].set_ylabel('Number of scaffolds')
ax[0].annotate(f'Singletons:\n{singletons} ({sing_pct:.0f}%)',xy=(0.55,0.7),xycoords='axes fraction',
               fontsize=10,bbox=dict(boxstyle='round',fc='white',ec='gray'))
PL(ax[0],'A')
# B: train/test 양성 per subtype
x=np.arange(len(order))
trv=[s3[r]['train'] for r in order]; tev=[s3[r]['test'] for r in order]
ax[1].bar(x,trv,color=P_BLUE,edgecolor='#555',label=f'Train (n={tr_train})')
ax[1].bar(x,tev,bottom=trv,color=P_RED,edgecolor='#555',label=f'Test (n={tr_test})')
ax[1].set_xticks(x); ax[1].set_xticklabels([r.replace('TAS2R','T') for r in order],rotation=90,fontsize=7)
ax[1].set_xlabel('TAS2R subtype'); ax[1].set_ylabel('Associations'); ax[1].legend(fontsize=8); PL(ax[1],'B')
plt.tight_layout(); plt.savefig('figS3_scaffold.png',dpi=200,bbox_inches='tight'); plt.close()
print(f"figS3 재생성: {n_scaff} scaffolds, {n_comp} compounds, singletons {singletons}({sing_pct:.0f}%)")
print(f"train assoc {tr_train}, test assoc {tr_test}")
