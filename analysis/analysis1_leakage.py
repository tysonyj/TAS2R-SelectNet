"""
분석 #1: Leakage 메커니즘 정량화
================================
test 화합물의 nearest-train Tanimoto 유사도가 AUC inflation을 설명하는가?
- random split vs scaffold split에서 nearest-train 유사도 분포 비교
- 유사도 bin별 예측 정확도(per-pair) → leakage가 유사도의 함수임을 정량화
"""
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

fp=np.load('morgan_fingerprints_v2.npy').astype(bool)
fpidx=pd.read_csv('fp_cid_index_v2.csv')['bdb_cid'].tolist()
cid2row={c:i for i,c in enumerate(fpidx)}

def tanimoto(a,b):
    # a: (n,2048) bool, b: (m,2048) bool -> (n,m)
    inter=a.astype(int)@b.T.astype(int)
    sa=a.sum(1)[:,None]; sb=b.sum(1)[None,:]
    return inter/(sa+sb-inter+1e-9)

# scaffold split
scf=pd.read_csv('scaffold_split_v2.csv')
train_cids=scf[scf['split']=='train']['bdb_cid'].tolist()
test_cids=scf[scf['split']=='test']['bdb_cid'].tolist()
tr_rows=[cid2row[c] for c in train_cids if c in cid2row]
te_rows=[cid2row[c] for c in test_cids if c in cid2row]

# scaffold split: test별 nearest-train 유사도
sim_scaf=tanimoto(fp[te_rows],fp[tr_rows]).max(1)
print(f"=== Scaffold split nearest-train Tanimoto ===")
print(f"  mean={sim_scaf.mean():.3f}, median={np.median(sim_scaf):.3f}, >0.4 비율={np.mean(sim_scaf>0.4):.1%}")

# random split (동일 크기로 5회 평균)
rng=np.random.default_rng(42)
all_rows=tr_rows+te_rows
n_test=len(te_rows)
sims_rand=[]
for _ in range(5):
    perm=rng.permutation(all_rows)
    rte=perm[:n_test]; rtr=perm[n_test:]
    sims_rand.append(tanimoto(fp[rte],fp[rtr]).max(1))
sim_rand=np.concatenate(sims_rand)
print(f"=== Random split nearest-train Tanimoto (5x) ===")
print(f"  mean={sim_rand.mean():.3f}, median={np.median(sim_rand):.3f}, >0.4 비율={np.mean(sim_rand>0.4):.1%}")

# 통계 검정: 두 분포 차이
from scipy import stats
u,p=stats.mannwhitneyu(sim_rand,sim_scaf,alternative='greater')
print(f"\nMann-Whitney U (random > scaffold nearest-sim): p={p:.2e}")
print(f"Δ median similarity: {np.median(sim_rand)-np.median(sim_scaf):+.3f}")

json.dump({
 'scaffold':{'mean':round(float(sim_scaf.mean()),3),'median':round(float(np.median(sim_scaf)),3),'frac_gt04':round(float(np.mean(sim_scaf>0.4)),3)},
 'random':{'mean':round(float(sim_rand.mean()),3),'median':round(float(np.median(sim_rand)),3),'frac_gt04':round(float(np.mean(sim_rand>0.4)),3)},
 'mannwhitney_p':float(p),
 'sim_scaf':sim_scaf.tolist(),'sim_rand':sim_rand.tolist()
}, open('/home/claude/tex_work/leakage_sim.json','w'))
print("\n저장: leakage_sim.json")
