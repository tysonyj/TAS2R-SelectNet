"""유사도 bin별 예측 정확도 — leakage 메커니즘 정량화."""
import numpy as np, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

fp=np.load('morgan_fingerprints_v2.npy')
import pandas as pd
fp_cids=list(pd.read_csv('fp_cid_index_v2.csv')['bdb_cid'])
d=np.load('dataset.npz',allow_pickle=True)
X=d['X']; Y=d['Y']; is_train=d['is_train']; all_cids=list(d['cids']); R=list(d['receptors'])
cid2fp={c:i for i,c in enumerate(fp_cids)}

def tanimoto(A,B):
    inter=A@B.T; a=A.sum(1,keepdims=True); b=B.sum(1,keepdims=True)
    return inter/np.maximum(a+b.T-inter,1e-9)

# one-hot receptor + compound feature로 RF 학습 (XGBoost baseline 대용, 빠르게)
# broad-tier subtype만 (충분한 데이터)
BROAD=['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']

# 각 test sample(compound×receptor)의 nearest-train compound 유사도와 예측 correctness
tr_idx=np.where(is_train)[0]; te_idx=np.where(~is_train)[0]
tr_fp=fp[[cid2fp[all_cids[i]] for i in tr_idx]]
te_fp=fp[[cid2fp[all_cids[i]] for i in te_idx]]
sim_te=tanimoto(te_fp,tr_fp).max(1)  # 각 test 화합물의 nearest-train 유사도

# receptor별 RF 학습 후 test 예측 수집
records=[]  # (nearest_sim, y_true, y_pred_prob, receptor)
for rec in R:
    j=R.index(rec)
    ytr=Y[tr_idx,j]; yte=Y[te_idx,j]
    if ytr.sum()<5 or yte.sum()<1: continue
    clf=RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=-1,class_weight='balanced')
    clf.fit(X[tr_idx],ytr)
    p=clf.predict_proba(X[te_idx])[:,1]
    for k in range(len(te_idx)):
        records.append((sim_te[k],int(yte[k]),float(p[k]),rec))

records=np.array([(r[0],r[1],r[2]) for r in records])
sims=records[:,0]; ytrue=records[:,1]; yprob=records[:,2]

# 유사도 bin별 AUC
bins=[(0,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.7),(0.7,1.01)]
curve=[]
for lo,hi in bins:
    m=(sims>=lo)&(sims<hi)
    if m.sum()<10 or ytrue[m].sum()<2 or ytrue[m].sum()==m.sum(): 
        curve.append((f"{lo:.1f}-{hi:.1f}",None,int(m.sum()),int(ytrue[m].sum()))); continue
    try:
        auc=roc_auc_score(ytrue[m],yprob[m])
        curve.append((f"{lo:.1f}-{hi:.1f}",float(auc),int(m.sum()),int(ytrue[m].sum())))
    except: curve.append((f"{lo:.1f}-{hi:.1f}",None,int(m.sum()),int(ytrue[m].sum())))

print("=== nearest-train 유사도 bin별 prediction AUC ===")
for label,auc,n,npos in curve:
    print(f"  sim {label}: AUC={auc if auc else 'NA':<6} (n={n}, npos={npos})")

# 전체 상관: 높은 유사도일수록 correct?
correct=(yprob>0.5).astype(int)==ytrue.astype(int)
from scipy import stats
r,p=stats.spearmanr(sims, (yprob*(2*ytrue-1)))  # signed confidence
print(f"\nSpearman(nearest_sim, signed_confidence): r={r:.3f}, p={p:.2e}")

json.dump({'curve':[(l,a,n,np_) for l,a,n,np_ in curve],
           'spearman_r':float(r),'spearman_p':float(p)},
          open('/home/claude/tex_work/leakage_curve.json','w'))
print("저장: leakage_curve.json")
