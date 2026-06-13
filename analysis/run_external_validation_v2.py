"""
External validation (Revision 1)
================================
esm2_pocket 모델을 전체 BitterDB로 학습 -> 문헌 외부 화합물 예측.
BitterDB에 이미 있는 화합물(InChIKey 매칭)은 제외하여 진짜 held-out만 평가.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen
import warnings; warnings.filterwarnings('ignore')

d = np.load('dataset.npz', allow_pickle=True)
X, Y, is_train = d['X'], d['Y'], d['is_train']
RECEPTORS = list(d['receptors'])
cids = list(d['cids'])
Xt, Yt = torch.tensor(X), torch.tensor(Y)
R = torch.tensor(np.load('emb_esm2_pocket.npy'))

# descriptor 정규화 파라미터 재현 (학습셋 기준) — featurize 일치용
desc = pd.read_csv('compounds_with_descriptors_v2.csv').set_index('bdb_cid')
desc_cols=['mw','alogp','tpsa','hbd','hba','rotb','rings','arom_rings','fsp3','formal_charge','heavy_atoms','globularity_proxy']
fpidx=pd.read_csv('fp_cid_index_v2.csv')['bdb_cid'].tolist()
Draw=np.zeros((len(fpidx),12),dtype=np.float32)
for i,c in enumerate(fpidx):
    if c in desc.index: Draw[i]=desc.loc[c,desc_cols].values.astype(np.float32)
DMU, DSD = Draw.mean(0), Draw.std(0)+1e-8

# BitterDB InChIKey 집합 (leakage 필터)
bdb_keys=set()
for c in fpidx:
    if c in desc.index:
        s=desc.loc[c,'smiles']
        m=Chem.MolFromSmiles(s) if isinstance(s,str) else None
        if m: bdb_keys.add(Chem.MolToInchiKey(m))
print(f"BitterDB InChIKey: {len(bdb_keys)}개")

def featurize(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None: return None
    fp=AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048)
    arr=np.zeros(2048,dtype=np.float32); Chem.DataStructs.ConvertToNumpyArray(fp,arr)
    ring=sum(1 for a in m.GetAtoms() if a.IsInRing()); heavy=m.GetNumHeavyAtoms()
    dd=np.array([Descriptors.MolWt(m),Crippen.MolLogP(m),rdMolDescriptors.CalcTPSA(m),
        rdMolDescriptors.CalcNumHBD(m),rdMolDescriptors.CalcNumHBA(m),
        rdMolDescriptors.CalcNumRotatableBonds(m),rdMolDescriptors.CalcNumRings(m),
        rdMolDescriptors.CalcNumAromaticRings(m),rdMolDescriptors.CalcFractionCSP3(m),
        Chem.GetFormalCharge(m),heavy,(ring/heavy if heavy else 0)],dtype=np.float32)
    dd=(dd-DMU)/DSD
    return np.concatenate([arr,dd]), Chem.MolToInchiKey(m)

class SelectNet(nn.Module):
    def __init__(s,cd,rd,e=256,n=23):
        super().__init__()
        s.comp=nn.Sequential(nn.Linear(cd,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,e))
        s.rec=nn.Sequential(nn.Linear(rd,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,e))
        s.W=nn.Parameter(torch.randn(e,e)*0.02); s.b=nn.Parameter(torch.zeros(n))
    def forward(s,xc,Rm): return s.comp(xc)@s.W@s.rec(Rm).T+s.b
def fl(l,t,g=2.0):
    p=torch.sigmoid(l); ce=nn.functional.binary_cross_entropy_with_logits(l,t,reduction='none')
    pt=p*t+(1-p)*(1-t); return ((1-pt)**g*ce).mean()

# 전체 데이터로 학습 (외부검증이므로 BitterDB 전체 사용)
torch.manual_seed(42); np.random.seed(42)
model=SelectNet(X.shape[1],R.shape[1])
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80)
for ep in range(80):
    model.train(); perm=torch.randperm(len(Xt))
    for i in range(0,len(Xt),128):
        bi=perm[i:i+128]; opt.zero_grad()
        fl(model(Xt[bi],R),Yt[bi]).backward(); opt.step()
    sch.step()
print("학습 완료 (80 epoch, 전체 BitterDB)")

# 외부 화합물 예측
ext=pd.read_csv('external_validation_set_v2.csv')
model.eval()
rows=[]
for _,r in ext.iterrows():
    s=r['SMILES']
    if not isinstance(s,str) or not s.strip(): continue
    out=featurize(s)
    if out is None: continue
    x,key=out
    in_bdb = key in bdb_keys
    with torch.no_grad():
        prob=torch.sigmoid(model(torch.tensor(x)[None],R)).numpy().ravel()
    tj=RECEPTORS.index(r['receptor'])
    p_other=max(p for i,p in enumerate(prob) if i!=tj)
    rows.append({'compound':r['compound_name'],'receptor':r['receptor'],
        'in_bitterdb':in_bdb,'p_target':round(float(prob[tj]),3),
        'p_max_other':round(float(p_other),3),
        'argmax_correct':bool(prob.argmax()==tj),
        'pred_receptor':RECEPTORS[prob.argmax()]})
res=pd.DataFrame(rows)
res.to_csv('external_validation_predictions_v2.csv',index=False)
print("\n=== 전체 예측 ===")
print(res.to_string(index=False))

held=res[~res.in_bitterdb]
print(f"\n=== 진짜 held-out (BitterDB 미등재): {len(held)}개 ===")
print(held[['compound','receptor','p_target','argmax_correct','pred_receptor']].to_string(index=False))
print(f"\nHit rate (p_target>0.5): {(held.p_target>0.5).mean():.1%}")
print(f"Median p_target: {held.p_target.median():.3f}")
print(f"올바른 수용체 argmax: {held.argmax_correct.mean():.1%}")
json.dump({'n_held_out':int(len(held)),
    'hit_rate':round(float((held.p_target>0.5).mean()),3),
    'median_p':round(float(held.p_target.median()),3),
    'argmax_acc':round(float(held.argmax_correct.mean()),3)},
    open('external_validation_summary_v2.json','w'),indent=2)

# ── tier별 층화 분석 (broadly-tuned vs near-orphan) ──────────────────
import pandas as _pd
_df = _pd.read_csv('external_validation_predictions_v2.csv')
_held = _df[~_df.in_bitterdb]
_broad = _held[_held.receptor.isin(['TAS2R14','TAS2R46','TAS2R38'])]
_orphan = _held[_held.receptor=='TAS2R2']
print("\n=== tier별 층화 ===")
print(f"Broad/mod (R14/R46/R38): n={len(_broad)}, argmax={_broad.argmax_correct.mean():.0%}, "
      f"hit={(_broad.p_target>0.5).mean():.0%}, median_p={_broad.p_target.median():.3f}")
print(f"Near-orphan (R2): n={len(_orphan)}, argmax={_orphan.argmax_correct.mean():.0%}")
json.dump({
 'all_held_out':{'n':int(len(_held)),'argmax':round(float(_held.argmax_correct.mean()),3),
                 'hit':round(float((_held.p_target>0.5).mean()),3)},
 'broad_held_out':{'n':int(len(_broad)),'argmax':round(float(_broad.argmax_correct.mean()),3),
                   'hit':round(float((_broad.p_target>0.5).mean()),3),
                   'median_p':round(float(_broad.p_target.median()),3)},
 'orphan_held_out':{'n':int(len(_orphan)),'argmax':round(float(_orphan.argmax_correct.mean()),3)}
}, open('external_validation_stratified.json','w'), indent=2)
print("저장: external_validation_stratified.json")
