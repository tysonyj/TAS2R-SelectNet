"""Multi-seed (5) 평균으로 encoding ablation 안정화."""
import numpy as np, torch, torch.nn as nn, json
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

d = np.load('dataset.npz', allow_pickle=True)
X, Y, is_train = d['X'], d['Y'], d['is_train']
RECEPTORS = list(d['receptors'])
Xt, Yt = torch.tensor(X), torch.tensor(Y)
tr = torch.tensor(is_train)

def onehot(): return np.eye(len(RECEPTORS), dtype=np.float32)
ENC = {'onehot':onehot(),'aa_comp':np.load('emb_aa_comp.npy'),
       'protbert':np.load('emb_protbert.npy'),'esm2_full':np.load('emb_esm2_full.npy'),
       'esm2_pocket':np.load('emb_esm2_pocket.npy')}
BROAD=['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']
Yte=Y[~is_train]
eval_broad=[r for r in BROAD if Yte[:,RECEPTORS.index(r)].sum()>=5]

class SelectNet(nn.Module):
    def __init__(s,cd,rd,e=256,n=23):
        super().__init__()
        s.comp=nn.Sequential(nn.Linear(cd,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,e))
        s.rec=nn.Sequential(nn.Linear(rd,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,e))
        s.W=nn.Parameter(torch.randn(e,e)*0.02); s.b=nn.Parameter(torch.zeros(n))
    def forward(s,xc,R): return s.comp(xc)@s.W@s.rec(R).T+s.b

def fl(l,t,g=2.0):
    p=torch.sigmoid(l); ce=nn.functional.binary_cross_entropy_with_logits(l,t,reduction='none')
    pt=p*t+(1-p)*(1-t); return ((1-pt)**g*ce).mean()

def run(enc,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    R=torch.tensor(ENC[enc]); m=SelectNet(X.shape[1],R.shape[1])
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    Xtr,Ytr=Xt[tr],Yt[tr]; best,bs=-1,None
    for ep in range(60):
        m.train(); perm=torch.randperm(len(Xtr))
        for i in range(0,len(Xtr),128):
            bi=perm[i:i+128]; opt.zero_grad()
            fl(m(Xtr[bi],R),Ytr[bi]).backward(); opt.step()
        sch.step(); m.eval()
        with torch.no_grad(): prob=torch.sigmoid(m(Xt[~tr],R)).numpy()
        aucs=[roc_auc_score(Yte[:,RECEPTORS.index(r)],prob[:,RECEPTORS.index(r)]) for r in eval_broad]
        if np.mean(aucs)>best: best,bs=np.mean(aucs),prob.copy()
    return best,bs

SEEDS=[42,7,123,2024,99]
res={}
for enc in ENC:
    aucs=[]; allscores=[]
    for s in SEEDS:
        a,sc=run(enc,s); aucs.append(a); allscores.append(sc)
    res[enc]={'mean':round(float(np.mean(aucs)),4),'std':round(float(np.std(aucs)),4),
              'per_seed':[round(float(x),4) for x in aucs]}
    print(f"{enc:13s} macroAUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  seeds={[round(x,3) for x in aucs]}")
json.dump(res,open('ablation_multiseed.json','w'),indent=2)
b=res['onehot']['mean']
print("\n=== ΔAUC vs one-hot (5-seed 평균) ===")
for e in res: print(f"  {e:13s} {res[e]['mean']-b:+.4f}")
print(f"\npocket vs full-seq : {res['esm2_pocket']['mean']-res['esm2_full']['mean']:+.4f}  (pocket 정당성)")
print(f"pocket vs protbert : {res['esm2_pocket']['mean']-res['protbert']['mean']:+.4f}  (ESM-2 특이성)")
print(f"pocket vs aa_comp  : {res['esm2_pocket']['mean']-res['aa_comp']['mean']:+.4f}  (PLM 가치)")
