"""우리 645 데이터로 SI candidate 수 재계산 (전체 학습 모델 사용)."""
import numpy as np, torch, torch.nn as nn, json
import warnings; warnings.filterwarnings('ignore')

d=np.load('dataset.npz',allow_pickle=True)
X,Y=d['X'],d['Y']; RECEPTORS=list(d['receptors'])
Xt,Yt=torch.tensor(X),torch.tensor(Y)
R=torch.tensor(np.load('emb_esm2_pocket.npy'))

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

torch.manual_seed(42); np.random.seed(42)
m=SelectNet(X.shape[1],R.shape[1])
opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80)
for ep in range(80):
    m.train(); perm=torch.randperm(len(Xt))
    for i in range(0,len(Xt),128):
        bi=perm[i:i+128]; opt.zero_grad(); fl(m(Xt[bi],R),Yt[bi]).backward(); opt.step()
    sch.step()
m.eval()
with torch.no_grad(): P=torch.sigmoid(m(Xt,R)).numpy()  # (645,23)

# SI = p_i / mean_j(p_j)
SI = P / P.mean(axis=1,keepdims=True)
# monoselective: p>0.5 and SI>1.5 for exactly one subtype
mono=0; per_sub={r:0 for r in RECEPTORS}
for c in range(len(P)):
    hits=[(j) for j in range(23) if P[c,j]>0.5 and SI[c,j]>1.5]
    if len(hits)==1:
        mono+=1; per_sub[RECEPTORS[hits[0]]]+=1
n_sub=sum(1 for v in per_sub.values() if v>0)
print(f"화합물 645개 기준 monoselective candidates: {mono}")
print(f"커버 subtype 수: {n_sub}")
print("subtype별 (>0):", {k:v for k,v in sorted(per_sub.items(),key=lambda x:-x[1]) if v>0})
json.dump({'n_candidates':mono,'n_subtypes':n_sub,'per_subtype':per_sub}, open('candidate_recount.json','w'),indent=2)
