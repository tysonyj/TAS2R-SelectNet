"""
모든 production figure 재생성 — A/B 패널 레이블 + 내부 제목 제거 + fig5D 실데이터
================================================================================
전제: build_pocket_embeddings_2560.py + run_production.py 이미 실행됨
      (tas2r_pocket_embeddings_v2.npy 존재)
이 스크립트는 모델을 다시 학습하지 않고, 빠르게 재학습 후 figure만 재생성.
출력: fig3_3way.png, fig5_selectivity.png, figS4_broad.png, fig6_training.png
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
try:
    from adjustText import adjust_text
    HAS_ADJ=True
except: HAS_ADJ=False

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM,POCKET_DIM,COMPOUND_DIM=256,2560,2060
RECEPTORS=['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7','TAS2R8','TAS2R9','TAS2R10','TAS2R13','TAS2R14','TAS2R16','TAS2R38','TAS2R39','TAS2R40','TAS2R41','TAS2R42','TAS2R43','TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']
BROAD=['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']

def PL(ax,lab):
    ax.annotate(lab, xy=(0,1), xycoords='axes fraction', xytext=(-30,12),
                textcoords='offset points', fontsize=15, fontweight='bold',
                va='bottom', ha='left', annotation_clip=False)

# ── 데이터 (run_production과 동일) ──
pocket_emb=np.load("tas2r_pocket_embeddings_v2.npy"); pocket_idx=json.load(open("tas2r_embedding_index_v2.json"))
fp_matrix=np.load("morgan_fingerprints_v2.npy"); fp_cids=pd.read_csv("fp_cid_index_v2.csv")['bdb_cid'].tolist()
desc_cols=['mw','alogp','tpsa','hbd','hba','rotb','rings','arom_rings','fsp3','formal_charge','heavy_atoms','globularity_proxy']
desc_df=pd.read_csv("compounds_with_descriptors_v2.csv").set_index('bdb_cid')
cid_to_idx={c:i for i,c in enumerate(fp_cids)}
draw=np.array([desc_df.loc[c,desc_cols].values if c in desc_df.index else np.zeros(12) for c in fp_cids],dtype=np.float32)
dmu,dsd=draw.mean(0),draw.std(0)+1e-8
compound_features={}
for c in fp_cids:
    if c in desc_df.index:
        compound_features[c]=np.concatenate([fp_matrix[cid_to_idx[c]].astype(np.float32),(desc_df.loc[c,desc_cols].values.astype(np.float32)-dmu)/dsd])
df_lr=pd.read_csv("ligandReceptors_2024.csv"); human=[1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
df_h=df_lr[df_lr['rID'].isin(human)].copy(); df_h['receptor']=df_h['rID'].map({r:f'TAS2R{r}' for r in human})
split=pd.read_csv("scaffold_split_v2.csv"); train_cids=set(split[split['split']=='train']['bdb_cid']); test_cids=set(split[split['split']=='test']['bdb_cid'])
annotated=set(df_h['cID'].unique())
def bp(cset):
    p=[]
    for rec in RECEPTORS:
        pos=set(df_h[df_h['receptor']==rec]['cID'].unique()); neg=annotated-pos
        for c in cset:
            if c in compound_features: p.append((c,rec,1 if c in pos else 0))
    return p
class DS(Dataset):
    def __init__(s,pr): s.d=[(torch.FloatTensor(compound_features[c]),torch.FloatTensor(pocket_emb[pocket_idx[r]]),pocket_idx[r],float(l)) for c,r,l in pr]
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i]
class SelectNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.comp=nn.Sequential(nn.Linear(COMPOUND_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,EMBED_DIM))
        s.rec=nn.Sequential(nn.Linear(POCKET_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,EMBED_DIM))
        s.W=nn.Parameter(torch.randn(EMBED_DIM,EMBED_DIM)*0.02); s.b=nn.Parameter(torch.zeros(23))
    def forward(s,c,p,ri): return (s.comp(c)*(s.rec(p)@s.W)).sum(-1)+s.b[ri]
class Focal(nn.Module):
    def __init__(s,g=2.0): super().__init__(); s.g=g
    def forward(s,lg,t): bce=F.binary_cross_entropy_with_logits(lg,t,reduction='none'); pt=torch.exp(-bce); return ((1-pt)**s.g*bce).mean()

torch.manual_seed(42); np.random.seed(42)
tr_dl=DataLoader(DS(bp(train_cids)),batch_size=128,shuffle=True); te_dl=DataLoader(DS(bp(test_cids)),batch_size=128,shuffle=False)
model=SelectNet().to(DEVICE); crit=Focal(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
curve=[]; best=-1; best_state=None
for stage,(eps,lr) in enumerate([(60,1e-3),(60,3e-4)]):
    for g in opt.param_groups: g['lr']=lr
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=eps)
    for ep in range(1,eps+1):
        model.train(); losses=[]
        for c,p,r,l in tr_dl:
            opt.zero_grad(); loss=crit(model(c.to(DEVICE),p.to(DEVICE),r.to(DEVICE)),l.to(DEVICE)); loss.backward(); opt.step(); losses.append(loss.item())
        sch.step()
        if ep%5==0:
            model.eval(); P,L,Rc=[],[],[]
            with torch.no_grad():
                for c,p,r,l in te_dl: P.extend(torch.sigmoid(model(c.to(DEVICE),p.to(DEVICE),r.to(DEVICE))).cpu().numpy()); L.extend(l.numpy()); Rc.extend(r.numpy())
            P,L,Rc=np.array(P),np.array(L),np.array(Rc)
            aucs=[roc_auc_score(L[Rc==ri],P[Rc==ri]) for ri in np.unique(Rc) if 0<L[Rc==ri].sum()<(Rc==ri).sum()]
            macro=np.mean(aucs); curve.append((stage*60+ep,macro,np.mean(losses)))
            if macro>best: best=macro; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
model.load_state_dict(best_state); model.eval()
print(f"Best macro AUC: {best:.3f}")

gt=json.load(open('ground_truth.json'))
sn=gt['per_subtype_selectnet']; scf=gt['per_subtype_onehot_scaffold']; rnd=gt['per_subtype_onehot_random']

# ════ fig6: training (A/B 레이블, 제목 제거) ════
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ex=[c[0] for c in curve]; ay=[c[1] for c in curve]; ly=[c[2] for c in curve]
ax[0].plot(ex,ay,'o-',color='#E8A7A0',ms=4,label='Macro AUC (scaffold)')
ax[0].axhline(gt['broad_n6']['onehot_scaffold'],ls=':',color='gray',label='one-hot scaffold (0.727)')
ib=int(np.argmax(ay)); ax[0].plot(ex[ib],ay[ib],'*',ms=16,color='#E8A7A0')
ax[0].annotate(f"Best {ay[ib]:.3f}",(ex[ib],ay[ib]),textcoords="offset points",xytext=(5,8),fontsize=10,weight='bold')
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Macro AUC-ROC (scaffold split)'); ax[0].legend(fontsize=8); PL(ax[0],'A')
ax[1].semilogy(ex,ly,'o-',color='#8FB8DC',ms=4); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Focal loss (log scale)'); PL(ax[1],'B')
plt.tight_layout(); plt.savefig('fig6_training.png',dpi=200,bbox_inches='tight'); plt.close()
print("fig6 OK")

# ════ fig5: selectivity (A/B/C/D, D 실데이터) ════
all_cids=[c for c in fp_cids if c in compound_features]
all_feat=np.array([compound_features[c] for c in all_cids])
with torch.no_grad():
    allP=[]
    for i in range(0,len(all_feat),256):
        b=torch.FloatTensor(all_feat[i:i+256]).to(DEVICE); bp_=[]
        for r in RECEPTORS:
            pe=torch.FloatTensor(pocket_emb[pocket_idx[r]]).unsqueeze(0).repeat(len(b),1).to(DEVICE)
            ri=torch.full((len(b),),pocket_idx[r],dtype=torch.long).to(DEVICE)
            bp_.append(torch.sigmoid(model(b,pe,ri)).cpu().numpy())
        allP.append(np.stack(bp_,axis=1))
    allP=np.concatenate(allP)
SI=allP/allP.mean(axis=1,keepdims=True)
# near-orphan 후보 cid 수집
ORPHAN=['TAS2R2','TAS2R9','TAS2R13']  # near-orphan only (<10 ligands); R49,R50 are MED-tier data-sparse
orphan_pts={r:[] for r in ORPHAN}
mono_per={r:0 for r in RECEPTORS}; total=0
for ci in range(len(allP)):
    hits=[j for j in range(23) if allP[ci,j]>0.5 and SI[ci,j]>1.5]
    if len(hits)==1:
        r=RECEPTORS[hits[0]]; mono_per[r]+=1; total+=1
        if r in ORPHAN:
            cid=all_cids[ci]
            if cid in desc_df.index:
                orphan_pts[r].append((desc_df.loc[cid,'mw'],desc_df.loc[cid,'alogp']))

fig,ax=plt.subplots(2,2,figsize=(14,10))
# A: SI 분포
sel_si=SI[(allP>0.5)&(SI>1.5)]
ax[0,0].hist(sel_si[sel_si<25],bins=40,color='#E8A7A0',alpha=0.65,edgecolor='white',lw=0.3)
ax[0,0].axvline(1.5,ls='--',color='gray'); ax[0,0].set_xlabel('Selectivity Index (SI)'); ax[0,0].set_ylabel('Count'); PL(ax[0,0],'A')
# B: candidate per subtype
ps={k:v for k,v in sorted(mono_per.items(),key=lambda x:-x[1]) if v>0}
recs=list(ps.keys()); vals=list(ps.values())
tcol={r:'#E8A7A0' for r in BROAD}
ax[0,1].bar(range(len(recs)),vals,color=[tcol.get(r,'#A8CCE5') for r in recs],edgecolor='#555')
ax[0,1].set_xticks(range(len(recs))); ax[0,1].set_xticklabels([r.replace('TAS2R','T') for r in recs],rotation=45,fontsize=8)
ax[0,1].set_ylabel('Monoselective candidates'); PL(ax[0,1],'B')
# C: prob vs SI
m=allP>0.5
ax[1,0].scatter(SI[m][SI[m]<25],allP[m][SI[m]<25],s=5,alpha=0.3,color='gray')
ax[1,0].axhline(0.5,ls='--',color='gray',alpha=0.5); ax[1,0].axvline(1.5,ls='--',color='gray',alpha=0.5)
ax[1,0].set_xlabel('Selectivity Index (SI)'); ax[1,0].set_ylabel(r'Predicted probability $\hat{p}$'); PL(ax[1,0],'C')
# D: near-orphan MW vs AlogP (실데이터!)
ocol={'TAS2R2':'#A8CCE5','TAS2R9':'#F2C49B','TAS2R13':'#A8D5BA','TAS2R49':'#F0A8A0','TAS2R50':'#C9A8D8'}
for r in ORPHAN:
    if orphan_pts[r]:
        mws=[p[0] for p in orphan_pts[r]]; alps=[p[1] for p in orphan_pts[r]]
        ax[1,1].scatter(mws,alps,c=ocol[r],s=70,edgecolor='#555',lw=0.5,label=r,zorder=3)
ax[1,1].axhline(5,ls=':',color='gray',alpha=0.6); ax[1,1].axvline(500,ls=':',color='gray',alpha=0.6)
ax[1,1].text(505,0.2,'Lipinski limits',fontsize=8,color='gray')
ax[1,1].set_xlabel('Molecular Weight (Da)'); ax[1,1].set_ylabel('AlogP')
if any(orphan_pts.values()): ax[1,1].legend(title='Receptor',fontsize=8)
PL(ax[1,1],'D')
plt.tight_layout(); plt.savefig('fig5_selectivity.png',dpi=200,bbox_inches='tight'); plt.close()
print(f"fig5 OK (near-orphan 후보점: {sum(len(v) for v in orphan_pts.values())}개)")

# ════ fig3, figS4 (ground_truth만으로) ════
order=['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10','TAS2R43','TAS2R38','TAS2R5','TAS2R16','TAS2R7','TAS2R47','TAS2R40','TAS2R8','TAS2R44','TAS2R2','TAS2R13']
order=[r for r in order if r in sn and sn[r]['auc'] is not None]
fig,ax=plt.subplots(1,3,figsize=(17,5)); x=np.arange(len(order)); w=0.27
ax[0].bar(x-w,[rnd.get(r,0) or 0 for r in order],w,label='Random (one-hot)',color='#D9D9D9')
ax[0].bar(x,[scf.get(r,0) or 0 for r in order],w,label='Scaffold (one-hot)',color='#A8CCE5')
ax[0].bar(x+w,[sn[r]['auc'] for r in order],w,label='Scaffold (SelectNet)',color='#E8A7A0')
ax[0].set_xticks(x); ax[0].set_xticklabels([r.replace('TAS2R','') for r in order],fontsize=8)
ax[0].axhline(0.7,ls='--',color='gray',alpha=0.5); ax[0].set_ylabel('AUC-ROC'); ax[0].set_ylim(0.2,1.05); ax[0].set_xlabel('TAS2R subtype'); ax[0].legend(fontsize=8,loc='lower left'); PL(ax[0],'A')
gains=[(sn[r]['auc']-(scf.get(r,0) or 0)) for r in order]; sg=sorted(zip(order,gains),key=lambda v:-v[1]); so=[s[0] for s in sg]; sgv=[s[1] for s in sg]
ax[1].bar(range(len(so)),sgv,color=['#A8D5BA' if g>0 else '#F0A8A0' for g in sgv]); ax[1].axhline(0,color='black',lw=0.8)
ax[1].set_xticks(range(len(so))); ax[1].set_xticklabels([r.replace('TAS2R','T') for r in so],rotation=45,fontsize=8); ax[1].set_xlabel('TAS2R subtype'); ax[1].set_ylabel(r'$\Delta$AUC (SelectNet $-$ one-hot scaffold)'); PL(ax[1],'B')
ax[2].plot([0.25,1.05],[0.25,1.05],'--',color='gray',alpha=0.5); texts=[]
for r in order:
    xx=scf.get(r,0) or 0; yy=sn[r]['auc']
    ax[2].scatter(xx,yy,c={rr:'#E8A7A0' for rr in BROAD}.get(r,'#A8CCE5'),s=60,edgecolor='#555',lw=0.5,zorder=3)
    texts.append(ax[2].text(xx,yy,r.replace('TAS2R','T'),fontsize=8))
if HAS_ADJ: adjust_text(texts,ax=ax[2],arrowprops=dict(arrowstyle='-',color='gray',lw=0.5))
ax[2].set_xlabel('AUC — Scaffold One-Hot'); ax[2].set_ylabel('AUC — Scaffold SelectNet'); ax[2].set_xlim(0.25,1.08); ax[2].set_ylim(0.25,1.08); PL(ax[2],'C')
plt.tight_layout(); plt.savefig('fig3_3way.png',dpi=200,bbox_inches='tight'); plt.close()
print("fig3 OK")

fig,ax=plt.subplots(1,2,figsize=(13,4.8)); x=np.arange(len(BROAD)); w=0.27
ax[0].bar(x-w,[rnd.get(r,0) or 0 for r in BROAD],w,label='Random (one-hot)',color='#D9D9D9')
ax[0].bar(x,[scf.get(r,0) or 0 for r in BROAD],w,label='Scaffold (one-hot)',color='#A8CCE5')
ax[0].bar(x+w,[sn[r]['auc'] or 0 for r in BROAD],w,label='Scaffold (SelectNet)',color='#E8A7A0')
ax[0].set_xticks(x); ax[0].set_xticklabels(BROAD,rotation=30,ha='right',fontsize=8); ax[0].set_ylabel('AUC-ROC'); ax[0].set_ylim(0.5,1.05); ax[0].legend(fontsize=8); PL(ax[0],'A')
means=[gt['broad_n6']['onehot_random'],gt['broad_n6']['onehot_scaffold'],gt['broad_n6']['selectnet']]
bars=ax[1].bar(['Random\n(one-hot)','Scaffold\n(one-hot)','Scaffold\n(SelectNet)'],means,color=['#D9D9D9','#A8CCE5','#E8A7A0'],edgecolor='#555')
for b,mv in zip(bars,means): ax[1].text(b.get_x()+b.get_width()/2,mv+0.005,f'{mv:.3f}',ha='center',weight='bold')
ax[1].set_ylabel('Mean AUC-ROC (broadly-tuned)'); ax[1].set_ylim(0.6,0.9); PL(ax[1],'B')
plt.tight_layout(); plt.savefig('figS4_broad.png',dpi=200,bbox_inches='tight'); plt.close()
print("figS4 OK")
print("\n전체 figure 재생성 완료 (A/B 레이블 + 제목제거 + fig5D 실데이터)")
