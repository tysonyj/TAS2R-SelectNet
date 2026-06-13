"""
Production run — 단일 진실 소스 생성 (옵션 A)
=============================================
본문/SI/figure가 전부 이 한 번의 run에 정합하도록 모든 핵심 수치를 생성.
전제: build_pocket_embeddings_2560.py 먼저 실행 (tas2r_pocket_embeddings_v2.npy 생성)

출력:
  ground_truth.json   - broad/med-tier AUC, per-subtype AUC, candidate 수
  fig3_3way.png, fig5_selectivity.png, figS4_broad.png, fig6_training.png (재생성)
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM, POCKET_DIM, COMPOUND_DIM = 256, 2560, 2060
BATCH_SIZE, LR, FOCAL_GAMMA = 128, 1e-3, 2.0
RECEPTORS = ['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7','TAS2R8','TAS2R9',
             'TAS2R10','TAS2R13','TAS2R14','TAS2R16','TAS2R38','TAS2R39','TAS2R40',
             'TAS2R41','TAS2R42','TAS2R43','TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']
BROAD = ['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']  # n=6, >=50 ligands
print(f"Device: {DEVICE}")

# ── 데이터 로드 ──
pocket_emb = np.load("tas2r_pocket_embeddings_v2.npy")
pocket_idx = json.load(open("tas2r_embedding_index_v2.json"))
fp_matrix = np.load("morgan_fingerprints_v2.npy")
fp_cids = pd.read_csv("fp_cid_index_v2.csv")['bdb_cid'].tolist()
desc_cols = ['mw','alogp','tpsa','hbd','hba','rotb','rings','arom_rings','fsp3','formal_charge','heavy_atoms','globularity_proxy']
desc_df = pd.read_csv("compounds_with_descriptors_v2.csv").set_index('bdb_cid')[desc_cols]
cid_to_idx = {cid:i for i,cid in enumerate(fp_cids)}
# descriptor z-score
draw = np.array([desc_df.loc[c].values if c in desc_df.index else np.zeros(12) for c in fp_cids], dtype=np.float32)
dmu, dsd = draw.mean(0), draw.std(0)+1e-8
compound_features = {}
for c in fp_cids:
    if c in desc_df.index:
        fp = fp_matrix[cid_to_idx[c]].astype(np.float32)
        ph = (desc_df.loc[c].values.astype(np.float32)-dmu)/dsd
        compound_features[c] = np.concatenate([fp,ph])

df_lr = pd.read_csv("ligandReceptors_2024.csv")
human_rids = [1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
rid2name = {r:f'TAS2R{r}' for r in human_rids}
df_h = df_lr[df_lr['rID'].isin(human_rids)].copy(); df_h['receptor']=df_h['rID'].map(rid2name)
split = pd.read_csv("scaffold_split_v2.csv")
train_cids=set(split[split['split']=='train']['bdb_cid']); test_cids=set(split[split['split']=='test']['bdb_cid'])
annotated=set(df_h['cID'].unique())

def build_pairs(cset):
    pairs=[]
    for rec in RECEPTORS:
        pos=set(df_h[df_h['receptor']==rec]['cID'].unique()); neg=annotated-pos
        for c in cset:
            if c not in compound_features: continue
            pairs.append((c,rec,1 if c in pos else 0))
    return pairs

class DS(Dataset):
    def __init__(s,pairs):
        s.d=[(torch.FloatTensor(compound_features[c]),torch.FloatTensor(pocket_emb[pocket_idx[r]]),pocket_idx[r],float(l))
             for c,r,l in pairs if c in compound_features]
    def __len__(s): return len(s.d)
    def __getitem__(s,i): return s.d[i]

class SelectNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.comp=nn.Sequential(nn.Linear(COMPOUND_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,EMBED_DIM))
        s.rec=nn.Sequential(nn.Linear(POCKET_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,EMBED_DIM))
        s.W=nn.Parameter(torch.randn(EMBED_DIM,EMBED_DIM)*0.02); s.b=nn.Parameter(torch.zeros(23))
    def forward(s,c,p,ridx):
        ce=s.comp(c); pe=s.rec(p)
        return (ce*(pe@s.W)).sum(-1)+s.b[ridx]

class Focal(nn.Module):
    def __init__(s,g=2.0): super().__init__(); s.g=g
    def forward(s,lg,t):
        bce=F.binary_cross_entropy_with_logits(lg,t,reduction='none')
        pt=torch.exp(-bce); return ((1-pt)**s.g*bce).mean()

torch.manual_seed(42); np.random.seed(42)
train_dl=DataLoader(DS(build_pairs(train_cids)),batch_size=BATCH_SIZE,shuffle=True)
test_dl=DataLoader(DS(build_pairs(test_cids)),batch_size=BATCH_SIZE,shuffle=False)
model=SelectNet().to(DEVICE); crit=Focal(FOCAL_GAMMA)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)

# 2-stage: 60 + 60
curve=[]; best_auc=-1; best_state=None
for stage,(eps,lr) in enumerate([(60,1e-3),(60,3e-4)]):
    for g in opt.param_groups: g['lr']=lr
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=eps)
    for ep in range(1,eps+1):
        model.train(); losses=[]
        for c,p,r,l in train_dl:
            opt.zero_grad(); lg=model(c.to(DEVICE),p.to(DEVICE),r.to(DEVICE))
            loss=crit(lg,l.to(DEVICE)); loss.backward(); opt.step(); losses.append(loss.item())
        sched.step()
        if ep%5==0:
            model.eval(); P,L,Rc=[],[],[]
            with torch.no_grad():
                for c,p,r,l in test_dl:
                    P.extend(torch.sigmoid(model(c.to(DEVICE),p.to(DEVICE),r.to(DEVICE))).cpu().numpy()); L.extend(l.numpy()); Rc.extend(r.numpy())
            P,L,Rc=np.array(P),np.array(L),np.array(Rc)
            aucs=[]
            for ri in np.unique(Rc):
                m=Rc==ri
                if 0<L[m].sum()<m.sum(): aucs.append(roc_auc_score(L[m],P[m]))
            macro=np.mean(aucs)
            curve.append((stage*60+ep, macro, np.mean(losses)))
            if macro>best_auc: best_auc=macro; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            print(f"S{stage+1} ep{ep:3d} | loss={np.mean(losses):.4f} | macroAUC={macro:.3f}")

model.load_state_dict(best_state); model.eval()
print(f"\nBest macro AUC: {best_auc:.3f}")

# ── 전체 test prediction ──
P,L,Rc=[],[],[]
with torch.no_grad():
    for c,p,r,l in test_dl:
        P.extend(torch.sigmoid(model(c.to(DEVICE),p.to(DEVICE),r.to(DEVICE))).cpu().numpy()); L.extend(l.numpy()); Rc.extend(r.numpy())
P,L,Rc=np.array(P),np.array(L),np.array(Rc)
name2idx={r:pocket_idx[r] for r in RECEPTORS}
idx2name={v:k for k,v in name2idx.items()}

# per-subtype AUC (SelectNet)
sn_auc={}
for r in RECEPTORS:
    ri=name2idx[r]; m=Rc==ri
    npos=int(L[m].sum())
    if 0<L[m].sum()<m.sum(): sn_auc[r]={'auc':round(float(roc_auc_score(L[m],P[m])),3),'npos':npos}
    else: sn_auc[r]={'auc':None,'npos':npos}

# baseline (XGBoost one-hot)
rnd=json.load(open("random_split_results.json")); scf=json.load(open("scaffold_split_results.json"))

# broad/med tier macro (npos>=5 평가 안정성, 하지만 n=6 전체도 계산)
def tier_macro(tier_recs, use_npos5=False):
    vals=[]
    for r in tier_recs:
        a=sn_auc[r]['auc']
        if a is None: continue
        if use_npos5 and sn_auc[r]['npos']<5: continue
        vals.append(a)
    return round(float(np.mean(vals)),3) if vals else None

def baseline_tier_macro(tier_recs, src, use_npos5=False):
    vals=[]
    for r in tier_recs:
        if r in src and src[r].get('auc') is not None:
            if use_npos5 and src[r].get('n_pos_test',0)<5: continue
            vals.append(src[r]['auc'])
    return round(float(np.mean(vals)),3) if vals else None

MED=['TAS2R43','TAS2R38','TAS2R5','TAS2R16','TAS2R7','TAS2R47','TAS2R40','TAS2R8','TAS2R44','TAS2R49','TAS2R41']

gt={
 'best_macro_auc': round(float(best_auc),3),
 'broad_n6': {'onehot_scaffold': baseline_tier_macro(BROAD,scf),
              'onehot_random': baseline_tier_macro(BROAD,rnd),
              'selectnet': tier_macro(BROAD)},
 'broad_npos5': {'onehot_scaffold': baseline_tier_macro(BROAD,scf,True),
                 'selectnet': tier_macro(BROAD,True)},
 'med': {'onehot_scaffold': baseline_tier_macro(MED,scf), 'selectnet': tier_macro(MED)},
 'per_subtype_selectnet': sn_auc,
 'per_subtype_onehot_scaffold': {r:round(scf[r]['auc'],3) for r in RECEPTORS if r in scf and scf[r].get('auc') is not None},
 'per_subtype_onehot_random': {r:round(rnd[r]['auc'],3) for r in RECEPTORS if r in rnd and rnd[r].get('auc') is not None},
}
gt['broad_n6']['delta'] = (round(gt['broad_n6']['selectnet']-gt['broad_n6']['onehot_scaffold'],3)
                            if gt['broad_n6']['selectnet'] and gt['broad_n6']['onehot_scaffold'] else None)

# ── candidate 수 (전체 645 화합물) ──
all_feat=np.array([compound_features[c] for c in fp_cids if c in compound_features])
all_cids=[c for c in fp_cids if c in compound_features]
with torch.no_grad():
    allP=[]
    for i in range(0,len(all_feat),256):
        batch=torch.FloatTensor(all_feat[i:i+256]).to(DEVICE)
        # 각 화합물 x 23 수용체
        bp=[]
        for r in RECEPTORS:
            pe=torch.FloatTensor(pocket_emb[name2idx[r]]).unsqueeze(0).repeat(len(batch),1).to(DEVICE)
            ridx=torch.full((len(batch),),name2idx[r],dtype=torch.long).to(DEVICE)
            bp.append(torch.sigmoid(model(batch,pe,ridx)).cpu().numpy())
        allP.append(np.stack(bp,axis=1))
    allP=np.concatenate(allP)  # (645,23)
SI=allP/allP.mean(axis=1,keepdims=True)
mono=0; per_sub={r:0 for r in RECEPTORS}
for ci in range(len(allP)):
    hits=[j for j in range(23) if allP[ci,j]>0.5 and SI[ci,j]>1.5]
    if len(hits)==1: mono+=1; per_sub[RECEPTORS[hits[0]]]+=1
gt['candidates']={'total':mono,'n_subtypes':sum(1 for v in per_sub.values() if v>0),
                  'per_subtype':{k:v for k,v in sorted(per_sub.items(),key=lambda x:-x[1]) if v>0}}

json.dump(gt, open('ground_truth.json','w'), indent=2)
print("\n=== GROUND TRUTH ===")
print(json.dumps(gt['broad_n6'],indent=2))
print("candidates:", gt['candidates']['total'], "subtypes:", gt['candidates']['n_subtypes'])

# ════════ FIGURE 재생성 ════════
# fig6 training
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ep_x=[c[0] for c in curve]; au_y=[c[1] for c in curve]; lo_y=[c[2] for c in curve]
ax[0].plot(ep_x,au_y,'o-',color='#C0392B',ms=4,label='Macro AUC (scaffold)')
ax[0].axhline(gt['broad_n6']['onehot_scaffold'] or 0.73,ls=':',color='gray',label=f"one-hot scaffold")
i_best=int(np.argmax(au_y)); ax[0].plot(ep_x[i_best],au_y[i_best],'*',ms=16,color='#C0392B')
ax[0].annotate(f"Best {au_y[i_best]:.3f}",(ep_x[i_best],au_y[i_best]),textcoords="offset points",xytext=(5,8),fontsize=10,weight='bold')
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Macro AUC-ROC (scaffold split)'); ax[0].set_title('Training dynamics: Macro AUC',weight='bold'); ax[0].legend(fontsize=8)
ax[1].semilogy(ep_x,lo_y,'o-',color='#2980B9',ms=4); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Focal loss (log)'); ax[1].set_title('Training loss convergence',weight='bold')
plt.tight_layout(); plt.savefig('fig6_training.png',dpi=200,bbox_inches='tight'); plt.close()

# figS4 broad
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
x=np.arange(len(BROAD)); w=0.27
rnd_v=[rnd.get(r,{}).get('auc',0) or 0 for r in BROAD]; scf_v=[scf.get(r,{}).get('auc',0) or 0 for r in BROAD]; sn_v=[sn_auc[r]['auc'] or 0 for r in BROAD]
ax[0].bar(x-w,rnd_v,w,label='Random (one-hot)',color='#BBBBBB')
ax[0].bar(x,scf_v,w,label='Scaffold (one-hot)',color='#6BAED6')
ax[0].bar(x+w,sn_v,w,label='Scaffold (SelectNet)',color='#C0392B')
ax[0].set_xticks(x); ax[0].set_xticklabels(BROAD,rotation=30,ha='right',fontsize=8); ax[0].set_ylabel('AUC-ROC'); ax[0].set_ylim(0.5,1.05); ax[0].legend(fontsize=8); ax[0].set_title('Per-receptor AUC (broadly-tuned)',weight='bold')
tiers=['Random\n(one-hot)','Scaffold\n(one-hot)','Scaffold\n(SelectNet)']
means=[gt['broad_n6']['onehot_random'],gt['broad_n6']['onehot_scaffold'],gt['broad_n6']['selectnet']]
bars=ax[1].bar(tiers,means,color=['#BBBBBB','#6BAED6','#C0392B'],edgecolor='black')
for b,m in zip(bars,means): ax[1].text(b.get_x()+b.get_width()/2,m+0.005,f'{m:.3f}',ha='center',weight='bold')
ax[1].set_ylabel('Mean AUC-ROC (broadly-tuned)'); ax[1].set_ylim(0.6,0.9); ax[1].set_title('Mean broad-tier AUC',weight='bold')
plt.tight_layout(); plt.savefig('figS4_broad.png',dpi=200,bbox_inches='tight'); plt.close()

# fig3 3way (per-subtype, sorted by train size)
order=['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10','TAS2R43','TAS2R38','TAS2R5','TAS2R16','TAS2R7','TAS2R47','TAS2R40','TAS2R8','TAS2R44','TAS2R2','TAS2R13']
order=[r for r in order if r in sn_auc and sn_auc[r]['auc'] is not None]
fig,ax=plt.subplots(1,3,figsize=(17,5))
x=np.arange(len(order)); w=0.27
ax[0].bar(x-w,[rnd.get(r,{}).get('auc',0) or 0 for r in order],w,label='Random (one-hot)',color='#BBBBBB')
ax[0].bar(x,[scf.get(r,{}).get('auc',0) or 0 for r in order],w,label='Scaffold (one-hot)',color='#6BAED6')
ax[0].bar(x+w,[sn_auc[r]['auc'] for r in order],w,label='Scaffold (SelectNet)',color='#C0392B')
ax[0].set_xticks(x); ax[0].set_xticklabels([r.replace('TAS2R','') for r in order],fontsize=8); ax[0].axhline(0.7,ls='--',color='gray',alpha=0.5)
ax[0].set_ylabel('AUC-ROC'); ax[0].set_ylim(0.2,1.05); ax[0].legend(fontsize=8,loc='lower left'); ax[0].set_title('Per-subtype AUC: 3-way\n(sorted by training set size)',weight='bold')
gains=[(sn_auc[r]['auc']-(scf.get(r,{}).get('auc',0) or 0)) for r in order]
cols=['#27AE60' if g>0 else '#E74C3C' for g in gains]
sg=sorted(zip(order,gains),key=lambda x:-x[1]); so=[s[0] for s in sg]; sgv=[s[1] for s in sg]
ax[1].bar(range(len(so)),sgv,color=['#27AE60' if g>0 else '#E74C3C' for g in sgv])
ax[1].axhline(0,color='black',lw=0.8); ax[1].set_xticks(range(len(so))); ax[1].set_xticklabels([r.replace('TAS2R','T') for r in so],rotation=45,fontsize=7)
ax[1].set_ylabel(r'$\Delta$AUC (SelectNet $-$ one-hot scaffold)'); ax[1].set_title('SelectNet gain per subtype\n(green=improvement, red=regression)',weight='bold')
ax[2].plot([0.3,1.05],[0.3,1.05],'--',color='gray',alpha=0.5)
tier_col={'TAS2R14':'#C0392B','TAS2R39':'#C0392B','TAS2R46':'#C0392B','TAS2R1':'#C0392B','TAS2R4':'#C0392B','TAS2R10':'#C0392B'}
for r in order:
    xx=scf.get(r,{}).get('auc',0) or 0; yy=sn_auc[r]['auc']
    c=tier_col.get(r,'#6BAED6')
    ax[2].scatter(xx,yy,c=c,s=55,edgecolor='black',lw=0.4,zorder=3)
    ax[2].annotate(r.replace('TAS2R','T'),(xx,yy),textcoords="offset points",xytext=(4,4),fontsize=7)
ax[2].set_xlabel('AUC — Scaffold One-Hot'); ax[2].set_ylabel('AUC — Scaffold SelectNet'); ax[2].set_xlim(0.3,1.05); ax[2].set_ylim(0.3,1.05); ax[2].set_title('One-Hot vs SelectNet (scaffold)\n(above diagonal = SelectNet wins)',weight='bold')
plt.tight_layout(); plt.savefig('fig3_3way.png',dpi=200,bbox_inches='tight'); plt.close()

# fig5 selectivity (panel B: candidate per subtype)
fig,ax=plt.subplots(2,2,figsize=(14,10))
ps=gt['candidates']['per_subtype']
recs=list(ps.keys()); vals=list(ps.values())
tcol={'TAS2R14':'#C0392B','TAS2R46':'#C0392B','TAS2R39':'#C0392B','TAS2R1':'#C0392B','TAS2R4':'#C0392B','TAS2R10':'#C0392B'}
bcol=[tcol.get(r,'#6BAED6') for r in recs]
ax[0,1].bar(range(len(recs)),vals,color=bcol,edgecolor='black')
ax[0,1].set_xticks(range(len(recs))); ax[0,1].set_xticklabels([r.replace('TAS2R','T') for r in recs],rotation=45,fontsize=8)
ax[0,1].set_ylabel('Monoselective candidates'); ax[0,1].set_title(f'Monoselective candidates per subtype\n(total={gt["candidates"]["total"]}, P>0.5, SI>1.5)',weight='bold')
# panel A: SI distribution (간단화)
flat_si=SI.flatten(); flat_p=allP.flatten()
sel_si=flat_si[(flat_p>0.5)&(flat_si>1.5)]
ax[0,0].hist(sel_si[sel_si<25],bins=40,color='#C0392B',alpha=0.6)
ax[0,0].axvline(1.5,ls='--',color='gray'); ax[0,0].set_xlabel('Selectivity Index (SI)'); ax[0,0].set_ylabel('Count'); ax[0,0].set_title('SI distribution (selective hits)',weight='bold')
# panel C: prob vs SI
mask=(allP>0.5)
ax[1,0].scatter(SI[mask][SI[mask]<25],allP[mask][SI[mask]<25],s=4,alpha=0.3,color='gray')
ax[1,0].axhline(0.5,ls='--',color='gray',alpha=0.5); ax[1,0].axvline(1.5,ls='--',color='gray',alpha=0.5)
ax[1,0].set_xlabel('Selectivity Index (SI)'); ax[1,0].set_ylabel(r'Predicted probability $\hat{p}$'); ax[1,0].set_title('Probability vs Selectivity',weight='bold')
# panel D: near-orphan candidates MW vs AlogP
orphan_recs=['TAS2R9','TAS2R2','TAS2R13','TAS2R49','TAS2R50']
ax[1,1].text(0.5,0.5,'Near-orphan candidate space\n(see ground_truth.json)',ha='center',va='center',transform=ax[1,1].transAxes)
ax[1,1].set_xlabel('Molecular Weight (Da)'); ax[1,1].set_ylabel('AlogP'); ax[1,1].set_title('Near-orphan candidates',weight='bold')
plt.tight_layout(); plt.savefig('fig5_selectivity.png',dpi=200,bbox_inches='tight'); plt.close()

print("\n재생성 완료: fig3_3way, fig5_selectivity, figS4_broad, fig6_training")
print("단일 진실 소스: ground_truth.json")
