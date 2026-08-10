"""
Leave-one-receptor-out CV with additional representation controls
(Reviewer 1, round 2)
==================================================================
The round-1 LORO experiment compared ESM-2 pocket embeddings against a
nearest-sequence one-hot baseline, and found a modest advantage for receptors
without a close training paralogue. The reviewer's random-residue control raises
the obvious follow-up: is that transfer advantage specific to the POCKET, or would
any ESM-2-derived receptor vector do just as well?

This script repeats the LORO protocol for four representations of the held-out
receptor:
  pocket            ESM-2 pocket mean pool
  random_nonpocket  ESM-2 embedding of ONE random residue NOT in the pocket
  fullseq           ESM-2 full-sequence mean pool
  nn_onehot         one-hot identity of the most sequence-similar training receptor

All are tiled to the same input width, and everything else (encoder, head, loss,
schedule, seed) is identical. If random_nonpocket and fullseq transfer as well as
pocket, the transfer benefit is not pocket-specific and must be described as such.

Requires GPU + fair-esm. Output: loro_representations.json
"""
import json, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM, COMPOUND_DIM, COMMON_DIM = 256, 2060, 2560
BATCH_SIZE, FOCAL_GAMMA = 128, 2.0
RECEPTORS = ['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7','TAS2R8','TAS2R9',
             'TAS2R10','TAS2R13','TAS2R14','TAS2R16','TAS2R38','TAS2R39','TAS2R40',
             'TAS2R41','TAS2R42','TAS2R43','TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']
N_REC = len(RECEPTORS)

_C = {}
def esm_per_residue(seq, rec):
    if 'm' not in _C:
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        bc = alphabet.get_batch_converter(); model = model.eval().to(DEVICE)
        if DEVICE == 'cuda': model = model.half()
        _C['m'] = (model, bc); _C['cache'] = {}
    if rec in _C['cache']: return _C['cache'][rec]
    model, bc = _C['m']
    with torch.no_grad():
        _,_,toks = bc([(rec, seq)]); toks = toks.to(DEVICE)
        rep = model(toks, repr_layers=[33])["representations"][33][0]
        pr = rep[1:len(seq)+1].float().cpu().numpy()
    _C['cache'][rec] = pr
    return pr

def tile_to(v, width=COMMON_DIM):
    reps = int(np.ceil(width/len(v)))
    return np.tile(v, reps)[:width].astype(np.float32)

def seq_identity(a, b):
    n, m = len(a), len(b)
    dp = np.zeros((n+1, m+1), dtype=np.int32)
    for i in range(1, n+1): dp[i,0] = -i
    for j in range(1, m+1): dp[0,j] = -j
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            dp[i,j] = max(dp[i-1,j-1] + (1 if ai==b[j-1] else 0), dp[i-1,j]-1, dp[i,j-1]-1)
    i, j, ident, alen = n, m, 0, 0
    while i>0 and j>0:
        if dp[i,j] == dp[i-1,j-1] + (1 if a[i-1]==b[j-1] else 0):
            if a[i-1]==b[j-1]: ident += 1
            i-=1; j-=1; alen+=1
        elif dp[i,j] == dp[i-1,j]-1: i-=1; alen+=1
        else: j-=1; alen+=1
    return ident/max(alen+i+j, 1)

def nearest_training(held, train_recs, seqs):
    best, bid = None, -1
    for r in train_recs:
        v = seq_identity(seqs[held], seqs[r])
        if v > bid: bid, best = v, r
    return best, bid


def build_reps(mode, seed=0):
    """(23, COMMON_DIM) receptor matrix for a representation mode."""
    seqs = json.load(open('tas2r_sequences_bdb.json'))
    out = np.zeros((N_REC, COMMON_DIM), dtype=np.float32)
    if mode == 'nn_onehot':
        for i in range(N_REC):
            v = np.zeros(N_REC, dtype=np.float32); v[i] = 1.0
            out[i] = tile_to(v)
        return out
    pocket = json.load(open('tas2r_pocket_definitions_v2.json'))
    rng = np.random.default_rng(seed)
    for i, rec in enumerate(RECEPTORS):
        s = seqs[rec]; pr = esm_per_residue(s, rec)
        pos = sorted(set(p-1 for p in (list(pocket[rec].get('orthosteric',[]))+
                                       list(pocket[rec].get('intracellular',[])))
                         if 1 <= p <= len(s)))
        if mode == 'pocket':
            v = pr[pos].mean(0) if pos else pr.mean(0)
        elif mode == 'fullseq':
            v = pr.mean(0)
        elif mode == 'random_nonpocket':
            nonp = [j for j in range(len(s)) if j not in set(pos)]
            v = pr[rng.choice(nonp)]
        else:
            raise ValueError(mode)
        out[i] = tile_to(v)
    return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.comp = nn.Sequential(nn.Linear(COMPOUND_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,EMBED_DIM))
        self.rec  = nn.Sequential(nn.Linear(COMMON_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,EMBED_DIM))
        self.W = nn.Parameter(torch.randn(EMBED_DIM,EMBED_DIM)*0.02); self.b = nn.Parameter(torch.zeros(N_REC))
    def forward(self,c,p,r):
        return (self.comp(c)*(self.rec(p)@self.W)).sum(-1) + self.b[r]

class Focal(nn.Module):
    def __init__(s,g=2.0): super().__init__(); s.g=g
    def forward(s,lg,t):
        bce=F.binary_cross_entropy_with_logits(lg,t,reduction='none'); pt=torch.exp(-bce)
        return ((1-pt)**s.g*bce).mean()


def main(epochs, seed):
    seqs = json.load(open('tas2r_sequences_bdb.json'))
    fp = np.load("morgan_fingerprints_v2.npy")
    cids = pd.read_csv("fp_cid_index_v2.csv")['bdb_cid'].tolist()
    dcols = ['mw','alogp','tpsa','hbd','hba','rotb','rings','arom_rings','fsp3','formal_charge','heavy_atoms','globularity_proxy']
    dd = pd.read_csv("compounds_with_descriptors_v2.csv").set_index('bdb_cid')[dcols]
    c2i = {c:i for i,c in enumerate(cids)}
    raw = np.array([dd.loc[c].values if c in dd.index else np.zeros(12) for c in cids], dtype=np.float32)
    mu, sd = raw.mean(0), raw.std(0)+1e-8
    feat = {c: np.concatenate([fp[c2i[c]].astype(np.float32), (dd.loc[c].values.astype(np.float32)-mu)/sd])
            for c in cids if c in dd.index}
    lr = pd.read_csv("ligandReceptors_2024.csv")
    human=[1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
    lr = lr[lr['rID'].isin(human)].copy(); lr['receptor']=lr['rID'].map({r:f'TAS2R{r}' for r in human})
    annotated=[c for c in lr['cID'].unique() if c in feat]
    pos_sets={r:set(lr[lr['receptor']==r]['cID'].unique()) for r in RECEPTORS}
    ridx={r:i for i,r in enumerate(RECEPTORS)}

    MODES = ['pocket','random_nonpocket','fullseq','nn_onehot']
    REPS = {m: build_reps(m, seed=0) for m in MODES}

    def run(held, mode):
        train_recs=[r for r in RECEPTORS if r!=held]
        R = REPS[mode]
        C,P,RI,Y=[],[],[],[]
        for r in train_recs:
            i=ridx[r]
            for c in annotated:
                C.append(feat[c]); P.append(R[i]); RI.append(i)
                Y.append(1.0 if c in pos_sets[r] else 0.0)
        C=torch.FloatTensor(np.array(C)); P=torch.FloatTensor(np.array(P))
        RI=torch.LongTensor(RI); Y=torch.FloatTensor(Y)
        torch.manual_seed(seed); np.random.seed(seed)
        m=Net().to(DEVICE); crit=Focal(FOCAL_GAMMA)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
        n=len(Y); idx=np.arange(n)
        for _ in range(epochs):
            m.train(); np.random.shuffle(idx)
            for s in range(0,n,BATCH_SIZE):
                b=idx[s:s+BATCH_SIZE]; opt.zero_grad()
                loss=crit(m(C[b].to(DEVICE),P[b].to(DEVICE),RI[b].to(DEVICE)),Y[b].to(DEVICE))
                loss.backward(); opt.step()
            sch.step()
        i_held=ridx[held]
        if mode=='nn_onehot':
            nn_rec,nn_id=nearest_training(held,train_recs,seqs)
            rep_held=REPS['nn_onehot'][ridx[nn_rec]]
            extra={'nn_receptor':nn_rec,'nn_identity':round(nn_id,3)}
        else:
            rep_held=REPS[mode][i_held]; extra={}
        m.eval()
        with torch.no_grad():
            reps=torch.FloatTensor(np.tile(rep_held,(len(annotated),1))).to(DEVICE)
            cs=torch.FloatTensor(np.array([feat[c] for c in annotated])).to(DEVICE)
            ris=torch.full((len(annotated),),i_held,dtype=torch.long).to(DEVICE)
            pr=torch.sigmoid(m(cs,reps,ris)).cpu().numpy()
        yt=np.array([1.0 if c in pos_sets[held] else 0.0 for c in annotated])
        npos=int(yt.sum())
        auc=float(roc_auc_score(yt,pr)) if 0<npos<len(yt) else None
        return auc, npos, extra

    results={}
    for held in RECEPTORS:
        row={}
        for mode in MODES:
            a,npos,ex=run(held,mode)
            row[f'auc_{mode}']=None if a is None else round(a,3)
            row.update(ex); row['npos']=npos
        results[held]=row
        print(f"{held:9s} npos={row['npos']:3d} | " +
              " ".join(f"{m}={row[f'auc_{m}']}" for m in MODES) +
              f" | nn_id={row.get('nn_identity')}")

    ev=[r for r in RECEPTORS if all(results[r][f'auc_{m}'] is not None for m in MODES)]
    summary={'n_evaluable':len(ev)}
    for m in MODES:
        vals=[results[r][f'auc_{m}'] for r in ev]
        summary[f'mean_{m}']=round(float(np.mean(vals)),3)
    # stratify by nearest-neighbour identity
    distant=[r for r in ev if results[r].get('nn_identity',1) < 0.5]
    close=[r for r in ev if results[r].get('nn_identity',1) >= 0.5]
    for label, group in [('distant_nn_lt0.5',distant),('close_nn_ge0.5',close)]:
        if group:
            summary[label]={'n':len(group)}
            for m in MODES:
                summary[label][m]=round(float(np.mean([results[r][f'auc_{m}'] for r in group])),3)

    json.dump({'per_receptor':results,'summary':summary},
              open('loro_representations.json','w'), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved: loro_representations.json")
    print("\nKey question: does 'pocket' transfer better than 'random_nonpocket' and")
    print("'fullseq' for receptors without a close paralogue? If not, the transfer")
    print("benefit is not pocket-specific.")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args()
    main(a.epochs,a.seed)
