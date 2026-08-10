"""
Random non-pocket residue control + TAS2R14 cholesterol-pocket evaluation
(Reviewer 1, round 2)
=========================================================================
Two experiments the reviewer asked for.

(ii) RANDOM NON-POCKET RESIDUE. Instead of pooling the defined binding pocket,
represent each receptor by the ESM-2 embedding of a SINGLE randomly chosen residue
that is NOT in the pocket. If this matches or beats the pooled full-protein / pocket
embedding, then 'any residue that distinguishes receptors' is enough and the pocket
is not special. If it is clearly worse, the pocket carries real information.
We repeat over several random seeds (different random residues) and report the
distribution, comparing against: single central POCKET residue, full pocket, and
full-sequence mean pool.

(iii) TAS2R14 CHOLESTEROL POCKET. Complete the TAS2R14 single-pocket control by
also evaluating the model when TAS2R14 is represented by ONLY its upper-
transmembrane (cholesterol-site) pocket, duplicated to 2560 dims, versus only its
intracellular (agonist) pocket. This isolates which of the two TAS2R14 sub-pockets
carries the signal.

Requires GPU + fair-esm, plus the usual dataset files and
tas2r_pocket_definitions_v2.json, tas2r_sequences_bdb.json.

Outputs: nonpocket_control_results.json, t14_cholesterol_results.json
"""
import json, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM, POCKET_DIM, COMPOUND_DIM = 256, 2560, 2060
BATCH_SIZE, FOCAL_GAMMA = 128, 2.0
RECEPTORS = ['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7','TAS2R8','TAS2R9',
             'TAS2R10','TAS2R13','TAS2R14','TAS2R16','TAS2R38','TAS2R39','TAS2R40',
             'TAS2R41','TAS2R42','TAS2R43','TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']
BROAD = ['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']
N_REC = len(RECEPTORS)

_ESM_CACHE = {}
def get_esm():
    if 'm' not in _ESM_CACHE:
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        bc = alphabet.get_batch_converter(); model = model.eval().to(DEVICE)
        if DEVICE == 'cuda': model = model.half()
        _ESM_CACHE['m'] = (model, bc)
    return _ESM_CACHE['m']

def per_residue(seq, rec):
    model, bc = get_esm()
    with torch.no_grad():
        _,_,toks = bc([(rec, seq)]); toks = toks.to(DEVICE)
        rep = model(toks, repr_layers=[33])["representations"][33][0]
        return rep[1:len(seq)+1].float().cpu().numpy()

def pocket_positions(rec, pocket, seqlen):
    pos = list(pocket[rec].get('orthosteric', [])) + list(pocket[rec].get('intracellular', []))
    return sorted(set(p-1 for p in pos if 1 <= p <= seqlen))


def build_random_nonpocket(seed):
    """Each receptor = one random residue NOT in its pocket, duplicated to 2560."""
    seqs = json.load(open('tas2r_sequences_bdb.json'))
    pocket = json.load(open('tas2r_pocket_definitions_v2.json'))
    rng = np.random.default_rng(seed)
    emb = np.zeros((N_REC, 2560), dtype=np.float32)
    for i, rec in enumerate(RECEPTORS):
        seq = seqs[rec]; pr = per_residue(seq, rec)
        pos = set(pocket_positions(rec, pocket, len(seq)))
        nonpocket = [j for j in range(len(seq)) if j not in pos]
        pick = rng.choice(nonpocket)
        v = pr[pick]
        emb[i] = np.concatenate([v, v])
    return emb


def build_single_pocket(seed=0):
    seqs = json.load(open('tas2r_sequences_bdb.json'))
    pocket = json.load(open('tas2r_pocket_definitions_v2.json'))
    emb = np.zeros((N_REC, 2560), dtype=np.float32)
    for i, rec in enumerate(RECEPTORS):
        seq = seqs[rec]; pr = per_residue(seq, rec)
        pos = pocket_positions(rec, pocket, len(seq))
        v = pr[pos[len(pos)//2]] if pos else pr.mean(0)
        emb[i] = np.concatenate([v, v])
    return emb


def build_full_pocket(seed=0):
    seqs = json.load(open('tas2r_sequences_bdb.json'))
    pocket = json.load(open('tas2r_pocket_definitions_v2.json'))
    emb = np.zeros((N_REC, 2560), dtype=np.float32)
    for i, rec in enumerate(RECEPTORS):
        seq = seqs[rec]; pr = per_residue(seq, rec)
        pos = pocket_positions(rec, pocket, len(seq))
        v = pr[pos].mean(0) if pos else pr.mean(0)
        emb[i] = np.concatenate([v, v])
    return emb


def build_t14_variant(which):
    """Production embeddings, but TAS2R14 replaced by a single sub-pocket.
    which in {'cholesterol','intracellular'}."""
    emb = np.load("tas2r_pocket_embeddings_v2.npy").copy()
    idx = json.load(open("tas2r_embedding_index_v2.json"))
    seqs = json.load(open('tas2r_sequences_bdb.json'))
    pocket = json.load(open('tas2r_pocket_definitions_v2.json'))
    rec = 'TAS2R14'; seq = seqs[rec]; pr = per_residue(seq, rec)
    orth = sorted(set(p-1 for p in pocket[rec].get('orthosteric', []) if 1 <= p <= len(seq)))
    intra = sorted(set(p-1 for p in pocket[rec].get('intracellular', []) if 1 <= p <= len(seq)))
    v = pr[orth].mean(0) if which == 'cholesterol' else pr[intra].mean(0)
    row = idx[rec]
    emb[row] = np.concatenate([v, v])
    return emb


# ---------- shared train/eval ----------
class SelectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.comp = nn.Sequential(nn.Linear(COMPOUND_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,EMBED_DIM))
        self.rec  = nn.Sequential(nn.Linear(POCKET_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,EMBED_DIM))
        self.W = nn.Parameter(torch.randn(EMBED_DIM,EMBED_DIM)*0.02); self.b = nn.Parameter(torch.zeros(N_REC))
    def forward(self, c, p, ridx):
        ce = self.comp(c); pe = self.rec(p)
        return (ce*(pe@self.W)).sum(-1) + self.b[ridx]

class Focal(nn.Module):
    def __init__(self,g=2.0): super().__init__(); self.g=g
    def forward(self,lg,t):
        bce=F.binary_cross_entropy_with_logits(lg,t,reduction='none'); pt=torch.exp(-bce)
        return ((1-pt)**self.g*bce).mean()

def load_dataset():
    fp_matrix = np.load("morgan_fingerprints_v2.npy")
    fp_cids = pd.read_csv("fp_cid_index_v2.csv")['bdb_cid'].tolist()
    desc_cols = ['mw','alogp','tpsa','hbd','hba','rotb','rings','arom_rings','fsp3','formal_charge','heavy_atoms','globularity_proxy']
    desc_df = pd.read_csv("compounds_with_descriptors_v2.csv").set_index('bdb_cid')[desc_cols]
    cid_to_idx = {cid:i for i,cid in enumerate(fp_cids)}
    draw = np.array([desc_df.loc[c].values if c in desc_df.index else np.zeros(12) for c in fp_cids], dtype=np.float32)
    dmu, dsd = draw.mean(0), draw.std(0)+1e-8
    feat={}
    for c in fp_cids:
        if c in desc_df.index:
            fp=fp_matrix[cid_to_idx[c]].astype(np.float32)
            ph=(desc_df.loc[c].values.astype(np.float32)-dmu)/dsd
            feat[c]=np.concatenate([fp,ph])
    df_lr=pd.read_csv("ligandReceptors_2024.csv")
    human_rids=[1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
    df_h=df_lr[df_lr['rID'].isin(human_rids)].copy(); df_h['receptor']=df_h['rID'].map({r:f'TAS2R{r}' for r in human_rids})
    split=pd.read_csv("scaffold_split_v2.csv")
    train_cids=set(split[split['split']=='train']['bdb_cid']); test_cids=set(split[split['split']=='test']['bdb_cid'])
    pos_sets={r:set(df_h[df_h['receptor']==r]['cID'].unique()) for r in RECEPTORS}
    ridx={r:i for i,r in enumerate(RECEPTORS)}
    return feat,pos_sets,ridx,train_cids,test_cids

def build_tensors(cset,feat,pos_sets,ridx,emb):
    C,P,RI,Y=[],[],[],[]
    for r in RECEPTORS:
        i=ridx[r]; rep=emb[i]
        for c in cset:
            if c not in feat: continue
            C.append(feat[c]); P.append(rep); RI.append(i); Y.append(1.0 if c in pos_sets[r] else 0.0)
    return (torch.FloatTensor(np.array(C)),torch.FloatTensor(np.array(P)),torch.LongTensor(RI),torch.FloatTensor(Y))

def train_eval(emb, ds, seed, epochs, tiers=None):
    feat,pos_sets,ridx,train_cids,test_cids=ds
    Ctr,Ptr,Rtr,Ytr=build_tensors(train_cids,feat,pos_sets,ridx,emb)
    Cte,Pte,Rte,Yte=build_tensors(test_cids,feat,pos_sets,ridx,emb)
    torch.manual_seed(seed); np.random.seed(seed)
    model=SelectNet().to(DEVICE); crit=Focal()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    n=len(Ytr); idx=np.arange(n); best=-1; best_state=None
    for eps,lr in [(epochs,1e-3),(epochs,3e-4)]:
        for g in opt.param_groups: g['lr']=lr
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=eps)
        for ep in range(eps):
            model.train(); np.random.shuffle(idx)
            for s in range(0,n,BATCH_SIZE):
                bi=idx[s:s+BATCH_SIZE]; opt.zero_grad()
                lg=model(Ctr[bi].to(DEVICE),Ptr[bi].to(DEVICE),Rtr[bi].to(DEVICE))
                loss=crit(lg,Ytr[bi].to(DEVICE)); loss.backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            Pr=torch.sigmoid(model(Cte.to(DEVICE),Pte.to(DEVICE),Rte.to(DEVICE))).cpu().numpy()
        L=Yte.numpy(); Rc=Rte.numpy()
        aucs=[roc_auc_score(L[Rc==ridx[r]],Pr[Rc==ridx[r]]) for r in BROAD
              if 0<L[Rc==ridx[r]].sum()<(Rc==ridx[r]).sum()]
        macro=float(np.mean(aucs)) if aucs else 0
        if macro>best: best=macro
    return best


def main(mode, seeds, epochs):
    ds = load_dataset()

    if mode in ('nonpocket', 'all'):
        print("\n=== (ii) Random non-pocket residue control ===")
        res = {}
        # references
        print("Building reference embeddings...")
        emb_single = build_single_pocket(); res['single_pocket'] = [train_eval(emb_single, ds, s, epochs) for s in range(3)]
        emb_full = build_full_pocket();     res['full_pocket']   = [train_eval(emb_full, ds, s, epochs) for s in range(3)]
        # random non-pocket over seeds
        rnd = []
        for sd in range(seeds):
            emb_r = build_random_nonpocket(sd)
            v = train_eval(emb_r, ds, 0, epochs)
            rnd.append(v); print(f"  random-nonpocket seed {sd}: broad AUC = {v:.3f}")
        res['random_nonpocket'] = rnd
        summary = {k: {'mean': round(float(np.mean(v)),3), 'std': round(float(np.std(v)),3),
                       'vals':[round(x,3) for x in v]} for k,v in res.items()}
        json.dump(summary, open("nonpocket_control_results.json","w"), indent=2)
        print("\nSummary:"); print(json.dumps(summary, indent=2))
        print("Saved: nonpocket_control_results.json")

    if mode in ('cholesterol', 'all'):
        print("\n=== (iii) TAS2R14 cholesterol vs intracellular sub-pocket ===")
        res = {}
        for which in ['cholesterol', 'intracellular']:
            emb = build_t14_variant(which)
            vals = [train_eval(emb, ds, s, epochs) for s in range(3)]
            res[which] = {'mean': round(float(np.mean(vals)),3), 'std': round(float(np.std(vals)),3),
                          'vals':[round(x,3) for x in vals]}
            print(f"  TAS2R14 = {which} only: broad AUC = {res[which]['mean']} +/- {res[which]['std']}")
        json.dump(res, open("t14_cholesterol_results.json","w"), indent=2)
        print("Saved: t14_cholesterol_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=['nonpocket','cholesterol','all'], default='all')
    ap.add_argument("--seeds", type=int, default=5, help="random non-pocket residue draws")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    main(a.mode, a.seeds, a.epochs)
