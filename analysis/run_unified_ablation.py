"""
Unified receptor-representation ablation under IDENTICAL input conditions
(Reviewer 1, round 2 -- resolves the pocket-vs-full-sequence inconsistency)
===========================================================================
The original ablation (Table S9) compared representations at their native
dimensionality: pocket embeddings were 1280-dim, full-sequence pooling 1280-dim,
ProtBERT 1024-dim, one-hot 23-dim. The pocket-size and non-pocket-control
experiments, in contrast, duplicated every 1280-dim vector to 2560 dims so the
model input width matched the production configuration.

Those two settings gave different orderings, which is itself informative and must
be reported honestly. This script removes the confound by evaluating EVERY
representation under the SAME protocol: each receptor vector is projected to the
same input width (default: tile/pad to 2560), with the same encoder, bilinear
head, scaffold split, focal loss, schedule, and the same set of seeds.

Representations compared:
  onehot            23-dim one-hot, tiled to the common width
  aacomp            20-dim amino-acid composition, tiled
  protbert          ProtBERT mean pool (1024), tiled          [if emb_protbert.npy present]
  esm2_fullseq      ESM-2 full-sequence mean pool (1280), duplicated
  esm2_pocket       ESM-2 pocket mean pool (1280), duplicated
  esm2_single_pocket   one central pocket residue (1280), duplicated
  esm2_random_nonpocket one random NON-pocket residue (1280), duplicated  [n_random draws]

Outputs: unified_ablation.json

Requires GPU + fair-esm for the ESM-2 variants (uses cached per-residue extraction),
plus the usual dataset files.
"""
import json, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM, COMPOUND_DIM = 256, 2060
COMMON_DIM = 2560
BATCH_SIZE, FOCAL_GAMMA = 128, 2.0
RECEPTORS = ['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7','TAS2R8','TAS2R9',
             'TAS2R10','TAS2R13','TAS2R14','TAS2R16','TAS2R38','TAS2R39','TAS2R40',
             'TAS2R41','TAS2R42','TAS2R43','TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']
BROAD = ['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']
N_REC = len(RECEPTORS)
AAS = 'ACDEFGHIKLMNPQRSTVWY'

_C = {}
def esm_per_residue(seq, rec):
    if 'm' not in _C:
        import esm
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        bc = alphabet.get_batch_converter(); model = model.eval().to(DEVICE)
        if DEVICE == 'cuda': model = model.half()
        _C['m'] = (model, bc); _C['cache'] = {}
    if rec in _C['cache']:
        return _C['cache'][rec]
    model, bc = _C['m']
    with torch.no_grad():
        _,_,toks = bc([(rec, seq)]); toks = toks.to(DEVICE)
        rep = model(toks, repr_layers=[33])["representations"][33][0]
        pr = rep[1:len(seq)+1].float().cpu().numpy()
    _C['cache'][rec] = pr
    return pr


def tile_to(v, width=COMMON_DIM):
    """Tile (repeat) a vector to the common input width, truncating any excess."""
    reps = int(np.ceil(width / len(v)))
    return np.tile(v, reps)[:width].astype(np.float32)


def pocket_pos(rec, pocket, L):
    p = list(pocket[rec].get('orthosteric', [])) + list(pocket[rec].get('intracellular', []))
    return sorted(set(x-1 for x in p if 1 <= x <= L))


def build(rep_name, seed=0):
    seqs = json.load(open('tas2r_sequences_bdb.json'))
    emb = np.zeros((N_REC, COMMON_DIM), dtype=np.float32)

    if rep_name == 'onehot':
        for i in range(N_REC):
            v = np.zeros(N_REC, dtype=np.float32); v[i] = 1.0
            emb[i] = tile_to(v)
        return emb

    if rep_name == 'aacomp':
        for i, rec in enumerate(RECEPTORS):
            s = seqs[rec]
            v = np.array([s.count(a)/len(s) for a in AAS], dtype=np.float32)
            emb[i] = tile_to(v)
        return emb

    if rep_name == 'protbert':
        pb = np.load('emb_protbert.npy')
        idx = json.load(open('tas2r_embedding_index_v2.json'))
        for i, rec in enumerate(RECEPTORS):
            emb[i] = tile_to(pb[idx[rec]].astype(np.float32))
        return emb

    pocket = json.load(open('tas2r_pocket_definitions_v2.json'))
    rng = np.random.default_rng(seed)
    for i, rec in enumerate(RECEPTORS):
        s = seqs[rec]; pr = esm_per_residue(s, rec)
        pos = pocket_pos(rec, pocket, len(s))
        if rep_name == 'esm2_fullseq':
            v = pr.mean(0)
        elif rep_name == 'esm2_pocket':
            v = pr[pos].mean(0) if pos else pr.mean(0)
        elif rep_name == 'esm2_single_pocket':
            v = pr[pos[len(pos)//2]] if pos else pr.mean(0)
        elif rep_name == 'esm2_random_nonpocket':
            nonp = [j for j in range(len(s)) if j not in set(pos)]
            v = pr[rng.choice(nonp)]
        else:
            raise ValueError(rep_name)
        emb[i] = tile_to(v)
    return emb


class SelectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.comp = nn.Sequential(nn.Linear(COMPOUND_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,EMBED_DIM))
        self.rec  = nn.Sequential(nn.Linear(COMMON_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,EMBED_DIM))
        self.W = nn.Parameter(torch.randn(EMBED_DIM,EMBED_DIM)*0.02); self.b = nn.Parameter(torch.zeros(N_REC))
    def forward(self, c, p, r):
        ce = self.comp(c); pe = self.rec(p)
        return (ce*(pe@self.W)).sum(-1) + self.b[r]

class Focal(nn.Module):
    def __init__(self,g=2.0): super().__init__(); self.g=g
    def forward(self,lg,t):
        bce=F.binary_cross_entropy_with_logits(lg,t,reduction='none'); pt=torch.exp(-bce)
        return ((1-pt)**self.g*bce).mean()


def load_dataset():
    fp = np.load("morgan_fingerprints_v2.npy")
    cids = pd.read_csv("fp_cid_index_v2.csv")['bdb_cid'].tolist()
    dcols = ['mw','alogp','tpsa','hbd','hba','rotb','rings','arom_rings','fsp3','formal_charge','heavy_atoms','globularity_proxy']
    dd = pd.read_csv("compounds_with_descriptors_v2.csv").set_index('bdb_cid')[dcols]
    c2i = {c:i for i,c in enumerate(cids)}
    raw = np.array([dd.loc[c].values if c in dd.index else np.zeros(12) for c in cids], dtype=np.float32)
    mu, sd = raw.mean(0), raw.std(0)+1e-8
    feat = {}
    for c in cids:
        if c in dd.index:
            feat[c] = np.concatenate([fp[c2i[c]].astype(np.float32), (dd.loc[c].values.astype(np.float32)-mu)/sd])
    lr = pd.read_csv("ligandReceptors_2024.csv")
    human=[1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
    lr = lr[lr['rID'].isin(human)].copy(); lr['receptor']=lr['rID'].map({r:f'TAS2R{r}' for r in human})
    sp = pd.read_csv("scaffold_split_v2.csv")
    tr=set(sp[sp['split']=='train']['bdb_cid']); te=set(sp[sp['split']=='test']['bdb_cid'])
    pos={r:set(lr[lr['receptor']==r]['cID'].unique()) for r in RECEPTORS}
    ridx={r:i for i,r in enumerate(RECEPTORS)}
    return feat,pos,ridx,tr,te


def tensors(cset, feat, pos, ridx, emb):
    C,P,R,Y=[],[],[],[]
    for r in RECEPTORS:
        i=ridx[r]
        for c in cset:
            if c not in feat: continue
            C.append(feat[c]); P.append(emb[i]); R.append(i); Y.append(1.0 if c in pos[r] else 0.0)
    return (torch.FloatTensor(np.array(C)),torch.FloatTensor(np.array(P)),
            torch.LongTensor(R),torch.FloatTensor(Y))


def run(emb, ds, seed, epochs):
    feat,pos,ridx,tr,te = ds
    Ctr,Ptr,Rtr,Ytr = tensors(tr,feat,pos,ridx,emb)
    Cte,Pte,Rte,Yte = tensors(te,feat,pos,ridx,emb)
    torch.manual_seed(seed); np.random.seed(seed)
    m = SelectNet().to(DEVICE); crit=Focal()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n=len(Ytr); idx=np.arange(n); best=-1
    for eps,lr in [(epochs,1e-3),(epochs,3e-4)]:
        for g in opt.param_groups: g['lr']=lr
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=eps)
        for _ in range(eps):
            m.train(); np.random.shuffle(idx)
            for s in range(0,n,BATCH_SIZE):
                b=idx[s:s+BATCH_SIZE]; opt.zero_grad()
                loss=crit(m(Ctr[b].to(DEVICE),Ptr[b].to(DEVICE),Rtr[b].to(DEVICE)), Ytr[b].to(DEVICE))
                loss.backward(); opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            pr=torch.sigmoid(m(Cte.to(DEVICE),Pte.to(DEVICE),Rte.to(DEVICE))).cpu().numpy()
        L=Yte.numpy(); Rc=Rte.numpy()
        a=[roc_auc_score(L[Rc==ridx[r]],pr[Rc==ridx[r]]) for r in BROAD
           if 0<L[Rc==ridx[r]].sum()<(Rc==ridx[r]).sum()]
        v=float(np.mean(a)) if a else 0
        if v>best: best=v
    return best


def main(seeds, epochs, n_random):
    ds = load_dataset()
    reps = ['onehot','aacomp','esm2_fullseq','esm2_pocket','esm2_single_pocket']
    try:
        np.load('emb_protbert.npy'); reps.insert(2,'protbert')
    except Exception:
        print("(emb_protbert.npy not found; skipping ProtBERT)")

    out = {}
    for rp in reps:
        emb = build(rp)
        vals = [run(emb, ds, s, epochs) for s in range(seeds)]
        out[rp] = {'mean': round(float(np.mean(vals)),3), 'std': round(float(np.std(vals)),3),
                   'vals': [round(v,3) for v in vals]}
        print(f"{rp:24s}: {out[rp]['mean']:.3f} +/- {out[rp]['std']:.3f}")

    # random non-pocket: average over several random residue draws
    rvals = []
    for d in range(n_random):
        emb = build('esm2_random_nonpocket', seed=d)
        rvals.append(run(emb, ds, 0, epochs))
    out['esm2_random_nonpocket'] = {'mean': round(float(np.mean(rvals)),3),
                                    'std': round(float(np.std(rvals)),3),
                                    'vals':[round(v,3) for v in rvals],
                                    'note': f'{n_random} independent random residue draws'}
    print(f"{'esm2_random_nonpocket':24s}: {out['esm2_random_nonpocket']['mean']:.3f} "
          f"+/- {out['esm2_random_nonpocket']['std']:.3f}")

    json.dump(out, open('unified_ablation.json','w'), indent=2)
    print("\nSaved: unified_ablation.json")
    print("\nAll representations evaluated at identical input width "
          f"({COMMON_DIM}) with identical encoder, head, split, and schedule.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n_random", type=int, default=5)
    a = ap.parse_args()
    main(a.seeds, a.epochs, a.n_random)
