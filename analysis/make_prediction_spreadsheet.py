"""
Generate the full compound x receptor prediction spreadsheet (Reviewer 1)
=========================================================================
Produces one row per (compound, receptor) pair for all 645 BitterDB compounds and
23 human TAS2R subtypes (14,835 rows), with:

  receptor            receptor name
  receptor_sequence   full receptor sequence used for the embedding
  bdb_cid             BitterDB compound id
  smiles              compound SMILES
  predicted_prob      model probability that the compound activates the receptor
  selectivity_index   SI = p(target) / mean p(all other 22 receptors)
  known_active        1 if the pair is a reported association in BitterDB 2024
  monoselective       1 if flagged as a predicted monoselective ligand
  split               train / test partition of the compound under scaffold split

Requires the production model checkpoint (or retrains it if absent) and the usual
input files. Output: TAS2R_SelectNet_predictions_645x23.csv (and .xlsx if openpyxl
is available).
"""
import json, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings; warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM, POCKET_DIM, COMPOUND_DIM = 256, 2560, 2060
BATCH_SIZE, FOCAL_GAMMA = 128, 2.0
RECEPTORS = ['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7','TAS2R8','TAS2R9',
             'TAS2R10','TAS2R13','TAS2R14','TAS2R16','TAS2R38','TAS2R39','TAS2R40',
             'TAS2R41','TAS2R42','TAS2R43','TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']
N_REC = len(RECEPTORS)
SI_THRESHOLD = 1.5      # manuscript: p > 0.5 and SI > 1.5, for exactly one receptor
P_THRESHOLD  = 0.5


class SelectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.comp = nn.Sequential(nn.Linear(COMPOUND_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,EMBED_DIM))
        self.rec  = nn.Sequential(nn.Linear(POCKET_DIM,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,EMBED_DIM))
        self.W = nn.Parameter(torch.randn(EMBED_DIM,EMBED_DIM)*0.02); self.b = nn.Parameter(torch.zeros(N_REC))
    def forward(self, c, p, r):
        return (self.comp(c)*(self.rec(p)@self.W)).sum(-1) + self.b[r]

class Focal(nn.Module):
    def __init__(s,g=2.0): super().__init__(); s.g=g
    def forward(s,lg,t):
        bce=F.binary_cross_entropy_with_logits(lg,t,reduction='none'); pt=torch.exp(-bce)
        return ((1-pt)**s.g*bce).mean()


def main(ckpt, epochs, out_csv):
    pocket = np.load("tas2r_pocket_embeddings_v2.npy")
    pidx = json.load(open("tas2r_embedding_index_v2.json"))
    seqs = json.load(open("tas2r_sequences_bdb.json"))
    fp = np.load("morgan_fingerprints_v2.npy")
    cids = pd.read_csv("fp_cid_index_v2.csv")['bdb_cid'].tolist()
    dcols = ['mw','alogp','tpsa','hbd','hba','rotb','rings','arom_rings','fsp3','formal_charge','heavy_atoms','globularity_proxy']
    cdf = pd.read_csv("compounds_with_descriptors_v2.csv")
    smiles_col = next((c for c in ['smiles','SMILES','canonical_smiles'] if c in cdf.columns), None)
    smiles_map = dict(zip(cdf['bdb_cid'], cdf[smiles_col])) if smiles_col else {}
    dd = cdf.set_index('bdb_cid')[dcols]
    c2i = {c:i for i,c in enumerate(cids)}
    raw = np.array([dd.loc[c].values if c in dd.index else np.zeros(12) for c in cids], dtype=np.float32)
    mu, sd = raw.mean(0), raw.std(0)+1e-8
    feat = {c: np.concatenate([fp[c2i[c]].astype(np.float32),
                               (dd.loc[c].values.astype(np.float32)-mu)/sd])
            for c in cids if c in dd.index}

    lr = pd.read_csv("ligandReceptors_2024.csv")
    human=[1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
    lr = lr[lr['rID'].isin(human)].copy(); lr['receptor']=lr['rID'].map({r:f'TAS2R{r}' for r in human})
    pos = {r:set(lr[lr['receptor']==r]['cID'].unique()) for r in RECEPTORS}
    split = pd.read_csv("scaffold_split_v2.csv")
    split_map = dict(zip(split['bdb_cid'], split['split']))
    ridx = {r:i for i,r in enumerate(RECEPTORS)}

    model = SelectNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"Loaded checkpoint: {ckpt}")
    except Exception:
        print("No checkpoint found; retraining production model on the scaffold split...")
        tr = set(split[split['split']=='train']['bdb_cid'])
        C,P,R,Y=[],[],[],[]
        for r in RECEPTORS:
            i=ridx[r]
            for c in tr:
                if c not in feat: continue
                C.append(feat[c]); P.append(pocket[pidx[r]]); R.append(i)
                Y.append(1.0 if c in pos[r] else 0.0)
        C=torch.FloatTensor(np.array(C)); P=torch.FloatTensor(np.array(P))
        R=torch.LongTensor(R); Y=torch.FloatTensor(Y)
        crit=Focal(FOCAL_GAMMA); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        n=len(Y); idx=np.arange(n); torch.manual_seed(42); np.random.seed(42)
        for eps,lr_ in [(epochs,1e-3),(epochs,3e-4)]:
            for g in opt.param_groups: g['lr']=lr_
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=eps)
            for _ in range(eps):
                model.train(); np.random.shuffle(idx)
                for s in range(0,n,BATCH_SIZE):
                    b=idx[s:s+BATCH_SIZE]; opt.zero_grad()
                    loss=crit(model(C[b].to(DEVICE),P[b].to(DEVICE),R[b].to(DEVICE)),Y[b].to(DEVICE))
                    loss.backward(); opt.step()
                sch.step()
        torch.save(model.state_dict(), "selectnet_production.pt")
        print("Saved checkpoint: selectnet_production.pt")

    # predict all pairs
    model.eval()
    all_c = [c for c in cids if c in feat]
    Cm = torch.FloatTensor(np.array([feat[c] for c in all_c]))
    prob = np.zeros((len(all_c), N_REC), dtype=np.float32)
    with torch.no_grad():
        for j, r in enumerate(RECEPTORS):
            P = torch.FloatTensor(np.tile(pocket[pidx[r]], (len(all_c),1)))
            R = torch.full((len(all_c),), ridx[r], dtype=torch.long)
            out = []
            for s in range(0, len(all_c), 512):
                out.append(torch.sigmoid(model(Cm[s:s+512].to(DEVICE), P[s:s+512].to(DEVICE),
                                               R[s:s+512].to(DEVICE))).cpu().numpy())
            prob[:, j] = np.concatenate(out)

    rows = []
    for i, c in enumerate(all_c):
        p = prob[i]
        # SI for every receptor, then apply the manuscript rule:
        # monoselective iff the compound passes (p > 0.5 and SI > 1.5) for EXACTLY one receptor
        sis = np.array([p[j] / (np.delete(p, j).mean() + 1e-9) for j in range(N_REC)])
        passes = (p > P_THRESHOLD) & (sis > SI_THRESHOLD)
        exactly_one = passes.sum() == 1
        for j, r in enumerate(RECEPTORS):
            si = float(sis[j])
            rows.append({
                'receptor': r,
                'receptor_sequence': seqs.get(r, ''),
                'bdb_cid': c,
                'smiles': smiles_map.get(c, ''),
                'predicted_prob': round(float(p[j]), 4),
                'selectivity_index': round(si, 3),
                'known_active': int(c in pos[r]),
                'monoselective': int(exactly_one and passes[j]),
                'split': split_map.get(c, 'unassigned'),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(df)} rows = {len(all_c)} compounds x {N_REC} receptors)")
    n_mono = int(df['monoselective'].sum())
    n_sub = df.loc[df['monoselective']==1,'receptor'].nunique()
    print(f"Predicted monoselective ligands: {n_mono} across {n_sub} receptor subtypes")
    print("(manuscript rule: p > 0.5 and SI > 1.5 for exactly one receptor)")
    try:
        df.to_excel(out_csv.replace('.csv','.xlsx'), index=False)
        print(f"Saved: {out_csv.replace('.csv','.xlsx')}")
    except Exception as e:
        print(f"(xlsx export skipped: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="selectnet_production.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="TAS2R_SelectNet_predictions_645x23.csv")
    a = ap.parse_args()
    main(a.ckpt, a.epochs, a.out)
