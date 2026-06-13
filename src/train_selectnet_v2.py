#!/usr/bin/env python3
"""
TAS2R-SelectNet: Full Model Training Pipeline
=============================================
GPU 서버에서 실행:
  python train_selectnet.py [--mode {train,eval,si}]

Requirements:
  pip install torch xgboost scikit-learn numpy pandas

Files needed (copy from Claude session):
  - tas2r_pocket_embeddings_v2.npy
  - tas2r_embedding_index_v2.json
  - morgan_fingerprints_v2.npy
  - fp_cid_index_v2.csv
  - compounds_with_descriptors_v2.csv
  - ligandReceptors_2024.csv
  - scaffold_split_v2.csv
"""

import argparse, json, os, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM   = 256
POCKET_DIM  = 2560    # ESM-2 650M: 1280*2 (orth + intra)
COMPOUND_DIM= 2060    # 2048 fp + 12 descriptors
BATCH_SIZE  = 128
EPOCHS      = 60
LR          = 1e-3
FOCAL_GAMMA = 2.0

RECEPTORS = ['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7',
             'TAS2R8','TAS2R9','TAS2R10','TAS2R13','TAS2R14','TAS2R16',
             'TAS2R38','TAS2R39','TAS2R40','TAS2R41','TAS2R42','TAS2R43',
             'TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']

print(f"Device: {DEVICE}")

# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()

# ── Model ──────────────────────────────────────────────────────────────────────
class TAS2RSelectNet(nn.Module):
    def __init__(self, compound_dim=COMPOUND_DIM, pocket_dim=POCKET_DIM,
                 embed_dim=EMBED_DIM, n_receptors=23):
        super().__init__()
        # Compound encoder
        self.compound_enc = nn.Sequential(
            nn.Linear(compound_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, embed_dim),    nn.LayerNorm(embed_dim),
        )
        # Pocket encoder
        self.pocket_enc = nn.Sequential(
            nn.Linear(pocket_dim, 512),   nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, embed_dim),    nn.LayerNorm(embed_dim),
        )
        # Bilinear interaction weight matrix W: embed × embed
        self.W = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        # Per-receptor bias
        self.bias = nn.Parameter(torch.zeros(n_receptors))

    def forward(self, compound_feat, pocket_feat, receptor_idx):
        """
        compound_feat: (B, compound_dim)
        pocket_feat:   (B, pocket_dim)  -- pre-looked-up ESM embedding
        receptor_idx:  (B,) int
        Returns: (B,) logits
        """
        c_emb = self.compound_enc(compound_feat)   # (B, D)
        p_emb = self.pocket_enc(pocket_feat)        # (B, D)
        # Bilinear score: c^T W p
        score = (c_emb @ self.W * p_emb).sum(dim=-1)   # (B,)
        bias  = self.bias[receptor_idx]                  # (B,)
        return score + bias

# ── Dataset ────────────────────────────────────────────────────────────────────
class TAS2RDataset(Dataset):
    def __init__(self, pairs, compound_features, pocket_embeddings, pocket_idx):
        """
        pairs: list of (cid, receptor_name, label)
        compound_features: dict cid → np.array (2060,)
        pocket_embeddings: np.array (23, 2560)
        pocket_idx: dict receptor_name → int
        """
        self.data = []
        for cid, rec, label in pairs:
            if cid not in compound_features: continue
            ridx = pocket_idx.get(rec, -1)
            if ridx < 0: continue
            self.data.append((
                torch.FloatTensor(compound_features[cid]),
                torch.FloatTensor(pocket_embeddings[ridx]),
                ridx,
                float(label)
            ))

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")

pocket_emb = np.load("tas2r_pocket_embeddings_v2.npy")      # (23, 2560)
with open("tas2r_embedding_index_v2.json") as f:
    pocket_idx = json.load(f)

fp_matrix  = np.load("morgan_fingerprints_v2.npy")          # (N, 2048)
fp_cids    = pd.read_csv("fp_cid_index_v2.csv")['bdb_cid'].tolist()
df_desc    = pd.read_csv("compounds_with_descriptors_v2.csv")
desc_cols  = ['mw','alogp','tpsa','hbd','hba','rotb','rings','arom_rings',
              'fsp3','formal_charge','heavy_atoms','globularity_proxy']
desc_df    = df_desc.set_index('bdb_cid')[desc_cols]
cid_to_idx = {cid: i for i, cid in enumerate(fp_cids)}

compound_features = {}
for cid in fp_cids:
    if cid in desc_df.index:
        fp = fp_matrix[cid_to_idx[cid]].astype(np.float32)
        ph = desc_df.loc[cid].values.astype(np.float32)
        compound_features[cid] = np.concatenate([fp, ph])

df_lr   = pd.read_csv("ligandReceptors_2024.csv")
human_rids = [1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
rid_to_name = {1:'TAS2R1',2:'TAS2R2',3:'TAS2R3',4:'TAS2R4',5:'TAS2R5',
               7:'TAS2R7',8:'TAS2R8',9:'TAS2R9',10:'TAS2R10',13:'TAS2R13',
               14:'TAS2R14',16:'TAS2R16',38:'TAS2R38',39:'TAS2R39',40:'TAS2R40',
               41:'TAS2R41',42:'TAS2R42',43:'TAS2R43',44:'TAS2R44',46:'TAS2R46',
               47:'TAS2R47',49:'TAS2R49',50:'TAS2R50'}
df_h = df_lr[df_lr['rID'].isin(human_rids)].copy()
df_h['receptor'] = df_h['rID'].map(rid_to_name)

split    = pd.read_csv("scaffold_split_v2.csv")
train_cids = set(split[split['split']=='train']['bdb_cid'])
test_cids  = set(split[split['split']=='test']['bdb_cid'])
annotated  = set(df_h['cID'].unique())

# Build pairs
def build_pairs(cid_set):
    pairs = []
    for rec in RECEPTORS:
        pos = set(df_h[df_h['receptor']==rec]['cID'].unique())
        neg = annotated - pos
        for cid in cid_set:
            if cid not in compound_features: continue
            if cid in pos:   pairs.append((cid, rec, 1))
            elif cid in neg: pairs.append((cid, rec, 0))
    return pairs

train_pairs = build_pairs(train_cids)
test_pairs  = build_pairs(test_cids)
print(f"Train pairs: {len(train_pairs)} ({sum(p[2] for p in train_pairs)} pos)")
print(f"Test pairs:  {len(test_pairs)} ({sum(p[2] for p in test_pairs)} pos)")

# ── Train ──────────────────────────────────────────────────────────────────────
train_ds = TAS2RDataset(train_pairs, compound_features, pocket_emb, pocket_idx)
test_ds  = TAS2RDataset(test_pairs,  compound_features, pocket_emb, pocket_idx)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = TAS2RSelectNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = FocalLoss(gamma=FOCAL_GAMMA)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Starting training...")

best_auc = 0
for epoch in range(1, EPOCHS+1):
    # Train
    model.train()
    losses = []
    for c_feat, p_feat, r_idx, labels in train_dl:
        c_feat = c_feat.to(DEVICE)
        p_feat = p_feat.to(DEVICE)
        r_idx  = r_idx.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(c_feat, p_feat, r_idx)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()

    # Eval every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        all_probs, all_labels, all_recs = [], [], []
        with torch.no_grad():
            for c_feat, p_feat, r_idx, labels in test_dl:
                logits = model(c_feat.to(DEVICE), p_feat.to(DEVICE), r_idx.to(DEVICE))
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())
                all_recs.extend(r_idx.numpy())

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_recs   = np.array(all_recs)

        # Macro AUC across receptors
        rec_names = {v: k for k, v in pocket_idx.items()}
        aucs = []
        for ridx in np.unique(all_recs):
            mask = all_recs == ridx
            if all_labels[mask].sum() < 1: continue
            try:
                aucs.append(roc_auc_score(all_labels[mask], all_probs[mask]))
            except: pass

        macro_auc = np.mean(aucs) if aucs else 0
        global_auc = roc_auc_score(all_labels, all_probs) if all_labels.sum() > 0 else 0
        print(f"Epoch {epoch:3d} | loss={np.mean(losses):.4f} | "
              f"macro_AUC={macro_auc:.3f} | global_AUC={global_auc:.3f}")

        if macro_auc > best_auc:
            best_auc = macro_auc
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'macro_auc': macro_auc
            }, "tas2r_selectnet_best_v2.pt")

print(f"\nBest macro AUC: {best_auc:.3f}")
print("Model saved: tas2r_selectnet_best_v2.pt")

# ── Selectivity Index ──────────────────────────────────────────────────────────
print("\nComputing Selectivity Index for all compounds...")
model.load_state_dict(torch.load("tas2r_selectnet_best_v2.pt", weights_only=False)['model_state'])
model.eval()

si_records = []
rec_names_sorted = sorted(pocket_idx.keys())
for cid in compound_features:
    probs = {}
    for rec in rec_names_sorted:
        ridx = pocket_idx[rec]
        c_t = torch.FloatTensor(compound_features[cid]).unsqueeze(0).to(DEVICE)
        p_t = torch.FloatTensor(pocket_emb[ridx]).unsqueeze(0).to(DEVICE)
        r_t = torch.tensor([ridx], device=DEVICE)
        with torch.no_grad():
            p = torch.sigmoid(model(c_t, p_t, r_t)).item()
        probs[rec] = p
    
    mean_p = np.mean(list(probs.values()))
    for rec, p in probs.items():
        si = p / (mean_p + 1e-8)
        if p > 0.5 and si > 1.5:
            n_active = sum(1 for pp in probs.values() if pp > 0.5)
            si_records.append({'cid': cid, 'receptor': rec, 'prob': p,
                                'si': si, 'n_active': n_active})

si_df = pd.DataFrame(si_records).sort_values('si', ascending=False)
si_df.to_csv("selectivity_index_selectnet_v2.csv", index=False)
print(f"SI results: {len(si_df)} predictions (P>0.5, SI>1.5)")
print(f"Covered receptors: {si_df['receptor'].nunique()}/23")
print("\nTop 10 selective candidates:")
print(si_df.head(10).to_string(index=False))
