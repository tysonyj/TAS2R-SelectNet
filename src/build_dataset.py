"""화합물 feature(2060) + 23-수용체 multilabel 행렬 구성, scaffold split 적용."""
import pandas as pd, numpy as np, json

RECEPTORS = json.load(open('receptor_order.json'))  # ['TAS2R1',...]
RID = [int(r.replace('TAS2R','')) for r in RECEPTORS]
rid2col = {rid:i for i,rid in enumerate(RID)}

# fingerprint (645 x 2048), 순서는 fp_cid_index_v2.csv
fp = np.load('morgan_fingerprints_v2.npy').astype(np.float32)
fpidx = pd.read_csv('fp_cid_index_v2.csv')['bdb_cid'].tolist()
cid2row = {c:i for i,c in enumerate(fpidx)}

# descriptor (12) — compounds_with_descriptors_v2.csv
desc = pd.read_csv('compounds_with_descriptors_v2.csv')
desc_cols = ['mw','alogp','tpsa','hbd','hba','rotb','rings','arom_rings',
             'fsp3','formal_charge','heavy_atoms','globularity_proxy']
desc = desc.set_index('bdb_cid')
# 정규화 전 원본 보관
D = np.zeros((len(fpidx), 12), dtype=np.float32)
for i,c in enumerate(fpidx):
    if c in desc.index:
        D[i] = desc.loc[c, desc_cols].values.astype(np.float32)
# z-score (descriptor만)
D = (D - D.mean(0)) / (D.std(0) + 1e-8)
X = np.concatenate([fp, D], axis=1)  # (645, 2060)

# label 행렬 (645 x 23): human TAS2R association만
lr = pd.read_csv('ligandReceptors_2024.csv')
lr = lr[lr['rID'].isin(RID)]
Y = np.zeros((len(fpidx), len(RECEPTORS)), dtype=np.float32)
n_assoc = 0
for _,row in lr.iterrows():
    c, rid = row['cID'], int(row['rID'])
    if c in cid2row and rid in rid2col:
        Y[cid2row[c], rid2col[rid]] = 1.0
        n_assoc += 1
print(f"X: {X.shape}, Y: {Y.shape}, association(human): {n_assoc}")
print(f"수용체별 양성 수: {dict(zip(RECEPTORS, Y.sum(0).astype(int)))}")

# scaffold split
split = pd.read_csv('scaffold_split_v2.csv').set_index('bdb_cid')['split'].to_dict()
is_train = np.array([split.get(c,'train')=='train' for c in fpidx])
print(f"train {is_train.sum()}, test {(~is_train).sum()}")

np.savez('dataset.npz', X=X, Y=Y, is_train=is_train,
         receptors=np.array(RECEPTORS), cids=np.array(fpidx))
print("저장: dataset.npz")
