"""
2560-dim pocket 임베딩 생성 (본문 Methods 정확히 재현)
======================================================
ESM-2 650M layer 33 (1280-dim) per-residue 추출.
- orthosteric pocket residue 평균 -> v_orth (1280)
- intracellular pocket residue 평균 -> v_intra (1280)  [TAS2R14만]
- intra가 비어있으면 v_intra = v_orth (본문 Methods)
- concat -> (2560,)
출력: tas2r_pocket_embeddings_v2.npy (23, 2560), tas2r_embedding_index_v2.json
"""
import json, numpy as np, torch, esm

SEQS = json.load(open('tas2r_sequences_bdb.json'))
POCKET = json.load(open('tas2r_pocket_definitions_v2.json'))
RECEPTORS = list(SEQS.keys())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}, 수용체 {len(RECEPTORS)}개")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
bc = alphabet.get_batch_converter()
model = model.eval().to(device)
if device=='cuda': model = model.half()

def pool(per_res, positions, seqlen):
    valid = [p-1 for p in positions if 1 <= p <= seqlen]
    if not valid: return None
    return per_res[torch.tensor(valid)].mean(0).numpy()

emb = np.zeros((len(RECEPTORS), 2560), dtype=np.float32)
with torch.no_grad():
    for i, rec in enumerate(RECEPTORS):
        seq = SEQS[rec]
        _,_,toks = bc([(rec, seq)]); toks = toks.to(device)
        rep = model(toks, repr_layers=[33])["representations"][33][0]
        per_res = rep[1:len(seq)+1].float().cpu()
        orth_pos = POCKET[rec].get('orthosteric', [])
        intra_pos = POCKET[rec].get('intracellular', [])
        v_orth = pool(per_res, orth_pos, len(seq))
        if v_orth is None: v_orth = per_res.mean(0).numpy()
        v_intra = pool(per_res, intra_pos, len(seq))
        if v_intra is None: v_intra = v_orth   # 본문 Methods: intra 없으면 orth 복제
        emb[i] = np.concatenate([v_orth, v_intra])
        tag = "dual" if intra_pos else "orth=intra"
        print(f"  {rec}: orth={len(orth_pos)} intra={len(intra_pos)} [{tag}] [{i+1}/23]")

np.save('tas2r_pocket_embeddings_v2.npy', emb)
json.dump({r:i for i,r in enumerate(RECEPTORS)}, open('tas2r_embedding_index_v2.json','w'))
print(f"\n저장: tas2r_pocket_embeddings_v2.npy {emb.shape}")
