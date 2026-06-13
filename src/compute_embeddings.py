"""
ESM-2 650M 임베딩 (full-seq + pocket) + AA composition
=======================================================
로컬 GPU(RTX 5060 Ti 16GB)에서 실행. CPU에서도 동작하나 650M은 느림.
CPU 검증판(150M)과 동일 로직, 모델만 650M(layer 33, 1280-dim)으로 교체.
실행:  python compute_embeddings.py
출력:  emb_esm2_full.npy (23x1280), emb_esm2_pocket.npy (23x1280),
       emb_aa_comp.npy (23x20), receptor_order.json
"""
import json, numpy as np, torch, esm

USE_FP16 = True   # 16GB면 fp32도 가능하나 fp16이 빠르고 안전

SEQS = json.load(open('tas2r_sequences_bdb.json'))
POCKET = json.load(open('tas2r_pocket_definitions_v2.json'))
RECEPTORS = list(SEQS.keys())
print(f"받은 수용체: {len(RECEPTORS)}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device=='cuda' else ""))

print("ESM-2 650M 로딩...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
bc = alphabet.get_batch_converter()
model = model.eval().to(device)
if USE_FP16 and device == 'cuda':
    model = model.half()
ESM_DIM, N_LAYER = 1280, 33

def get_pocket_residues(rec):
    v = POCKET[rec]; res = []
    if 'orthosteric' in v: res += v['orthosteric']
    if 'intracellular' in v: res += v['intracellular']
    if 'residues' in v: res += v['residues']
    return sorted(set(res))

emb_full   = np.zeros((len(RECEPTORS), ESM_DIM), dtype=np.float32)
emb_pocket = np.zeros((len(RECEPTORS), ESM_DIM), dtype=np.float32)
emb_aa     = np.zeros((len(RECEPTORS), 20), dtype=np.float32)
AA = "ACDEFGHIKLMNPQRSTVWY"

with torch.no_grad():
    for i, rec in enumerate(RECEPTORS):
        seq = SEQS[rec]
        for j, a in enumerate(AA):
            emb_aa[i, j] = seq.count(a) / len(seq)
        _, _, toks = bc([(rec, seq)])
        toks = toks.to(device)
        rep = model(toks, repr_layers=[N_LAYER])["representations"][N_LAYER][0]
        per_res = rep[1:len(seq)+1].float().cpu()
        emb_full[i] = per_res.mean(0).numpy()
        pocket_res = [p for p in get_pocket_residues(rec) if 1 <= p <= len(seq)]
        if pocket_res:
            idx = torch.tensor([p-1 for p in pocket_res])
            emb_pocket[i] = per_res[idx].mean(0).numpy()
        else:
            emb_pocket[i] = emb_full[i]
        print(f"  {rec}: pocket {len(pocket_res)} residues  [{i+1}/{len(RECEPTORS)}]")

np.save('emb_esm2_full.npy', emb_full)
np.save('emb_esm2_pocket.npy', emb_pocket)
np.save('emb_aa_comp.npy', emb_aa)
json.dump(RECEPTORS, open('receptor_order.json', 'w'))
print(f"\n저장 완료 (650M, {ESM_DIM}-dim)")
print("full norm 평균:", round(float(np.linalg.norm(emb_full, axis=1).mean()), 2))
print("pocket norm 평균:", round(float(np.linalg.norm(emb_pocket, axis=1).mean()), 2))
