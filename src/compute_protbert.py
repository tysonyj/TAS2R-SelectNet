"""ProtBERT pooled 임베딩 (다른 PLM 대조군)."""
import json, numpy as np, torch, re
from transformers import BertModel, BertTokenizer
SEQS = json.load(open('tas2r_sequences_bdb.json'))
RECEPTORS = json.load(open('receptor_order.json'))
print("ProtBERT 로딩...")
tok = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert").eval()
emb = np.zeros((len(RECEPTORS), 1024), dtype=np.float32)
with torch.no_grad():
    for i, rec in enumerate(RECEPTORS):
        seq = " ".join(re.sub(r"[UZOB]", "X", SEQS[rec]))
        ids = tok(seq, return_tensors="pt")
        out = model(**ids).last_hidden_state[0]
        emb[i] = out[1:-1].mean(0).numpy()
        print(f"  {rec} [{i+1}/{len(RECEPTORS)}]")
np.save('emb_protbert.npy', emb)
print("저장: emb_protbert.npy", emb.shape)
