"""
Are pocket embeddings just a 'more separable' (one-hot-like) representation?
(Reviewer 1, round 2)
============================================================================
Reviewer hypothesis: the pocket embeddings might help simply because they make
the 23 receptors look more different from each other (closer to one-hot), rather
than because they carry useful *structural* information.

Test: compare the distribution of pairwise distances between the 23 receptor
vectors under (a) pocket embeddings and (b) full-sequence mean-pooled embeddings.
If pocket embeddings were merely 'more separable', their pairwise distances would
be systematically *larger* (receptors pushed apart). We report cosine and
Euclidean distances, normalised so the two representations are comparable, plus
the mean/median and the coefficient of variation.

Interpretation guide:
- If pocket pairwise distances are NOT systematically larger than full-sequence
  ones, the benefit is not explained by 'more separable / more one-hot-like'.
- The one-hot baseline has, by construction, identical pairwise distances between
  every pair (all orthogonal), i.e. CV = 0. A representation closer to one-hot
  would have a LOWER coefficient of variation (more uniform distances). We report
  CV so the reviewer can see whether pocket embeddings are more or less one-hot-like
  than full-sequence embeddings.

Inputs (no GPU needed):
  tas2r_pocket_embeddings_v2.npy   (23 x 2560; pocket, duplicated halves)
  emb_esm2_full.npy                (23 x 1280; full-sequence mean pool)
  tas2r_embedding_index_v2.json
Output: embedding_separability.json  (+ printed summary)
"""
import json, numpy as np
from scipy.spatial.distance import pdist
from scipy import stats

RECEPTORS = ['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7','TAS2R8','TAS2R9',
             'TAS2R10','TAS2R13','TAS2R14','TAS2R16','TAS2R38','TAS2R39','TAS2R40',
             'TAS2R41','TAS2R42','TAS2R43','TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']


def summarize(vecs, label):
    """Pairwise distance stats after per-representation normalisation."""
    # z-score each dimension so representations of different scale/dim are comparable
    X = (vecs - vecs.mean(0)) / (vecs.std(0) + 1e-8)
    cos = pdist(X, metric='cosine')
    euc = pdist(X, metric='euclidean')
    # normalise euclidean by mean so CV is scale-free
    out = {
        'n_pairs': int(len(cos)),
        'cosine_mean': round(float(cos.mean()), 4),
        'cosine_median': round(float(np.median(cos)), 4),
        'cosine_cv': round(float(cos.std() / (cos.mean() + 1e-12)), 4),
        'euclidean_mean': round(float(euc.mean()), 4),
        'euclidean_cv': round(float(euc.std() / (euc.mean() + 1e-12)), 4),
    }
    print(f"{label:16s}: cos mean={out['cosine_mean']:.3f} median={out['cosine_median']:.3f} "
          f"CV={out['cosine_cv']:.3f} | euc CV={out['euclidean_cv']:.3f}")
    return out, cos, euc


def main():
    idx = json.load(open("tas2r_embedding_index_v2.json"))
    order = [idx[r] for r in RECEPTORS]

    pocket = np.load("tas2r_pocket_embeddings_v2.npy")[order]
    # pocket halves are duplicated; use the first half (1280) as the receptor vector
    pocket = pocket[:, :1280]
    full = np.load("emb_esm2_full.npy")[order]

    print("Pairwise-distance distribution between the 23 receptor vectors")
    print("(z-scored per representation; one-hot would give CV=0):\n")
    p_stats, p_cos, p_euc = summarize(pocket, "pocket")
    f_stats, f_cos, f_euc = summarize(full, "full-sequence")

    # Are pocket distances systematically larger? (paired over the same 253 pairs)
    # pairs are in the same order for both, since receptor order is fixed.
    d = p_cos - f_cos
    w = stats.wilcoxon(p_cos, f_cos)
    result = {
        'pocket': p_stats, 'full_sequence': f_stats,
        'pocket_minus_full_cosine_mean': round(float(d.mean()), 4),
        'wilcoxon_p_pocket_vs_full_cosine': round(float(w.pvalue), 4),
        'note': ('Pocket embeddings are NOT systematically more separable than full-'
                 'sequence embeddings if pocket_minus_full is near zero / negative and '
                 'the CV is not lower. A lower CV would indicate a more one-hot-like '
                 '(uniform-distance) representation.')
    }
    print(f"\npocket - full (cosine, paired over 253 pairs): "
          f"mean {result['pocket_minus_full_cosine_mean']:+.3f}, Wilcoxon p={w.pvalue:.3f}")
    print("\nInterpretation:")
    if p_stats['cosine_cv'] < f_stats['cosine_cv']:
        print("  Pocket embeddings have LOWER distance CV -> somewhat more uniform/"
              "one-hot-like than full-sequence, but still far from true one-hot (CV=0).")
    else:
        print("  Pocket embeddings do NOT have lower distance CV than full-sequence,")
        print("  so the benefit is not explained by a more one-hot-like representation.")

    json.dump(result, open("embedding_separability.json", "w"), indent=2)
    print("\nSaved: embedding_separability.json")


if __name__ == "__main__":
    main()
