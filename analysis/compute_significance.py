"""
Significance testing for the broadly-tuned tier DeltaAUC (main text Section 3.4).
Reports bootstrap 95% CI, sign-flip permutation test, and paired t-test
for the per-subtype SelectNet vs one-hot AUC difference (n=6 broad subtypes).

Usage:
    python compute_significance.py --gt ../results/ground_truth.json
"""
import json, argparse, numpy as np
from scipy import stats

BROAD = ['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']

def main(gt_path, n_boot=10000, seed=0):
    gt = json.load(open(gt_path))
    sn  = gt['per_subtype_selectnet']
    scf = gt['per_subtype_onehot_scaffold']

    sn_vals = np.array([sn[r]['auc'] for r in BROAD])
    oh_vals = np.array([scf[r]       for r in BROAD])
    diff = sn_vals - oh_vals
    obs = diff.mean()
    print(f"Per-subtype DeltaAUC: {dict(zip(BROAD, np.round(diff,3)))}")
    print(f"Mean DeltaAUC = {obs:+.4f}")

    t, p_t = stats.ttest_rel(sn_vals, oh_vals)
    print(f"Paired t-test:           t={t:.3f}, p={p_t:.3f}")
    try:
        w, p_w = stats.wilcoxon(sn_vals, oh_vals)
        print(f"Wilcoxon signed-rank:    W={w:.1f}, p={p_w:.3f}")
    except Exception as e:
        print("Wilcoxon:", e)

    rng = np.random.default_rng(seed)
    perms = np.array([(diff * rng.choice([-1, 1], size=len(diff))).mean()
                      for _ in range(n_boot)])
    p_perm = (np.abs(perms) >= np.abs(obs)).mean()
    print(f"Sign-flip permutation:   p={p_perm:.3f} ({n_boot} permutations)")

    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                      for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5])
    print(f"Bootstrap 95% CI:        [{ci[0]:+.3f}, {ci[1]:+.3f}] ({n_boot} resamples)")
    print(f"\nConclusion: {'non-significant (CI includes 0)' if ci[0] < 0 < ci[1] else 'significant'}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='../results/ground_truth.json')
    a = ap.parse_args()
    main(a.gt)
