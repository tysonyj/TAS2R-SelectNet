"""
Precision-recall metrics for broadly-tuned subtypes (Supplementary Table S5b).
Reports AUC-ROC vs AUPRC against the no-skill prevalence baseline,
quantifying how ROC metrics can overstate performance under class imbalance.

Usage:
    python compute_auprc.py --data ../data/dataset.npz --out ../results/auprc.json
"""
import numpy as np, json, argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings; warnings.filterwarnings('ignore')

BROAD = ['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']

def main(data_path, out_path):
    d = np.load(data_path, allow_pickle=True)
    X, Y, is_train, R = d['X'], d['Y'], d['is_train'], list(d['receptors'])
    tr, te = np.where(is_train)[0], np.where(~is_train)[0]

    results, aucs, auprcs, prevs = {}, [], [], []
    for rec in BROAD:
        j = R.index(rec); ytr, yte = Y[tr, j], Y[te, j]
        if yte.sum() < 1:
            continue
        clf = RandomForestClassifier(n_estimators=300, random_state=0,
                                     n_jobs=-1, class_weight='balanced')
        clf.fit(X[tr], ytr)
        p = clf.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(yte, p))
        ap  = float(average_precision_score(yte, p))
        prev = float(yte.mean())
        aucs.append(auc); auprcs.append(ap); prevs.append(prev)
        results[rec] = {'auc': round(auc, 3), 'auprc': round(ap, 3),
                        'prevalence': round(prev, 3), 'npos': int(yte.sum())}
        print(f"{rec}: AUC={auc:.3f}  AUPRC={ap:.3f}  "
              f"(prevalence={prev:.3f}, npos={int(yte.sum())})")

    results['_mean'] = {
        'auc': round(float(np.mean(aucs)), 3),
        'auprc': round(float(np.mean(auprcs)), 3),
        'prevalence': round(float(np.mean(prevs)), 3),
        'auprc_lift_over_prevalence': round(float(np.mean(auprcs) / np.mean(prevs)), 1),
    }
    print(f"\nMean: AUC={results['_mean']['auc']}, AUPRC={results['_mean']['auprc']}, "
          f"lift={results['_mean']['auprc_lift_over_prevalence']}x")
    json.dump(results, open(out_path, 'w'), indent=2)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../data/dataset.npz')
    ap.add_argument('--out',  default='../results/auprc.json')
    a = ap.parse_args()
    main(a.data, a.out)
