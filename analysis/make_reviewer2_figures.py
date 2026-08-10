"""
Figure data for Reviewer 1 round 2:
  (A) per-receptor AUC vs number of training ligands, per representation
  (C) per-receptor distribution of test-compound -> nearest-train distances,
      random vs scaffold split

No GPU needed. Uses existing per-subtype results and Morgan fingerprints.

Inputs:
  ground_truth.json                (production per-subtype SelectNet + one-hot AUC)
  scaffold_split_results.json, random_split_results.json  (per-subtype baselines)
  morgan_fingerprints_v2.npy, fp_cid_index_v2.csv
  scaffold_split_v2.csv, ligandReceptors_2024.csv
Optional (if present, added as extra lines in A):
  ablation per-representation per-subtype AUCs are not stored per-subtype by default,
  so (A) shows one-hot (scaffold), SelectNet (pocket, scaffold), and one-hot (random).

Outputs:
  fig_auc_vs_nligands.png
  fig_split_distance_boxplot.png
  reviewer2_figuredata.json
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P_RED='#E8A7A0'; P_BLUE='#8FB8DC'; P_GREY='#B0B0B0'; P_MINT='#A8D5BA'
RECEPTORS = ['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7','TAS2R8','TAS2R9',
             'TAS2R10','TAS2R13','TAS2R14','TAS2R16','TAS2R38','TAS2R39','TAS2R40',
             'TAS2R41','TAS2R42','TAS2R43','TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']


def load_auc(path):
    d = json.load(open(path))
    out = {}
    for r in RECEPTORS:
        v = d.get(r)
        if isinstance(v, dict): v = v.get('auc')
        out[r] = v
    return out


def get_selectnet_auc():
    gt = json.load(open('ground_truth.json'))
    sn = gt.get('per_subtype_selectnet', {})
    oh = gt.get('per_subtype_onehot_scaffold', {})
    def val(d, r):
        v = d.get(r)
        return v.get('auc') if isinstance(v, dict) else v
    return ({r: val(sn, r) for r in RECEPTORS}, {r: val(oh, r) for r in RECEPTORS})


def n_train_ligands():
    df = pd.read_csv("ligandReceptors_2024.csv")
    human = [1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
    df = df[df['rID'].isin(human)].copy()
    df['receptor'] = df['rID'].map({r:f'TAS2R{r}' for r in human})
    split = pd.read_csv("scaffold_split_v2.csv")
    train = set(split[split['split']=='train']['bdb_cid'])
    n = {}
    for r in RECEPTORS:
        pos = set(df[df['receptor']==r]['cID'].unique())
        n[r] = len(pos & train)
    return n


def panel_A():
    sn, oh_scaf = get_selectnet_auc()
    oh_rand = load_auc('random_split_results.json')
    nlig = n_train_ligands()

    xs = np.array([nlig[r] for r in RECEPTORS], dtype=float)
    order = np.argsort(xs)
    xs_s = xs[order]
    recs_s = [RECEPTORS[i] for i in order]

    def series(d): return np.array([d.get(r, np.nan) for r in recs_s], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for d, name, col, mk in [
        (oh_rand, 'One-hot (random split)', P_GREY, 's'),
        ({r: oh_scaf[r] for r in RECEPTORS}, 'One-hot (scaffold split)', P_BLUE, 'o'),
        (sn, 'TAS2R-SelectNet, pocket (scaffold split)', P_RED, '^'),
    ]:
        y = series(d)
        m = ~np.isnan(y)
        ax.plot(xs_s[m], y[m], mk, color=col, ms=7, label=name, alpha=0.85, markeredgecolor='#555', lw=0)
        # trend line (log x)
        if m.sum() > 2:
            lx = np.log10(xs_s[m] + 1)
            z = np.polyfit(lx, y[m], 1)
            xx = np.linspace(xs_s[m].min(), xs_s[m].max(), 100)
            ax.plot(xx, np.polyval(z, np.log10(xx + 1)), '-', color=col, alpha=0.5, lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('Number of training ligands for the receptor')
    ax.set_ylabel('Per-receptor AUC-ROC')
    ax.axhline(0.5, ls=':', color='gray', alpha=0.5)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('Per-receptor AUC vs training-set size', fontsize=11)
    plt.tight_layout(); plt.savefig('fig_auc_vs_nligands.png', dpi=200, bbox_inches='tight'); plt.close()
    print("Saved: fig_auc_vs_nligands.png")
    return {'n_train_ligands': nlig, 'selectnet': sn, 'onehot_scaffold': oh_scaf, 'onehot_random': oh_rand}


def panel_C():
    """Per-receptor test->nearest-train Tanimoto distance, random vs scaffold."""
    fp = np.load("morgan_fingerprints_v2.npy").astype(bool)
    cids = pd.read_csv("fp_cid_index_v2.csv")['bdb_cid'].tolist()
    cid2i = {c:i for i,c in enumerate(cids)}
    df = pd.read_csv("ligandReceptors_2024.csv")
    human = [1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
    df = df[df['rID'].isin(human)].copy()
    df['receptor'] = df['rID'].map({r:f'TAS2R{r}' for r in human})
    scaf = pd.read_csv("scaffold_split_v2.csv")
    strain = set(scaf[scaf['split']=='train']['bdb_cid']); stest = set(scaf[scaf['split']=='test']['bdb_cid'])
    try:
        rnd = json.load(open('random_split_assignment.json'))
        rtrain = set(rnd['train']); rtest = set(rnd['test'])
    except FileNotFoundError:
        # fall back: random split with same test size, seeded
        allc = list(strain | stest); rng = np.random.default_rng(0)
        rng.shuffle(allc); k = len(stest)
        rtest = set(allc[:k]); rtrain = set(allc[k:])

    def tanimoto_nn(query_cids, train_cids):
        tr = [cid2i[c] for c in train_cids if c in cid2i]
        if not tr: return []
        TR = fp[tr]
        out = []
        for c in query_cids:
            if c not in cid2i: continue
            q = fp[cid2i[c]]
            inter = (TR & q).sum(1); union = (TR | q).sum(1)
            sim = inter / np.clip(union, 1, None)
            out.append(float(sim.max()))   # nearest-train similarity
        return out

    data = {}
    for r in RECEPTORS:
        pos = set(df[df['receptor']==r]['cID'].unique())
        s_sim = tanimoto_nn(pos & stest, strain)
        r_sim = tanimoto_nn(pos & rtest, rtrain)
        data[r] = {'scaffold_sim': s_sim, 'random_sim': r_sim,
                   'n_test_scaffold': len(s_sim), 'n_test_random': len(r_sim)}

    # boxplot: distance = 1 - similarity, two boxes per receptor, only receptors with data
    recs = [r for r in RECEPTORS if data[r]['n_test_scaffold'] >= 3]
    fig, ax = plt.subplots(figsize=(12, 5))
    pos_i = np.arange(len(recs))
    scaf_d = [[1-x for x in data[r]['scaffold_sim']] for r in recs]
    rand_d = [[1-x for x in data[r]['random_sim']] for r in recs]
    bp1 = ax.boxplot(scaf_d, positions=pos_i*3, widths=0.9, patch_artist=True,
                     boxprops=dict(facecolor=P_RED, alpha=0.7), medianprops=dict(color='#333'), showfliers=False)
    bp2 = ax.boxplot(rand_d, positions=pos_i*3+1, widths=0.9, patch_artist=True,
                     boxprops=dict(facecolor=P_BLUE, alpha=0.7), medianprops=dict(color='#333'), showfliers=False)
    ax.set_xticks(pos_i*3 + 0.5); ax.set_xticklabels([r.replace('TAS2R','T') for r in recs], rotation=45, fontsize=8)
    ax.set_ylabel('Test compound distance to nearest training compound\n(1 - Tanimoto)')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Scaffold split', 'Random split'], fontsize=9, loc='lower right')
    ax.set_title('Per-receptor test-to-train chemical distance by split', fontsize=11)
    plt.tight_layout(); plt.savefig('fig_split_distance_boxplot.png', dpi=200, bbox_inches='tight'); plt.close()
    print("Saved: fig_split_distance_boxplot.png")

    summary = {r: {'scaffold_median_dist': round(1-np.median(data[r]['scaffold_sim']),3) if data[r]['scaffold_sim'] else None,
                   'random_median_dist': round(1-np.median(data[r]['random_sim']),3) if data[r]['random_sim'] else None,
                   'n_test_scaffold': data[r]['n_test_scaffold']}
               for r in RECEPTORS}
    return summary


def main():
    print("=== Panel A: AUC vs n training ligands ===")
    a = panel_A()
    print("\n=== Panel C: per-receptor test-train distance ===")
    c = panel_C()
    json.dump({'panelA': a, 'panelC': c}, open('reviewer2_figuredata.json','w'), indent=2)
    print("\nSaved: reviewer2_figuredata.json")


if __name__ == "__main__":
    main()
