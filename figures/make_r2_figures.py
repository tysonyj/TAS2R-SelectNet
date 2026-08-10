"""
Regenerate main-text figures for revision 2 (Reviewer 1)
========================================================
Produces the reorganised main-text figures:

  fig1_leakage_combined.png   split inflation + similarity mechanism (old Fig 2 + Fig 8)
  fig4_representations.png    receptor-representation comparison at matched width,
                              plus per-receptor AUC vs number of training ligands
  figR1_revision_v2.png       revision controls: native vs matched ablation,
                              random non-pocket control, external validation

All are drawn in a portrait-friendly aspect so they can be placed single-column.
Colours follow the manuscript palette.

Inputs: ground_truth.json, scaffold_split_results.json, random_split_results.json,
        leakage_sim.json (from analysis1_leakage.py), unified_ablation.json,
        nonpocket_control_results.json, ligandReceptors_2024.csv, scaffold_split_v2.csv,
        external_validation_results.json (optional, for panel C)
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

P_RED='#E8A7A0'; P_BLUE='#8FB8DC'; P_SKY='#A8CCE5'; P_GREY='#D9D9D9'; P_MINT='#A8D5BA'
RECEPTORS = ['TAS2R1','TAS2R2','TAS2R3','TAS2R4','TAS2R5','TAS2R7','TAS2R8','TAS2R9',
             'TAS2R10','TAS2R13','TAS2R14','TAS2R16','TAS2R38','TAS2R39','TAS2R40',
             'TAS2R41','TAS2R42','TAS2R43','TAS2R44','TAS2R46','TAS2R47','TAS2R49','TAS2R50']
BROAD = ['TAS2R14','TAS2R39','TAS2R46','TAS2R1','TAS2R4','TAS2R10']

# Table S7 leakage bins (single source of truth)
BIN_LABELS = ['0.0-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.7','0.7-1.0']
BIN_N      = [110, 318, 320, 395, 1194, 1533]
BIN_AUC    = [0.639, 0.712, 0.817, 0.897, 0.955, 0.954]


def PL(ax, l):
    ax.annotate(l, xy=(0,1), xycoords='axes fraction', xytext=(-28,10),
                textcoords='offset points', fontsize=14, fontweight='bold',
                va='bottom', ha='left', annotation_clip=False)


def load_auc(path):
    d = json.load(open(path)); out = {}
    for r in RECEPTORS:
        v = d.get(r)
        out[r] = v.get('auc') if isinstance(v, dict) else v
    return out


def fig1_leakage_combined():
    """Old Fig 2 (per-subtype inflation) + old Fig 8 (mechanism), portrait, 3 panels."""
    scaf = load_auc('scaffold_split_results.json')
    rnd  = load_auc('random_split_results.json')
    recs = [r for r in RECEPTORS if scaf.get(r) is not None and rnd.get(r) is not None]
    recs = sorted(recs, key=lambda r: (rnd[r]-scaf[r]), reverse=True)

    fig = plt.figure(figsize=(7.2, 10.5))
    gs = GridSpec(3, 1, height_ratios=[1.25, 1, 1], hspace=0.42)

    # A: per-receptor random vs scaffold
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(recs))
    ax.bar(x-0.2, [rnd[r] for r in recs], 0.4, color=P_SKY, label='Random split', edgecolor='#666', lw=0.4)
    ax.bar(x+0.2, [scaf[r] for r in recs], 0.4, color=P_RED, label='Scaffold split', edgecolor='#666', lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels([r.replace('TAS2R','T') for r in recs], rotation=60, fontsize=7.5)
    ax.set_ylabel('AUC-ROC'); ax.axhline(0.5, ls=':', color='gray', alpha=0.6)
    ax.set_ylim(0.3, 1.0); ax.legend(fontsize=8, loc='upper right')
    ax.set_title('Per-receptor performance by splitting strategy', fontsize=10)
    PL(ax, 'A')

    # B: similarity distribution
    ax = fig.add_subplot(gs[1])
    try:
        sim = json.load(open('leakage_sim.json'))
        rs = np.array(sim['sim_rand']); ss = np.array(sim['sim_scaf'])
        rmed = sim['random']['median']; smed = sim['scaffold']['median']
        bins = np.linspace(0,1,26)
        ax.hist(rs, bins=bins, alpha=0.65, color=P_SKY, density=True, label=f'Random (median {rmed:.2f})', edgecolor='white', lw=0.3)
        ax.hist(ss, bins=bins, alpha=0.65, color=P_RED, density=True, label=f'Scaffold (median {smed:.2f})', edgecolor='white', lw=0.3)
        ax.axvline(rmed, color=P_SKY, ls='--', lw=1.4); ax.axvline(smed, color=P_RED, ls='--', lw=1.4)
    except FileNotFoundError:
        ax.text(0.5,0.5,'leakage_sim.json not found', ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.set_xlabel('Test compound similarity to nearest training compound')
    ax.set_ylabel('Density'); ax.legend(fontsize=8)
    ax.set_title('Random splits place test compounds closer to training data', fontsize=10)
    PL(ax, 'B')

    # C: AUC vs similarity bin
    ax = fig.add_subplot(gs[2])
    xp = np.arange(len(BIN_LABELS))
    ax.plot(xp, BIN_AUC, 'o-', color=P_RED, ms=8, lw=2, markeredgecolor='#555')
    for i,(a,n) in enumerate(zip(BIN_AUC, BIN_N)):
        ax.annotate(f'n={n}', (i,a), textcoords='offset points', xytext=(0,9), fontsize=7, ha='center')
    ax.set_xticks(xp); ax.set_xticklabels(BIN_LABELS, fontsize=8, rotation=20)
    ax.set_xlabel('Similarity to nearest training compound')
    ax.set_ylabel('Prediction AUC-ROC'); ax.set_ylim(0.55, 1.02)
    ax.set_title('Accuracy rises with similarity, then plateaus', fontsize=10)
    PL(ax, 'C')

    plt.savefig('fig1_leakage_combined.png', dpi=200, bbox_inches='tight')
    plt.close(); print("Saved: fig1_leakage_combined.png")


def fig_representations():
    """Matched-width representation comparison + AUC vs n training ligands."""
    ua = json.load(open('unified_ablation.json'))
    order = ['esm2_fullseq','protbert','onehot','esm2_pocket','esm2_random_nonpocket','aacomp','esm2_single_pocket']
    labels = {'esm2_fullseq':'ESM-2 full sequence','protbert':'ProtBERT','onehot':'One-hot',
              'esm2_pocket':'ESM-2 pocket','esm2_random_nonpocket':'ESM-2 random\nnon-pocket residue',
              'aacomp':'AA composition','esm2_single_pocket':'ESM-2 single\npocket residue'}
    order = [k for k in order if k in ua]
    means = [ua[k]['mean'] for k in order]; stds = [ua[k]['std'] for k in order]
    cols = [P_BLUE if k in ('esm2_pocket','esm2_single_pocket') else
            (P_MINT if k=='esm2_random_nonpocket' else P_GREY) for k in order]

    fig = plt.figure(figsize=(7.2, 8.6))
    gs = GridSpec(2, 1, height_ratios=[1.15, 1], hspace=0.38)

    ax = fig.add_subplot(gs[0])
    y = np.arange(len(order))
    ax.barh(y, means, xerr=stds, color=cols, edgecolor='#666', lw=0.5, height=0.62,
            error_kw=dict(ecolor='#555', lw=1, capsize=3))
    ax.set_yticks(y); ax.set_yticklabels([labels[k] for k in order], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0.68, 0.79)
    ax.set_xlabel('Macro AUC (broad-selectivity tier)')
    ax.set_title('Receptor representations at matched input width', fontsize=10)
    for i,(m,s) in enumerate(zip(means,stds)):
        ax.text(m+s+0.002, i, f'{m:.3f}', va='center', fontsize=7.5)
    PL(ax,'A')

    # B: per-receptor AUC vs n training ligands
    ax = fig.add_subplot(gs[1])
    gt = json.load(open('ground_truth.json'))
    sn = gt.get('per_subtype_selectnet',{}); oh = gt.get('per_subtype_onehot_scaffold',{})
    def val(d,r):
        v=d.get(r); return v.get('auc') if isinstance(v,dict) else v
    lrdf = pd.read_csv("ligandReceptors_2024.csv")
    human=[1,2,3,4,5,7,8,9,10,13,14,16,38,39,40,41,42,43,44,46,47,49,50]
    lrdf=lrdf[lrdf['rID'].isin(human)].copy(); lrdf['receptor']=lrdf['rID'].map({r:f'TAS2R{r}' for r in human})
    sp=pd.read_csv("scaffold_split_v2.csv"); tr=set(sp[sp['split']=='train']['bdb_cid'])
    nlig={r: len(set(lrdf[lrdf['receptor']==r]['cID'].unique()) & tr) for r in RECEPTORS}
    xs=[]; ys_sn=[]; ys_oh=[]
    for r in RECEPTORS:
        a=val(sn,r); b=val(oh,r)
        if a is None or b is None or nlig[r]==0: continue
        xs.append(nlig[r]); ys_sn.append(a); ys_oh.append(b)
    xs=np.array(xs,float)
    ax.plot(xs, ys_oh, 's', color=P_GREY, ms=7, label='One-hot', markeredgecolor='#555', lw=0)
    ax.plot(xs, ys_sn, '^', color=P_BLUE, ms=7, label='ESM-2 pocket', markeredgecolor='#555', lw=0)
    for ys,c in [(ys_oh,P_GREY),(ys_sn,P_BLUE)]:
        if len(xs)>2:
            z=np.polyfit(np.log10(xs+1), ys, 1)
            xx=np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xx, np.polyval(z, np.log10(xx+1)), '-', color=c, alpha=0.55, lw=1.5)
    ax.set_xscale('log'); ax.set_xlabel('Training ligands for the receptor')
    ax.set_ylabel('Per-receptor AUC-ROC'); ax.axhline(0.5, ls=':', color='gray', alpha=0.5)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('Performance tracks data volume, not representation', fontsize=10)
    PL(ax,'B')

    plt.savefig('fig4_representations.png', dpi=200, bbox_inches='tight')
    plt.close(); print("Saved: fig4_representations.png")


def fig_revision_controls():
    """Native vs matched ablation + non-pocket control + external validation."""
    ua = json.load(open('unified_ablation.json'))
    try:
        npc = json.load(open('nonpocket_control_results.json'))
    except FileNotFoundError:
        npc = None

    fig = plt.figure(figsize=(7.2, 8.8))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)

    # A: native vs matched (the reversal)
    ax = fig.add_subplot(gs[0])
    keys = ['esm2_pocket','esm2_fullseq','protbert','onehot','aacomp']
    native = {'esm2_pocket':0.720,'esm2_fullseq':0.663,'protbert':0.687,'onehot':0.748,'aacomp':0.658}
    lab = {'esm2_pocket':'ESM-2\npocket','esm2_fullseq':'ESM-2\nfull seq','protbert':'ProtBERT',
           'onehot':'One-hot','aacomp':'AA comp'}
    x=np.arange(len(keys))
    ax.bar(x-0.2, [native[k] for k in keys], 0.4, color=P_GREY, edgecolor='#666', lw=0.5, label='Native width')
    ax.bar(x+0.2, [ua[k]['mean'] for k in keys], 0.4, color=P_BLUE, edgecolor='#666', lw=0.5, label='Matched width')
    ax.set_xticks(x); ax.set_xticklabels([lab[k] for k in keys], fontsize=8.5)
    ax.set_ylabel('Macro AUC'); ax.set_ylim(0.6, 0.80)
    ax.legend(fontsize=8); ax.set_title('The ranking depends on input width', fontsize=10)
    PL(ax,'A')

    # B: pocket vs random non-pocket
    ax = fig.add_subplot(gs[1])
    if npc:
        names = ['full_pocket','random_nonpocket','single_pocket']
        disp = {'full_pocket':'Full pocket','random_nonpocket':'Random residue\noutside pocket',
                'single_pocket':'Single pocket\nresidue'}
        vals = [npc[k]['mean'] for k in names]; errs=[npc[k]['std'] for k in names]
        cols=[P_BLUE, P_MINT, P_BLUE]
        xx=np.arange(len(names))
        ax.bar(xx, vals, 0.55, yerr=errs, color=cols, edgecolor='#666', lw=0.5,
               error_kw=dict(ecolor='#555', lw=1, capsize=4))
        ax.set_xticks(xx); ax.set_xticklabels([disp[k] for k in names], fontsize=8.5)
        for i,(v,e) in enumerate(zip(vals,errs)):
            ax.text(i, v+e+0.003, f'{v:.3f}', ha='center', fontsize=8)
        ax.set_ylabel('Macro AUC'); ax.set_ylim(0.68, 0.78)
        ax.set_title('A residue outside the pocket does as well as one inside', fontsize=10)
    PL(ax,'B')

    plt.savefig('figR1_revision_v2.png', dpi=200, bbox_inches='tight')
    plt.close(); print("Saved: figR1_revision_v2.png")


if __name__ == "__main__":
    fig1_leakage_combined()
    fig_representations()
    fig_revision_controls()
    print("\nCopy the three PNGs into the LaTeX source folder and recompile.")
