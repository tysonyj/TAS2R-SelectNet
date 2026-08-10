# Revision 2 experiments (JCIM ci-2026-01950g)

Controls requested by Reviewer 1 in the second round of review. Running these
changed the paper's conclusion about pocket-restricted embeddings, so they are
documented here in full.

All scripts run from a directory containing the standard inputs
(`tas2r_pocket_embeddings_v2.npy`, `tas2r_embedding_index_v2.json`,
`tas2r_sequences_bdb.json`, `tas2r_pocket_definitions_v2.json`,
`morgan_fingerprints_v2.npy`, `fp_cid_index_v2.csv`,
`compounds_with_descriptors_v2.csv`, `ligandReceptors_2024.csv`,
`scaffold_split_v2.csv`, `ground_truth.json`).

## Headline result

The earlier ablation compared receptor representations at their native
dimensionality, where pocket pooling led by ΔAUC = +0.057 over full-sequence
pooling. Repeating the comparison with every representation mapped to the same
input width reverses the ordering:

| Representation | Native width | Matched width (2560) |
|---|---|---|
| ESM-2 full-sequence pooling | 0.663 | **0.759 ± 0.004** |
| ProtBERT | 0.687 | 0.748 ± 0.010 |
| One-hot | 0.748 ± 0.013 | 0.744 ± 0.011 |
| ESM-2 pocket pooling | 0.720 ± 0.024 | 0.741 ± 0.004 |
| ESM-2 random non-pocket residue | — | 0.735 ± 0.010 |
| Amino-acid composition | 0.658 | 0.729 ± 0.006 |
| ESM-2 single pocket residue | — | 0.721 ± 0.010 |

Both ablations use the same data, model, and split. They differ only in whether
receptor-encoder input width is held constant.

## 1. Unified ablation at matched input width

```bash
python analysis/run_unified_ablation.py
```
Maps every representation to a common 2560-dim input by tiling, holding the
compound encoder, bilinear head, scaffold split, loss, schedule, and seeds fixed.
Output: `unified_ablation.json`.

## 2. Random non-pocket residue control, and the TAS2R14 sub-pockets

```bash
python analysis/run_nonpocket_control.py --mode all
```
Represents each receptor by a single residue drawn at random from *outside* its
annotated pocket. Five draws give 0.735 ± 0.010, against 0.721 ± 0.010 for a
single central pocket residue and 0.741 ± 0.004 for the full pocket — an
arbitrary residue outside the pocket does as well as the pocket.

The same script encodes TAS2R14 by each of its two sub-pockets in turn. The
cholesterol-binding upper transmembrane site (0.738 ± 0.016) and the
intracellular agonist site (0.739 ± 0.011) are indistinguishable, although only
the latter binds the resolved bitter agonists.

Outputs: `nonpocket_control_results.json`, `t14_cholesterol_results.json`.

## 3. Are pocket embeddings more separable?

```bash
python analysis/run_embedding_separability.py
```
Pairwise distances between the 23 receptor vectors are statistically
indistinguishable between pocket and full-sequence embeddings (mean cosine 1.032
vs 1.037; Wilcoxon p = 0.93 over 253 pairs). Pocket embeddings are marginally
more uniform (distance CV 0.29 vs 0.33) but far from the uniform distances a
one-hot encoding would give. Output: `embedding_separability.json`.

## 4. Leave-one-receptor-out transfer, four representations

```bash
python analysis/run_loro_representations.py
```
Extends the round-1 LORO experiment, which compared pocket embeddings only
against a nearest-paralogue one-hot baseline, by adding full-sequence and random
non-pocket controls.

| | all (23) | nn identity < 0.5 (15) | nn identity ≥ 0.5 (8) |
|---|---|---|---|
| Pocket | 0.688 | 0.648 | 0.763 |
| Full sequence | 0.689 | 0.652 | 0.758 |
| Random non-pocket | 0.651 | 0.631 | 0.689 |
| NN one-hot | 0.664 | 0.608 | 0.768 |

Pocket and full-sequence embeddings transfer equally well. What survives is a
claim about protein language model embeddings in general, not about pockets:
a continuous sequence-derived vector beats substituting the nearest paralogue's
identity, and the margin is largest where no close paralogue exists.
Output: `loro_representations.json`.

## 5. Figures and the prediction table

```bash
python analysis/analysis1_leakage.py        # -> leakage_sim.json (panel data)
python figures/make_r2_figures.py           # -> three main-text figures
python analysis/make_reviewer2_figures.py   # -> AUC-vs-data-volume, split-distance boxplot
python analysis/make_prediction_spreadsheet.py
```

`make_prediction_spreadsheet.py` scores all 645 BitterDB compounds against all 23
receptors (14,835 pairs) and applies the Selectivity Index rule from the paper
(p > 0.5 and SI > 1.5 for exactly one receptor), giving 392 predicted
monoselective ligands across 15 receptor subtypes. It writes
`TAS2R_SelectNet_predictions_645x23.csv` / `.xlsx` and, if no checkpoint is
present, trains and saves `selectnet_production.pt` so the table is reproducible.
