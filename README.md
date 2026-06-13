# TAS2R-SelectNet

Code and data for **"Benchmarking TAS2R Subtype-Selectivity Prediction under Scaffold-Aware Evaluation: A Controlled Analysis of Receptor Representations."**

TAS2R-SelectNet is a bilinear interaction model that predicts which human bitter taste receptor (TAS2R) subtype a compound activates, encoding receptors with ESM-2 binding-pocket embeddings instead of one-hot vectors. This repository reproduces every analysis, table, and figure in the manuscript.

## Repository layout

```
.
├── src/                  # dataset construction, embeddings, model training
├── analysis/             # the analyses reported in the paper
├── figures/              # figure-generation scripts
├── data/                 # curated inputs (BitterDB associations, splits, pockets)
├── results/              # precomputed analysis outputs (JSON/CSV)
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/park-lab-snu/TAS2R-SelectNet.git
cd TAS2R-SelectNet
pip install -r requirements.txt
```

A CUDA GPU is recommended for embedding generation and model training (developed on an NVIDIA RTX 5060 Ti, 16 GB). Analyses that consume the precomputed `results/` files run on CPU.

## Data sources

- **TAS2R agonist associations**: BitterDB 2024 (Niv laboratory, Hebrew University of Jerusalem). `data/ligandReceptors_2024.csv` contains the 1,114 curated compound–receptor associations across 23 subtypes used here.
- **Receptor sequences / pocket definitions**: UniProt accessions and binding-pocket residue sets (cryo-EM for TAS2R14/46; mutagenesis for TAS2R38/16; AlphaFold2-based topology for the remaining 19) are in `data/tas2r_*.json`.
- **Compound descriptors / scaffold split**: `data/compounds_with_descriptors_v2.csv`, `data/scaffold_split_v2.csv`.

> Note on labels: BitterDB records confirmed agonists but not systematically-tested inactives, so unreported compound–receptor pairs are treated as negatives (a positive–unlabelled setting). Absolute metrics are therefore lower bounds; see the manuscript Methods.

## Reproducing the pipeline

### 1. Build dataset and embeddings (`src/`)

| Step | Script | Output |
|------|--------|--------|
| Curate associations, fingerprints, descriptors, scaffold split | `src/build_dataset.py` | `dataset.npz` |
| ESM-2 per-residue embeddings (full-sequence + pocket) | `src/compute_embeddings.py` | `emb_esm2_*.npy` |
| ProtBERT baseline embeddings | `src/compute_protbert.py` | `emb_protbert.npy` |
| Assemble 2560-dim pocket embeddings (dual pocket for TAS2R14) | `src/build_pocket_embeddings_2560.py` | `tas2r_pocket_embeddings_v2.npy` |

### 2. Train and evaluate (`src/`)

| Step | Script |
|------|--------|
| Train TAS2R-SelectNet (bilinear, focal loss, two-stage cosine schedule) | `src/train_selectnet_v2.py` |
| Full production run (scaffold split, per-subtype AUC, candidates) | `src/run_production.py` |

`run_production.py` writes `results/ground_truth.json`, the single source of truth for all per-subtype metrics reported in the paper.

### 3. Analyses (`analysis/`)

| Manuscript item | Script | Output |
|-----------------|--------|--------|
| §3.2 Similarity leakage: nearest-train Tanimoto (random vs scaffold) | `analysis/analysis1_leakage.py` | `results/leakage_sim.json` |
| §3.2 Leakage curve: AUC vs similarity bin (Fig 8B) | `analysis/leakage_curve.py` | `results/leakage_curve.json` |
| §3.4 Broad-tier significance: bootstrap CI, permutation, paired t | `analysis/compute_significance.py` | console |
| Table S5b: AUC-ROC vs AUPRC under class imbalance | `analysis/compute_auprc.py` | `results/auprc.json` |
| §3.5 Receptor-encoding ablation (Table S9, Fig 6) | `analysis/run_ablation_multiseed.py` | `results/ablation_multiseed.json` |
| §3.6 External validation on held-out agonists (Table S10) | `analysis/run_external_validation_v2.py` | `results/external_validation_predictions_v2.csv` |
| §3.8 Selectivity Index candidates | `analysis/recompute_candidates.py` | candidate lists |

Example (uses precomputed results, CPU-only):

```bash
cd analysis
python compute_significance.py --gt ../results/ground_truth.json
python compute_auprc.py --data ../data/dataset.npz --out ../results/auprc.json
```

### 4. Figures (`figures/`)

| Figure | Script |
|--------|--------|
| Fig 1 (dataset characteristics) | `figures/regen_fig1.py` |
| Fig 2 (scaffold-split inflation) | `figures/regen_fig2.py` |
| Fig 3 / Fig 6 / Fig S4 (per-subtype, training, broad tier) | `figures/regenerate_all_figures.py` |
| Fig 8 (similarity-leakage mechanism) | `figures/make_leakage_fig.py` |
| Fig S3 (Murcko scaffold split) | `figures/regen_figS3.py` |
| Shared pastel palette / panel labels | `figures/regen_pastel.py` |

## Key reported values

| Quantity | Value |
|----------|-------|
| Associations / compounds / subtypes | 1,114 / 645 / 23 |
| Scaffold split | 282 scaffolds, 607 compounds (train 526 / test 81) |
| Random-split AUC inflation (broad tier) | up to +0.21 (TAS2R39 +0.210, TAS2R14 +0.188) |
| Similarity–correctness correlation | Spearman r = 0.25, p = 1.6×10⁻¹⁸ |
| Broad-tier ΔAUC (SelectNet − one-hot) | +0.041 (95% CI [−0.035, +0.116]; permutation p = 0.41) |
| Ablation: pocket vs full-seq / ProtBERT / AA-comp | +0.057 / +0.033 / +0.062 |
| All-evaluable macro AUC | 0.628 (n = 15) |

## Citation

If you use this code or data, please cite the manuscript (citation to be added upon publication).

## License

MIT (see `LICENSE`). BitterDB data are redistributed under their original terms; please also cite BitterDB 2024.
