# SpaAxFormer

SpaAxFormer is a Transformer-based framework for spatial transcriptomics that
represents each retained observation–gene pair as an entry-level token. Axial
attention models relationships across genes within each observation and applies
gene-specific spatial attention across neighboring observations. Contextualized
gene states support expression reconstruction, while a CLS token provides an
observation-level representation for spatial-domain identification.

This repository supports spatial-domain clustering and gene-expression imputation.
Training combines gene-expression prediction (GEP), CLS-conditioned gene-expression
prediction (GEPC), and optional weighted multi-positive spatial contrastive learning.
The objectives used depend on the task and configuration.

## Key features

- **Entry-level representation:** explicit correspondence between each spatial
  observation and retained gene.
- **Axial attention:** relationships among genes within observations and corresponding
  genes across spatial neighbors.
- **Dual-level representations:** contextualized gene states for reconstruction and
  CLS embeddings for observation-level analysis.
- **Multiple objectives:** GEP, GEPC, and weighted multi-positive spatial contrastive
  learning.
- **Clustering and imputation workflows:** spatial-domain clustering and
  fixed-holdout gene-expression imputation, with their evaluation paths.

## Installation

Create and activate the provided Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate spaaxformer-release
```

Spatial-domain clustering uses `mclust` through `rpy2`, so R and the `mclust`
package must also be available.

## Data preparation

Use `--data-root` (alias `--data_path`) for the root shown below and `--sample`
for the sample directory or supported filename.

**Clustering**

For DLPFC, use `--dataset Visium` with this layout:

```text
<data-root>/
└── <sample>/
    ├── <sample>_filtered_feature_bc_matrix.h5
    ├── spatial/
    └── <sample>_truth.txt
```

The truth file contains tab-separated barcodes and labels, without a header.
Supported sample IDs are `151507`–`151510` and `151669`–`151676`.
The Visium `spatial/` folder must contain `tissue_positions_list.csv`,
`scalefactors_json.json`, `tissue_hires_image.png` and `tissue_lowres_image.png`.

- **BC1:** use `--dataset Visium --sample section1`. Provide the same Visium
  layout under `<data-root>/section1/`, with
  `section1_filtered_feature_bc_matrix.h5`. Replace the truth file with
  `gt/tissue_positions_list_GTs.txt`: a headerless CSV with barcodes in the first
  column and integer labels in column numbered 6.
- **STARMap:** use `--dataset StarMap` and
  `--sample STARmap_20180505_BY3_1k.h5ad`, with that file directly under the root.
  The H5AD needs dense expression counts in `X`, coordinates in `obsm["spatial"]`,
  and exactly four `obs` columns ordered as total counts, image row, image column
  and cluster label.

**Imputation**

Provide prepared inputs in this layout:

```text
<data-root>/
└── data_split/
    └── <sample>/
        ├── filtered_adata.h5ad
        ├── split_0_test_mask.npz
        ├── split_1_test_mask.npz
        └── ...
```

The H5AD needs prepared, log-transformed expression in `X`, including the true
holdout values, and coordinates in `obsm["spatial"]`. Each mask file stores a
boolean `arr_0` array matching `X` in shape and observation/gene order; `True`
marks held-out entries. No additional Visium files or annotations are needed.
Use the prepared directory name as `--sample`, without a filename extension.

## Quickstart

Run from the repository root after activating the environment. Replace the data
paths with your own. `--device 0` selects CUDA device 0; use `--device -1` for CPU.

### Spatial-domain clustering

```bash
python -B src/main.py \
  --dataset Visium --sample 151507 --data-root /path/to/DLPFC \
  --task cluster --epochs 100 --repetitions 1 \
  --spatial-radius 150 --expression-pca-dim 100 \
  --contrast --contrastive-weight 0.02 --num-clusters 7 \
  --device 0 --output-dir results/dlpfc_example
```

Set `--num-clusters` to the number of domains for your sample.

### Gene-expression imputation

```bash
python -B src/main.py \
  --dataset Slide_seq_V2 --sample mouse_olfactory_bulb \
  --data-root /path/to/Slide_seq_V2 \
  --task imputation --no_gepc --epochs 100 --repetitions 1 \
  --imputation-split 0 --contrastive-weight 0 \
  --spatial-radius 150 --expression-pca-dim 100 \
  --device 0 --output-dir results/slide_imputation_example
```

For GEP-only imputation, use `--no_gepc` and omit `--contrast`.
`--contrastive-weight 0` is still required by the parser. Supply one existing
`--imputation-split` ID per repetition.

## Citation

If you use SpaAxFormer in your research, please cite our manuscript.

Citation information will be added after the bioRxiv preprint is posted.
