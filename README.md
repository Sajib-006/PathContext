# PathContext

Code and compact reproducibility artifacts for:

**Patch-Level Tissue Context Improves Learning from Frozen Pathology Foundation Model Embeddings**

The paper studies lightweight tissue-context modeling for cell-level classification from frozen pathology foundation model embeddings. The main experiments use the BRCA-M2C breast cancer histopathology dataset and compare cell-only frozen embedding baselines against patch-context fusion, graph-context propagation, local-neighbor context, hybrid morphology-context features, and multi-encoder fusion.

## Repository Contents

- `src/run_brca_m2c_eval.py` extracts frozen cell embeddings from BRCA-M2C crops and evaluates baseline analyses.
- `src/pfm_context_benchmark_full_brca_m2c.py` runs the main downstream benchmark, context methods, multi-encoder fusion, and ablations from cached embeddings.
- `src/run_acmbcb_plot.py` regenerates paper-ready summary plots from the benchmark CSV.
- `src/run_acmbcb_qual.py` generates qualitative prediction panels for cell-only, patch-context, and graph-context methods.
- `data/splits/` contains the BRCA-M2C train/validation/test split lists used in the paper.
- `data/brca_m2c_dataset_stats.csv` contains per-patch dataset statistics used for reporting split and class distributions.
- `results/brca_m2c/` contains compact CSV/JSON outputs from the paper benchmark.
- `figures/brca_m2c/` contains generated figures derived from the benchmark CSVs.

Raw histology images, labels, cached embedding arrays, trained model checkpoints, and large generated qualitative panels are intentionally excluded from this repository.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some encoders are gated or require external checkpoints/tokens. Follow the licensing and access terms for each foundation model.

## Data

Download or clone BRCA-M2C separately, then arrange it with the expected structure:

```text
Dataset-BRCA-M2C/
  images/
  labels/
  brca_ds_train.txt
  brca_ds_val.txt
  brca_ds_test.txt
```

The split files in `data/splits/` are included for reproducibility. Copy them into the dataset root if needed.

## Embedding Extraction

Example single-encoder extraction:

```bash
python src/run_brca_m2c_eval.py \
  --repo_root /path/to/Dataset-BRCA-M2C \
  --encoders provgigapath virchow2 uni ctranspath dinov2_vitl mae_vitl vit_large_patch16_224 resnet50 conch \
  --crop_size 64 \
  --max_cells_per_image 300 \
  --out_dir outputs_brca_mc2 \
  --device cuda \
  --use_cache
```

This creates cached embedding files such as:

```text
outputs_brca_mc2/embeddings/train_virchow2_crop64_cap300.npz
outputs_brca_mc2/embeddings/val_virchow2_crop64_cap300.npz
outputs_brca_mc2/embeddings/test_virchow2_crop64_cap300.npz
```

## Main Benchmark

```bash
python src/pfm_context_benchmark_full_brca_m2c.py \
  --emb_dir outputs_brca_mc2/embeddings \
  --encoders provgigapath virchow2 uni ctranspath dinov2_vitl mae_vitl vit_large_patch16_224 resnet50 conch \
  --crop_size 64 \
  --cap 300 \
  --out_dir outputs_brca_mc2_full_benchmark \
  --device cuda
```

The main output is:

```text
outputs_brca_mc2_full_benchmark/results_all_methods_and_ablations.csv
```

## Figures

```bash
python src/run_acmbcb_plot.py \
  --csv outputs_brca_mc2_full_benchmark/results_all_methods_and_ablations.csv \
  --out_dir paper_figures_brca_m2c
```

## Qualitative Panels

```bash
python src/run_acmbcb_qual.py \
  --repo_root /path/to/Dataset-BRCA-M2C \
  --emb_dir outputs_brca_mc2/embeddings \
  --encoder virchow2 \
  --crop_size 64 \
  --cap 300 \
  --out_dir qualitative_analysis \
  --device cuda
```

## Notes

The included CSV/JSON result artifacts are small enough for Git and are provided to document the exact benchmark outputs used for the paper plots. Large intermediate artifacts should be regenerated locally and are ignored by `.gitignore`.
