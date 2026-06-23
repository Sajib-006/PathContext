# Artifact Scope

This release folder was assembled from the final ContextPatho ACM-BCB submission and the remote FairPath workspace.

## Included

- BRCA-M2C embedding extraction and baseline evaluation code.
- Full frozen-embedding context benchmark code.
- Paper plotting code and compact generated figures.
- Qualitative prediction-panel generation code.
- BRCA-M2C split files and per-patch dataset statistics.
- Small CSV/JSON benchmark outputs needed to trace the reported paper figures and tables.

## Intentionally Excluded

- Raw BRCA-M2C images and labels.
- Cached embedding `.npz` files.
- Large output folders such as full confusion-matrix image collections and qualitative patch panels.
- Camelyon17 and NDM/reliability scripts from the broader FairPath workspace.
- Exploratory dual-scale image-context code that was not part of the final paper's stated main method.

## Paper-to-Code Mapping

- Frozen embedding extraction: `src/run_brca_m2c_eval.py`
- Cell-only baselines: `src/pfm_context_benchmark_full_brca_m2c.py`
- Patch-context fusion: `run_patch_context_mlp` in `src/pfm_context_benchmark_full_brca_m2c.py`
- Graph-context propagation: `run_graph_label_propagation` in `src/pfm_context_benchmark_full_brca_m2c.py`
- Local-neighbor context: `run_local_context_mlp` in `src/pfm_context_benchmark_full_brca_m2c.py`
- Hybrid morphology-context features: `run_hybrid_morph_context` in `src/pfm_context_benchmark_full_brca_m2c.py`
- Multi-encoder context fusion: `run_multi_encoder_context_fusion` in `src/pfm_context_benchmark_full_brca_m2c.py`
- Sensitivity and ablations: ablation blocks in `src/pfm_context_benchmark_full_brca_m2c.py`
- Main paper plots: `src/run_acmbcb_plot.py`
- Qualitative maps: `src/run_acmbcb_qual.py`

## Not Fully Recovered As Standalone Scripts

The supplementary text mentions PanNuke external validation and PEFT/LoRA comparison. I did not find standalone `.py` scripts for those analyses in the remote FairPath folder during packaging; only notebook-level traces were visible. They are therefore not included as executable release scripts in this folder.
