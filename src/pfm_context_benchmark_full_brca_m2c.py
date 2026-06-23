#!/usr/bin/env python3
"""
Full benchmark + ablation runner for BRCA-M2C frozen pathology foundation model embeddings.

What this script covers
-----------------------
A. Encoder sweep (8-9 encoders)
B. Baselines
   1) linear_probe
   2) mlp_head
   3) residual_boosting
C. Context methods
   4) patch_context_mlp          <-- main proposed method
   5) graph_label_propagation
   6) hybrid_morph_context
   7) local_context_mlp          (kept for completeness / negative result)
   8) multiscale_proxy_mlp       (kept for completeness / negative result)
D. Multi-encoder fusion
   9) multi_encoder_context_fusion
E. Ablations
   - feature ablation: cell_only vs patch_only vs cell+patch vs cell+patch+residual
   - patch aggregation ablation: mean vs max
   - neighborhood-k ablation for local context / graph smoothing
   - patch cell subsampling ablation: use first N cells per patch when building patch context
   - cross-encoder summary + average improvements

Input assumptions
-----------------
The script reads cached single-scale embeddings in the same format as your previous extraction pipeline:
  train_<encoder>_crop64_cap300.npz
  val_<encoder>_crop64_cap300.npz
  test_<encoder>_crop64_cap300.npz
Each NPZ stores:
  Z:    [N, D]
  Y:    [N]
  meta: dict-of-lists with at least patch_id, x, y, class_id

Recommended encoder list
------------------------
provgigapath virchow2 uni ctranspath dinov2_vitl mae_vitl vit_large_patch16_224 resnet50 conch

Example full run
----------------
python pfm_context_benchmark_full_brca_m2c.py \
  --emb_dir ./outputs_brca_mc2/embeddings \
  --encoders provgigapath virchow2 uni ctranspath dinov2_vitl mae_vitl vit_large_patch16_224 resnet50 conch \
  --crop_size 64 \
  --cap 300 \
  --out_dir ./outputs_brca_mc2_full_benchmark \
  --device cuda

Smaller run for quick validation
--------------------------------
python pfm_context_benchmark_full_brca_m2c.py \
  --emb_dir ./outputs_brca_mc2/embeddings \
  --encoders provgigapath virchow2 dinov2_vitl \
  --crop_size 64 \
  --cap 300 \
  --out_dir ./outputs_brca_mc2_full_benchmark_small \
  --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

CLASS_MAP = {0: "lymphocyte", 1: "tumor_epi", 2: "stroma"}


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    d = np.load(npz_path, allow_pickle=True)
    Z = d["Z"]
    Y = d["Y"]
    meta = d["meta"].item()
    M = pd.DataFrame(meta)

    def _to_python_scalar(v):
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                return v.item()
            return v.detach().cpu().numpy().tolist()
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return v.item()
            return v.tolist()
        return v

    for col in M.columns:
        M[col] = M[col].map(_to_python_scalar)

    for col in ["x", "y", "class_id"]:
        if col in M.columns:
            M[col] = pd.to_numeric(M[col], errors="coerce")

    if "x" in M.columns and M["x"].isna().any():
        raise ValueError(f"Column x could not be converted cleanly in {npz_path}")
    if "y" in M.columns and M["y"].isna().any():
        raise ValueError(f"Column y could not be converted cleanly in {npz_path}")
    if "patch_id" not in M.columns:
        raise ValueError(f"NPZ missing required meta column patch_id: {npz_path}")

    M["patch_id"] = M["patch_id"].astype(str)
    return Z, Y, M


def build_npz_path(emb_dir: Path, encoder: str, split: str, crop_size: int, cap: int) -> Path:
    return emb_dir / f"{split}_{encoder}_crop{crop_size}_cap{cap}.npz"


def load_encoder_splits(emb_dir: Path, encoder: str, crop_size: int, cap: int):
    paths = {s: build_npz_path(emb_dir, encoder, s, crop_size, cap) for s in ["train", "val", "test"]}
    for s, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {s} NPZ for encoder={encoder}: {p}")
    tr = load_npz(paths["train"])
    va = load_npz(paths["val"])
    te = load_npz(paths["test"])
    return tr, va, te, paths


# -----------------------------------------------------------------------------
# Metrics / plotting
# -----------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def save_cm(cm: np.ndarray, out_png: Path, labels: List[str], title: str):
    plt.figure(figsize=(4.8, 3.9))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_barplot(df: pd.DataFrame, x: str, y: str, hue: Optional[str], title: str, out_png: Path, top_n: Optional[int] = None):
    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)
    plt.figure(figsize=(max(8, 0.8 * len(plot_df)), 5))
    if hue is None:
        xs = np.arange(len(plot_df))
        plt.bar(xs, plot_df[y].to_numpy())
        plt.xticks(xs, plot_df[x].astype(str).tolist(), rotation=45, ha="right")
    else:
        # simple grouped bar without seaborn
        xcats = plot_df[x].astype(str).unique().tolist()
        hcats = plot_df[hue].astype(str).unique().tolist()
        width = 0.8 / max(1, len(hcats))
        base = np.arange(len(xcats))
        for j, hc in enumerate(hcats):
            sub = plot_df[plot_df[hue].astype(str) == hc].set_index(x).reindex(xcats)
            vals = sub[y].to_numpy()
            plt.bar(base + j * width - 0.4 + width / 2, vals, width=width, label=hc)
        plt.xticks(base, xcats, rotation=45, ha="right")
        plt.legend()
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Core feature builders
# -----------------------------------------------------------------------------

def _group_indices_by_patch(meta: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {pid: g.index.to_numpy() for pid, g in meta.groupby("patch_id", sort=False)}


def _compute_patch_agg(Z: np.ndarray, meta: pd.DataFrame, agg: str = "mean", max_cells_per_patch: Optional[int] = None) -> np.ndarray:
    out = np.zeros_like(Z, dtype=np.float32)
    for _, idx in _group_indices_by_patch(meta).items():
        idx2 = idx
        if max_cells_per_patch is not None and len(idx2) > max_cells_per_patch:
            idx2 = idx2[:max_cells_per_patch]
        z = Z[idx2]
        if agg == "mean":
            pooled = z.mean(axis=0, keepdims=True)
        elif agg == "max":
            pooled = z.max(axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown agg={agg}")
        out[idx] = pooled.astype(np.float32)
    return out


def _compute_patch_counts(meta: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(meta), 1), dtype=np.float32)
    for _, idx in _group_indices_by_patch(meta).items():
        out[idx, 0] = float(len(idx))
    return out


def _compute_patch_bbox_norm(meta: pd.DataFrame) -> np.ndarray:
    feats = np.zeros((len(meta), 3), dtype=np.float32)
    for _, idx in _group_indices_by_patch(meta).items():
        xs = meta.loc[idx, "x"].to_numpy(dtype=np.float32)
        ys = meta.loc[idx, "y"].to_numpy(dtype=np.float32)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        xr = max(xmax - xmin, 1.0)
        yr = max(ymax - ymin, 1.0)
        xn = (xs - xmin) / xr
        yn = (ys - ymin) / yr
        r = np.sqrt((xn - 0.5) ** 2 + (yn - 0.5) ** 2)
        feats[idx, 0] = xn
        feats[idx, 1] = yn
        feats[idx, 2] = r
    return feats


def _neighbor_stats_for_split(
    Z_ref: np.ndarray,
    M_ref: pd.DataFrame,
    Z_query: np.ndarray,
    M_query: pd.DataFrame,
    k: int = 8,
) -> Dict[str, np.ndarray]:
    d = Z_ref.shape[1]
    mean_out = np.zeros((len(M_query), d), dtype=np.float32)
    std_out = np.zeros((len(M_query), d), dtype=np.float32)
    dist_out = np.zeros((len(M_query), 2), dtype=np.float32)

    ref_groups = _group_indices_by_patch(M_ref)
    query_groups = _group_indices_by_patch(M_query)

    for pid, q_idx in query_groups.items():
        if pid not in ref_groups:
            continue
        r_idx = ref_groups[pid]
        ref_xy = M_ref.loc[r_idx, ["x", "y"]].to_numpy(dtype=np.float32)
        qry_xy = M_query.loc[q_idx, ["x", "y"]].to_numpy(dtype=np.float32)

        n_neighbors = min(len(r_idx), k + 1)
        if n_neighbors <= 1:
            mean_out[q_idx] = Z_ref[r_idx].mean(axis=0, keepdims=True)
            std_out[q_idx] = 0.0
            dist_out[q_idx, 0] = 0.0
            dist_out[q_idx, 1] = 1.0
            continue

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(ref_xy)
        dist, ind = nn.kneighbors(qry_xy)

        same_frame = M_ref is M_query and Z_ref is Z_query
        if same_frame:
            ref_keys = M_ref.loc[r_idx, ["x", "y"]].astype(str).agg("_".join, axis=1).to_numpy()
            qry_keys = M_query.loc[q_idx, ["x", "y"]].astype(str).agg("_".join, axis=1).to_numpy()
        else:
            ref_keys, qry_keys = None, None

        for j, qi in enumerate(q_idx):
            neighbors_local = ind[j]
            neighbors_global = r_idx[neighbors_local]
            dists = dist[j]

            if same_frame:
                keep = ref_keys[neighbors_local] != qry_keys[j]
                neighbors_global = neighbors_global[keep]
                dists = dists[keep]

            if len(neighbors_global) == 0:
                neighbors_global = r_idx[:1]
                dists = np.array([0.0], dtype=np.float32)

            neighbors_global = neighbors_global[:k]
            dists = dists[:k]

            z_nb = Z_ref[neighbors_global]
            mean_out[qi] = z_nb.mean(axis=0).astype(np.float32)
            std_out[qi] = z_nb.std(axis=0).astype(np.float32)
            md = float(dists.mean()) if len(dists) else 0.0
            inv_density = 1.0 / max(md, 1e-3)
            dist_out[qi, 0] = md
            dist_out[qi, 1] = inv_density

    return {"neighbor_mean": mean_out, "neighbor_std": std_out, "neighbor_dist": dist_out}


def _compute_train_patch_label_priors(meta: pd.DataFrame, y: np.ndarray) -> Dict[str, np.ndarray]:
    df = meta.copy()
    df["y"] = y
    priors = {}
    for pid, g in df.groupby("patch_id", sort=False):
        p = np.bincount(g["y"].to_numpy(), minlength=3).astype(np.float32)
        p = p / max(p.sum(), 1.0)
        priors[pid] = p
    return priors


def _map_patch_priors(meta: pd.DataFrame, prior_map: Dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((len(meta), 3), dtype=np.float32)
    for i, pid in enumerate(meta["patch_id"].tolist()):
        out[i] = prior_map.get(pid, np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32))
    return out


# -----------------------------------------------------------------------------
# MLP blocks
# -----------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusionMLP(nn.Module):
    def __init__(self, dims: List[int], hidden: int = 256, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.dims = dims
        total = sum(dims)
        self.gate = nn.Sequential(nn.Linear(total, len(dims)), nn.Softmax(dim=1))
        self.proj = nn.ModuleList([nn.Linear(d, hidden) for d in dims])
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, parts: List[torch.Tensor]):
        x = torch.cat(parts, dim=1)
        g = self.gate(x)
        fused = 0.0
        for i, p in enumerate(parts):
            fused = fused + g[:, i:i + 1] * self.proj[i](p)
        return self.classifier(fused)


def _train_mlp_arrays(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    device: str,
    hidden: int = 256,
    batch_size: int = 512,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    tr_ds = TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr.astype(np.int64)))
    va_ds = TensorDataset(torch.from_numpy(Xva_s), torch.from_numpy(yva.astype(np.int64)))
    te_ds = TensorDataset(torch.from_numpy(Xte_s), torch.from_numpy(yte.astype(np.int64)))

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(in_dim=Xtr.shape[1], hidden=hidden, num_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_va = -1.0
    patience = 6
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits = model(xb)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                ys.append(yb.numpy())
        va_acc = accuracy_score(np.concatenate(ys), np.concatenate(preds))
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for xb, _ in te_dl:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            preds.append(prob.argmax(axis=1))
            probas.append(prob)
    return np.concatenate(preds), np.concatenate(probas), scaler


def _train_gated_parts(
    parts_tr: List[np.ndarray],
    ytr: np.ndarray,
    parts_va: List[np.ndarray],
    yva: np.ndarray,
    parts_te: List[np.ndarray],
    yte: np.ndarray,
    device: str,
    hidden: int = 256,
    batch_size: int = 512,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    scalers = []
    tr_s, va_s, te_s, dims = [], [], [], []
    for a_tr, a_va, a_te in zip(parts_tr, parts_va, parts_te):
        sc = StandardScaler()
        b_tr = sc.fit_transform(a_tr).astype(np.float32)
        b_va = sc.transform(a_va).astype(np.float32)
        b_te = sc.transform(a_te).astype(np.float32)
        scalers.append(sc)
        tr_s.append(b_tr)
        va_s.append(b_va)
        te_s.append(b_te)
        dims.append(b_tr.shape[1])

    tr_tensors = [torch.from_numpy(x) for x in tr_s]
    va_tensors = [torch.from_numpy(x) for x in va_s]
    te_tensors = [torch.from_numpy(x) for x in te_s]

    class PartsDataset(torch.utils.data.Dataset):
        def __init__(self, parts, y):
            self.parts = parts
            self.y = torch.from_numpy(y.astype(np.int64))
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return [p[idx] for p in self.parts], self.y[idx]

    def collate(batch):
        parts = list(zip(*[b[0] for b in batch]))
        parts = [torch.stack(p, dim=0) for p in parts]
        y = torch.stack([b[1] for b in batch], dim=0)
        return parts, y

    tr_dl = DataLoader(PartsDataset(tr_tensors, ytr), batch_size=batch_size, shuffle=True, collate_fn=collate)
    va_dl = DataLoader(PartsDataset(va_tensors, yva), batch_size=batch_size, shuffle=False, collate_fn=collate)
    te_dl = DataLoader(PartsDataset(te_tensors, yte), batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = GatedFusionMLP(dims=dims, hidden=hidden, num_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_va = -1.0
    patience = 6
    bad = 0

    for _ in range(epochs):
        model.train()
        for parts, yb in tr_dl:
            parts = [p.to(device) for p in parts]
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(parts)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for parts, yb in va_dl:
                parts = [p.to(device) for p in parts]
                logits = model(parts)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                ys.append(yb.numpy())
        va_acc = accuracy_score(np.concatenate(ys), np.concatenate(preds))
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for parts, _ in te_dl:
            parts = [p.to(device) for p in parts]
            logits = model(parts)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            preds.append(prob.argmax(axis=1))
            probas.append(prob)
    return np.concatenate(preds), np.concatenate(probas), scalers


# -----------------------------------------------------------------------------
# Baseline methods
# -----------------------------------------------------------------------------

def run_linear_probe(Ztr, Ytr, Zva, Yva, Zte, Yte):
    Xtr = np.concatenate([Ztr, Zva], axis=0)
    ytr = np.concatenate([Ytr, Yva], axis=0)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, n_jobs=-1)),
    ])
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Zte)
    pred = proba.argmax(axis=1)
    return pred, proba


def run_mlp_head(Ztr, Ytr, Zva, Yva, Zte, Yte, device: str):
    pred, proba, _ = _train_mlp_arrays(Ztr, Ytr, Zva, Yva, Zte, Yte, device=device, hidden=256)
    return pred, proba


def run_residual_boosting(Ztr, Ytr, Zva, Yva, Zte, Yte):
    Xtr = np.concatenate([Ztr, Zva], axis=0)
    ytr = np.concatenate([Ytr, Yva], axis=0)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Zte)
    base = LogisticRegression(max_iter=3000, n_jobs=-1)
    base.fit(Xtr_s, ytr)
    base_te = base.predict_proba(Xte_s)
    Xstack_tr = np.concatenate([Xtr_s, base.predict_proba(Xtr_s)], axis=1)
    Xstack_te = np.concatenate([Xte_s, base_te], axis=1)
    tree = HistGradientBoostingClassifier(loss="log_loss", max_depth=6, max_iter=200, learning_rate=0.06, random_state=42)
    tree.fit(Xstack_tr, ytr)
    proba = 0.4 * base_te + 0.6 * tree.predict_proba(Xstack_te)
    pred = proba.argmax(axis=1)
    return pred, proba


# -----------------------------------------------------------------------------
# Context methods
# -----------------------------------------------------------------------------

def run_local_context_mlp(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device: str, k: int = 8):
    tr_nb = _neighbor_stats_for_split(Ztr, Mtr, Ztr, Mtr, k=k)
    va_nb = _neighbor_stats_for_split(Ztr, Mtr, Zva, Mva, k=k)
    te_nb = _neighbor_stats_for_split(Ztr, Mtr, Zte, Mte, k=k)
    Xtr = np.concatenate([Ztr, tr_nb["neighbor_mean"], tr_nb["neighbor_std"], tr_nb["neighbor_dist"]], axis=1)
    Xva = np.concatenate([Zva, va_nb["neighbor_mean"], va_nb["neighbor_std"], va_nb["neighbor_dist"]], axis=1)
    Xte = np.concatenate([Zte, te_nb["neighbor_mean"], te_nb["neighbor_std"], te_nb["neighbor_dist"]], axis=1)
    pred, proba, _ = _train_mlp_arrays(Xtr, Ytr, Xva, Yva, Xte, Yte, device=device, hidden=256)
    return pred, proba


def run_patch_context_mlp(
    Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device: str,
    feature_mode: str = "cell_patch_resid", agg: str = "mean", max_cells_per_patch: Optional[int] = None,
):
    tr_patch = _compute_patch_agg(Ztr, Mtr, agg=agg, max_cells_per_patch=max_cells_per_patch)
    va_patch = _compute_patch_agg(Zva, Mva, agg=agg, max_cells_per_patch=max_cells_per_patch)
    te_patch = _compute_patch_agg(Zte, Mte, agg=agg, max_cells_per_patch=max_cells_per_patch)

    if feature_mode == "cell_only":
        Xtr, Xva, Xte = Ztr, Zva, Zte
    elif feature_mode == "patch_only":
        Xtr, Xva, Xte = tr_patch, va_patch, te_patch
    elif feature_mode == "cell_patch":
        Xtr = np.concatenate([Ztr, tr_patch], axis=1)
        Xva = np.concatenate([Zva, va_patch], axis=1)
        Xte = np.concatenate([Zte, te_patch], axis=1)
    elif feature_mode == "cell_patch_resid":
        Xtr = np.concatenate([Ztr, tr_patch, Ztr - tr_patch], axis=1)
        Xva = np.concatenate([Zva, va_patch, Zva - va_patch], axis=1)
        Xte = np.concatenate([Zte, te_patch, Zte - te_patch], axis=1)
    else:
        raise ValueError(f"Unknown feature_mode={feature_mode}")

    pred, proba, _ = _train_mlp_arrays(Xtr, Ytr, Xva, Yva, Xte, Yte, device=device, hidden=256)
    return pred, proba


def run_multiscale_proxy_mlp(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device: str, k: int = 8):
    tr_patch = _compute_patch_agg(Ztr, Mtr, agg="mean")
    va_patch = _compute_patch_agg(Zva, Mva, agg="mean")
    te_patch = _compute_patch_agg(Zte, Mte, agg="mean")
    tr_nb = _neighbor_stats_for_split(Ztr, Mtr, Ztr, Mtr, k=k)
    va_nb = _neighbor_stats_for_split(Ztr, Mtr, Zva, Mva, k=k)
    te_nb = _neighbor_stats_for_split(Ztr, Mtr, Zte, Mte, k=k)
    parts_tr = [Ztr, tr_nb["neighbor_mean"], tr_patch, tr_nb["neighbor_dist"]]
    parts_va = [Zva, va_nb["neighbor_mean"], va_patch, va_nb["neighbor_dist"]]
    parts_te = [Zte, te_nb["neighbor_mean"], te_patch, te_nb["neighbor_dist"]]
    pred, proba, _ = _train_gated_parts(parts_tr, Ytr, parts_va, Yva, parts_te, Yte, device=device, hidden=256)
    return pred, proba


def run_graph_label_propagation(Ztr, Ytr, Mtr, Zte, Yte, Mte, k: int = 8, alpha: float = 0.35, steps: int = 3):
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, n_jobs=-1)),
    ])
    base.fit(Ztr, Ytr)
    proba = base.predict_proba(Zte).astype(np.float32)
    new_proba = proba.copy()

    for _, idx in _group_indices_by_patch(Mte).items():
        if len(idx) <= 2:
            continue
        xy = Mte.loc[idx, ["x", "y"]].to_numpy(dtype=np.float32)
        n_neighbors = min(k + 1, len(idx))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(xy)
        _, ind = nn.kneighbors(xy)
        P = proba[idx].copy()
        for _ in range(steps):
            P_new = P.copy()
            for i in range(len(idx)):
                nb = ind[i, 1:] if n_neighbors > 1 else ind[i]
                if len(nb) == 0:
                    continue
                P_new[i] = (1 - alpha) * P[i] + alpha * P[nb].mean(axis=0)
                P_new[i] /= max(P_new[i].sum(), 1e-8)
            P = P_new
        new_proba[idx] = P
    pred = new_proba.argmax(axis=1)
    return pred, new_proba


def run_hybrid_morph_context(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device: str):
    tr_patch = _compute_patch_agg(Ztr, Mtr, agg="mean")
    va_patch = _compute_patch_agg(Zva, Mva, agg="mean")
    te_patch = _compute_patch_agg(Zte, Mte, agg="mean")
    tr_nb = _neighbor_stats_for_split(Ztr, Mtr, Ztr, Mtr, k=8)
    va_nb = _neighbor_stats_for_split(Ztr, Mtr, Zva, Mva, k=8)
    te_nb = _neighbor_stats_for_split(Ztr, Mtr, Zte, Mte, k=8)
    tr_pos = _compute_patch_bbox_norm(Mtr)
    va_pos = _compute_patch_bbox_norm(Mva)
    te_pos = _compute_patch_bbox_norm(Mte)
    tr_cnt = _compute_patch_counts(Mtr)
    va_cnt = _compute_patch_counts(Mva)
    te_cnt = _compute_patch_counts(Mte)
    train_priors = _compute_train_patch_label_priors(pd.concat([Mtr, Mva], axis=0, ignore_index=True), np.concatenate([Ytr, Yva]))
    tr_prior = _map_patch_priors(Mtr, train_priors)
    va_prior = _map_patch_priors(Mva, train_priors)
    te_prior = _map_patch_priors(Mte, train_priors)

    Xtr = np.concatenate([Ztr, tr_patch, tr_nb["neighbor_dist"], tr_pos, tr_cnt, tr_prior], axis=1)
    Xva = np.concatenate([Zva, va_patch, va_nb["neighbor_dist"], va_pos, va_cnt, va_prior], axis=1)
    Xte = np.concatenate([Zte, te_patch, te_nb["neighbor_dist"], te_pos, te_cnt, te_prior], axis=1)
    pred, proba, _ = _train_mlp_arrays(Xtr, Ytr, Xva, Yva, Xte, Yte, device=device, hidden=256)
    return pred, proba


def run_multi_encoder_context_fusion(train_dict, val_dict, test_dict, device: str):
    encs = list(train_dict.keys())
    ref = encs[0]

    def _split_keys(pack):
        _, y, m = pack
        return m[["patch_id", "x", "y"]].astype(str).agg("|".join, axis=1) + "|" + pd.Series(y).astype(str)

    common_train = set(_split_keys(train_dict[ref]).tolist())
    common_val = set(_split_keys(val_dict[ref]).tolist())
    common_test = set(_split_keys(test_dict[ref]).tolist())
    for enc in encs[1:]:
        common_train &= set(_split_keys(train_dict[enc]).tolist())
        common_val &= set(_split_keys(val_dict[enc]).tolist())
        common_test &= set(_split_keys(test_dict[enc]).tolist())

    if min(len(common_train), len(common_val), len(common_test)) == 0:
        raise ValueError("No common aligned samples across all encoders. Re-extract embeddings with fixed sampling for exact fusion.")

    def _ordered_common(pack, common_set):
        Z, y, m = pack
        keys = m[["patch_id", "x", "y"]].astype(str).agg("|".join, axis=1) + "|" + pd.Series(y).astype(str)
        keep = keys.isin(common_set)
        Z = Z[keep.to_numpy()]
        y = y[keep.to_numpy()]
        m = m.loc[keep].reset_index(drop=True)
        keys = keys.loc[keep].reset_index(drop=True)
        order = np.argsort(keys.to_numpy())
        return Z[order], y[order], m.iloc[order].reset_index(drop=True), keys.iloc[order].to_numpy()

    tr_parts, va_parts, te_parts = [], [], []
    Ytr = Yva = Yte = None
    common_sizes = {}

    for enc in encs:
        Ztr, ytr, Mtr, ktr = _ordered_common(train_dict[enc], common_train)
        Zva, yva, Mva, kva = _ordered_common(val_dict[enc], common_val)
        Zte, yte, Mte, kte = _ordered_common(test_dict[enc], common_test)
        if Ytr is None:
            Ytr, Yva, Yte = ytr, yva, yte
            ref_ktr, ref_kva, ref_kte = ktr, kva, kte
        else:
            if not (np.array_equal(ktr, ref_ktr) and np.array_equal(kva, ref_kva) and np.array_equal(kte, ref_kte)):
                raise ValueError("Common-key ordering mismatch across encoders during fusion.")
        tr_patch = _compute_patch_agg(Ztr, Mtr, agg="mean")
        va_patch = _compute_patch_agg(Zva, Mva, agg="mean")
        te_patch = _compute_patch_agg(Zte, Mte, agg="mean")
        tr_parts.extend([Ztr, tr_patch])
        va_parts.extend([Zva, va_patch])
        te_parts.extend([Zte, te_patch])
        common_sizes[enc] = {"train": len(Ztr), "val": len(Zva), "test": len(Zte)}

    pred, proba, _ = _train_gated_parts(tr_parts, Ytr, va_parts, Yva, te_parts, Yte, device=device, hidden=384)
    return pred, proba, Yte, common_sizes


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def evaluate_method(name: str, y_true: np.ndarray, pred: np.ndarray, out_dir: Path):
    metrics = compute_metrics(y_true, pred)
    cm = confusion_matrix(y_true, pred)
    save_cm(cm, out_dir / f"cm_{name}.png", [CLASS_MAP[i] for i in sorted(CLASS_MAP)], name)
    report = classification_report(y_true, pred, target_names=[CLASS_MAP[i] for i in sorted(CLASS_MAP)], output_dict=True)
    return metrics, cm, report


def add_result(summary_rows, full_reports, encoder, name, y_true, pred, out_dir, runtime_s, extra=None):
    metrics, cm, report = evaluate_method(f"{encoder}_{name}", y_true, pred, out_dir)
    row = {"encoder": encoder, "method": name, "runtime_s": runtime_s, **metrics}
    if extra:
        row.update(extra)
    summary_rows.append(row)
    if encoder not in full_reports:
        full_reports[encoder] = {}
    full_reports[encoder][name] = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "report": report,
        **(extra or {}),
    }
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", type=str, required=True)
    ap.add_argument("--encoders", type=str, nargs="+", required=True)
    ap.add_argument("--crop_size", type=int, default=64)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--out_dir", type=str, default="outputs_brca_mc2_full_benchmark")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_negative_controls", action="store_true", help="Skip local_context_mlp and multiscale_proxy_mlp")
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Embedding dir: {emb_dir}")

    train_dict, val_dict, test_dict = {}, {}, {}
    loaded_encoders = []
    for enc in args.encoders:
        try:
            tr, va, te, _ = load_encoder_splits(emb_dir, enc, args.crop_size, args.cap)
            train_dict[enc] = tr
            val_dict[enc] = va
            test_dict[enc] = te
            loaded_encoders.append(enc)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")

    if len(loaded_encoders) == 0:
        raise RuntimeError("No encoder NPZs were found.")

    summary_rows = []
    full_reports = {}

    # ------------------------------------------------------------------
    # Per-encoder full benchmark
    # ------------------------------------------------------------------
    for enc in loaded_encoders:
        print("\n" + "=" * 88)
        print(f"Running methods for encoder: {enc}")
        print("=" * 88)

        Ztr, Ytr, Mtr = train_dict[enc]
        Zva, Yva, Mva = val_dict[enc]
        Zte, Yte, Mte = test_dict[enc]
        enc_dir = out_dir / enc
        enc_dir.mkdir(parents=True, exist_ok=True)

        # Baselines
        for name, fn in [
            ("linear_probe", lambda: run_linear_probe(Ztr, Ytr, Zva, Yva, Zte, Yte)),
            ("mlp_head", lambda: run_mlp_head(Ztr, Ytr, Zva, Yva, Zte, Yte, device=device)),
            ("residual_boosting", lambda: run_residual_boosting(Ztr, Ytr, Zva, Yva, Zte, Yte)),
            ("patch_context_mlp", lambda: run_patch_context_mlp(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device=device, feature_mode="cell_patch_resid", agg="mean")),
            ("graph_label_propagation", lambda: run_graph_label_propagation(np.concatenate([Ztr, Zva], axis=0), np.concatenate([Ytr, Yva], axis=0), pd.concat([Mtr, Mva], axis=0, ignore_index=True), Zte, Yte, Mte, k=8, alpha=0.35, steps=3)),
            ("hybrid_morph_context", lambda: run_hybrid_morph_context(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device=device)),
        ]:
            t0 = time.perf_counter()
            pred, proba = fn()
            dt = time.perf_counter() - t0
            metrics = add_result(summary_rows, full_reports, enc, name, Yte, pred, enc_dir, dt)
            print(f"{enc:>20s} | {name:>26s} | acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_acc']:.4f} time={dt:.2f}s")

        if not args.skip_negative_controls:
            for name, fn in [
                ("local_context_mlp", lambda: run_local_context_mlp(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device=device, k=8)),
                ("multiscale_proxy_mlp", lambda: run_multiscale_proxy_mlp(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device=device, k=8)),
            ]:
                t0 = time.perf_counter()
                pred, proba = fn()
                dt = time.perf_counter() - t0
                metrics = add_result(summary_rows, full_reports, enc, name, Yte, pred, enc_dir, dt)
                print(f"{enc:>20s} | {name:>26s} | acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_acc']:.4f} time={dt:.2f}s")

        # ------------------------------------------------------------------
        # Ablation 1: feature contribution for patch context
        # ------------------------------------------------------------------
        print(f"{'':>20s}   [Ablation] feature contribution")
        for feature_mode in ["cell_only", "patch_only", "cell_patch", "cell_patch_resid"]:
            name = f"ablate_patchfeat_{feature_mode}"
            t0 = time.perf_counter()
            pred, proba = run_patch_context_mlp(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device=device, feature_mode=feature_mode, agg="mean")
            dt = time.perf_counter() - t0
            metrics = add_result(summary_rows, full_reports, enc, name, Yte, pred, enc_dir, dt, extra={"ablation_group": "feature_mode", "feature_mode": feature_mode})
            print(f"{enc:>20s} | {name:>26s} | acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_acc']:.4f}")

        # ------------------------------------------------------------------
        # Ablation 2: patch aggregation type
        # ------------------------------------------------------------------
        print(f"{'':>20s}   [Ablation] patch aggregation")
        for agg in ["mean", "max"]:
            name = f"ablate_patchagg_{agg}"
            t0 = time.perf_counter()
            pred, proba = run_patch_context_mlp(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device=device, feature_mode="cell_patch_resid", agg=agg)
            dt = time.perf_counter() - t0
            metrics = add_result(summary_rows, full_reports, enc, name, Yte, pred, enc_dir, dt, extra={"ablation_group": "patch_agg", "patch_agg": agg})
            print(f"{enc:>20s} | {name:>26s} | acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_acc']:.4f}")

        # ------------------------------------------------------------------
        # Ablation 3: patch cell budget
        # ------------------------------------------------------------------
        print(f"{'':>20s}   [Ablation] patch cell budget")
        for budget in [100, 200, 300]:
            name = f"ablate_patchbudget_{budget}"
            t0 = time.perf_counter()
            pred, proba = run_patch_context_mlp(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device=device, feature_mode="cell_patch_resid", agg="mean", max_cells_per_patch=budget)
            dt = time.perf_counter() - t0
            metrics = add_result(summary_rows, full_reports, enc, name, Yte, pred, enc_dir, dt, extra={"ablation_group": "patch_budget", "patch_budget": budget})
            print(f"{enc:>20s} | {name:>26s} | acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_acc']:.4f}")

        # ------------------------------------------------------------------
        # Ablation 4: neighborhood size
        # ------------------------------------------------------------------
        print(f"{'':>20s}   [Ablation] neighborhood size")
        for k in [4, 8, 12]:
            # graph smoothing
            name = f"ablate_graph_k{k}"
            t0 = time.perf_counter()
            pred, proba = run_graph_label_propagation(np.concatenate([Ztr, Zva], axis=0), np.concatenate([Ytr, Yva], axis=0), pd.concat([Mtr, Mva], axis=0, ignore_index=True), Zte, Yte, Mte, k=k, alpha=0.35, steps=3)
            dt = time.perf_counter() - t0
            metrics = add_result(summary_rows, full_reports, enc, name, Yte, pred, enc_dir, dt, extra={"ablation_group": "graph_k", "k": k})
            print(f"{enc:>20s} | {name:>26s} | acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_acc']:.4f}")

            if not args.skip_negative_controls:
                name2 = f"ablate_localctx_k{k}"
                t0 = time.perf_counter()
                pred, proba = run_local_context_mlp(Ztr, Ytr, Mtr, Zva, Yva, Mva, Zte, Yte, Mte, device=device, k=k)
                dt = time.perf_counter() - t0
                metrics = add_result(summary_rows, full_reports, enc, name2, Yte, pred, enc_dir, dt, extra={"ablation_group": "localctx_k", "k": k})
                print(f"{enc:>20s} | {name2:>26s} | acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_acc']:.4f}")

    # ------------------------------------------------------------------
    # Multi-encoder fusion
    # ------------------------------------------------------------------
    if len(loaded_encoders) >= 2:
        print("\n" + "=" * 88)
        print("Running multi-encoder context fusion")
        print("=" * 88)
        t0 = time.perf_counter()
        pred, proba, Yte_fusion, common_sizes = run_multi_encoder_context_fusion(train_dict, val_dict, test_dict, device=device)
        dt = time.perf_counter() - t0
        fusion_dir = out_dir / "fusion"
        fusion_dir.mkdir(parents=True, exist_ok=True)
        metrics = add_result(summary_rows, full_reports, "fusion", "multi_encoder_context_fusion", Yte_fusion, pred, fusion_dir, dt, extra={"encoders": loaded_encoders, "common_sizes": common_sizes})
        print(f"{'fusion':>20s} | {'multi_encoder_context_fusion':>26s} | acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_acc']:.4f} time={dt:.2f}s")

    # ------------------------------------------------------------------
    # Save summaries
    # ------------------------------------------------------------------
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["balanced_acc", "macro_f1", "acc"], ascending=False).reset_index(drop=True)
    summary.to_csv(out_dir / "results_all_methods_and_ablations.csv", index=False)
    (out_dir / "reports_all_methods_and_ablations.json").write_text(json.dumps(full_reports, indent=2))

    # main method table only
    main_methods = {
        "linear_probe", "mlp_head", "residual_boosting", "patch_context_mlp", "graph_label_propagation", "hybrid_morph_context",
        "local_context_mlp", "multiscale_proxy_mlp", "multi_encoder_context_fusion"
    }
    main_df = summary[summary["method"].isin(main_methods)].copy()
    main_df.to_csv(out_dir / "results_main_methods.csv", index=False)

    # encoder x method pivot
    pivot_acc = main_df.pivot_table(index="encoder", columns="method", values="acc")
    pivot_f1 = main_df.pivot_table(index="encoder", columns="method", values="macro_f1")
    pivot_bal = main_df.pivot_table(index="encoder", columns="method", values="balanced_acc")
    pivot_acc.to_csv(out_dir / "pivot_acc.csv")
    pivot_f1.to_csv(out_dir / "pivot_macro_f1.csv")
    pivot_bal.to_csv(out_dir / "pivot_balanced_acc.csv")

    # average across encoders for main methods (exclude fusion)
    avg_main = main_df[main_df["encoder"] != "fusion"].groupby("method")[["acc", "macro_f1", "balanced_acc", "runtime_s"]].mean().reset_index()
    avg_main = avg_main.sort_values(["balanced_acc", "macro_f1", "acc"], ascending=False)
    avg_main.to_csv(out_dir / "average_main_methods_across_encoders.csv", index=False)

    # improvement over mlp_head and linear_probe per encoder
    improvements = []
    for enc in [e for e in loaded_encoders if e in main_df["encoder"].unique()]:
        sub = main_df[main_df["encoder"] == enc].set_index("method")
        if "patch_context_mlp" in sub.index:
            row = {"encoder": enc}
            if "mlp_head" in sub.index:
                row["delta_acc_vs_mlp"] = float(sub.loc["patch_context_mlp", "acc"] - sub.loc["mlp_head", "acc"])
                row["delta_bal_vs_mlp"] = float(sub.loc["patch_context_mlp", "balanced_acc"] - sub.loc["mlp_head", "balanced_acc"])
            if "linear_probe" in sub.index:
                row["delta_acc_vs_linear"] = float(sub.loc["patch_context_mlp", "acc"] - sub.loc["linear_probe", "acc"])
                row["delta_bal_vs_linear"] = float(sub.loc["patch_context_mlp", "balanced_acc"] - sub.loc["linear_probe", "balanced_acc"])
            improvements.append(row)
    if improvements:
        imp_df = pd.DataFrame(improvements)
        imp_df.to_csv(out_dir / "patch_context_improvements.csv", index=False)

    # ablation tables
    for group_name, fname in [
        ("feature_mode", "ablation_feature_mode.csv"),
        ("patch_agg", "ablation_patch_agg.csv"),
        ("patch_budget", "ablation_patch_budget.csv"),
        ("graph_k", "ablation_graph_k.csv"),
        ("localctx_k", "ablation_localctx_k.csv"),
    ]:
        sub = summary[summary.get("ablation_group", pd.Series([None] * len(summary))) == group_name].copy()
        if len(sub):
            sub.to_csv(out_dir / fname, index=False)

    # simple plots
    try:
        save_barplot(avg_main, x="method", y="balanced_acc", hue=None, title="Average balanced accuracy across encoders", out_png=out_dir / "avg_balanced_acc_main_methods.png")
        top_plot_df = main_df.sort_values(["balanced_acc", "macro_f1", "acc"], ascending=False).head(20)
        save_barplot(top_plot_df, x="encoder", y="balanced_acc", hue="method", title="Top 20 method-encoder pairs by balanced accuracy", out_png=out_dir / "top20_balanced_acc.png")
    except Exception as e:
        print(f"[WARN] plotting failed: {e}")

    print("\nSaved:")
    print(out_dir / "results_all_methods_and_ablations.csv")
    print(out_dir / "results_main_methods.csv")
    print(out_dir / "average_main_methods_across_encoders.csv")
    print(out_dir / "reports_all_methods_and_ablations.json")
    print("\nTop results:")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
