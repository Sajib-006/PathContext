#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate qualitative BRCA-M2C prediction panels for cell-only, patch-context, and graph-context models."
    )
    ap.add_argument("--repo_root", type=str, required=True, help="Path to the Dataset-BRCA-M2C root directory")
    ap.add_argument("--emb_dir", type=str, required=True, help="Directory containing cached train/val/test NPZ embeddings")
    ap.add_argument("--out_dir", type=str, default="qualitative_analysis", help="Output directory for summaries and panels")
    ap.add_argument("--encoder", type=str, default="virchow2")
    ap.add_argument("--crop_size", type=int, default=64)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--num_rescue", type=int, default=5, help="Number of strongest patch-context rescue patches to render")
    ap.add_argument("--num_failure", type=int, default=3, help="Number of strongest patch-context failure patches to render")
    return ap.parse_args()


ARGS = parse_args()

REPO_ROOT = Path(ARGS.repo_root).resolve()
EMB_DIR = Path(ARGS.emb_dir).resolve()
OUT_DIR = Path(ARGS.out_dir).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODER = ARGS.encoder
CROP_SIZE = ARGS.crop_size
CAP = ARGS.cap
DEVICE = ARGS.device or ("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {0: "lymphocyte", 1: "tumor_epi", 2: "stroma"}
# CLASS_COLORS = {
#     0: (59, 130, 246),   # blue
#     1: (239, 68, 68),    # red
#     2: (34, 197, 94),    # green
# }
CLASS_COLORS = {
    0: (49, 130, 189),   # blue
    1: (255, 193, 7),    # yellow/orange for tumor
    2: (0, 166, 81),     # green
}

# ----------------------------
# NPZ loading
# ----------------------------
def load_npz(npz_path: Path):
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
    M["patch_id"] = M["patch_id"].astype(str)
    return Z, Y, M

def build_npz_path(split: str):
    return EMB_DIR / f"{split}_{ENCODER}_crop{CROP_SIZE}_cap{CAP}.npz"

Ztr, Ytr, Mtr = load_npz(build_npz_path("train"))
Zva, Yva, Mva = load_npz(build_npz_path("val"))
Zte, Yte, Mte = load_npz(build_npz_path("test"))

# ----------------------------
# Helper features
# ----------------------------
def group_indices_by_patch(meta: pd.DataFrame):
    return {pid: g.index.to_numpy() for pid, g in meta.groupby("patch_id", sort=False)}

def compute_patch_mean(Z: np.ndarray, meta: pd.DataFrame):
    out = np.zeros_like(Z, dtype=np.float32)
    for _, idx in group_indices_by_patch(meta).items():
        mu = Z[idx].mean(axis=0, keepdims=True).astype(np.float32)
        out[idx] = mu
    return out

def neighbor_stats(Z_ref, M_ref, Z_query, M_query, k=8):
    d = Z_ref.shape[1]
    mean_out = np.zeros((len(M_query), d), dtype=np.float32)

    ref_groups = group_indices_by_patch(M_ref)
    query_groups = group_indices_by_patch(M_query)

    for pid, q_idx in query_groups.items():
        if pid not in ref_groups:
            continue
        r_idx = ref_groups[pid]
        ref_xy = M_ref.loc[r_idx, ["x", "y"]].to_numpy(dtype=np.float32)
        qry_xy = M_query.loc[q_idx, ["x", "y"]].to_numpy(dtype=np.float32)

        n_neighbors = min(len(r_idx), k + 1)
        if n_neighbors <= 1:
            mean_out[q_idx] = Z_ref[r_idx].mean(axis=0, keepdims=True)
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
            if same_frame:
                keep = ref_keys[neighbors_local] != qry_keys[j]
                neighbors_global = neighbors_global[keep]
            if len(neighbors_global) == 0:
                neighbors_global = r_idx[:1]
            neighbors_global = neighbors_global[:k]
            mean_out[qi] = Z_ref[neighbors_global].mean(axis=0).astype(np.float32)
    return mean_out

# ----------------------------
# Models
# ----------------------------
def train_mlp(Xtr, ytr, Xva, yva, Xte, device="cpu", hidden=256, epochs=25, batch_size=512):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)
    Xte_s = scaler.transform(Xte).astype(np.float32)

    tr_ds = TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr.astype(np.int64)))
    va_ds = TensorDataset(torch.from_numpy(Xva_s), torch.from_numpy(yva.astype(np.int64)))
    te_ds = TensorDataset(torch.from_numpy(Xte_s), torch.from_numpy(np.zeros(len(Xte_s), dtype=np.int64)))

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, 3),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best_state = None
    best_va = -1
    bad = 0
    patience = 5

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                pred = model(xb).argmax(dim=1).cpu().numpy()
                preds.append(pred)
                ys.append(yb.numpy())
        va_pred = np.concatenate(preds)
        va_true = np.concatenate(ys)
        va_acc = accuracy_score(va_true, va_pred)

        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in te_dl:
            xb = xb.to(device)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds)

# cell-only
pred_cell = train_mlp(Ztr, Ytr, Zva, Yva, Zte, device=DEVICE)

# patch-context
P_tr = compute_patch_mean(Ztr, Mtr)
P_va = compute_patch_mean(Zva, Mva)
P_te = compute_patch_mean(Zte, Mte)

Xtr_patch = np.concatenate([Ztr, P_tr, Ztr - P_tr], axis=1)
Xva_patch = np.concatenate([Zva, P_va, Zva - P_va], axis=1)
Xte_patch = np.concatenate([Zte, P_te, Zte - P_te], axis=1)
pred_patch = train_mlp(Xtr_patch, Ytr, Xva_patch, Yva, Xte_patch, device=DEVICE)

# graph-context from linear probe
base = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=3000, n_jobs=-1)),
])
base.fit(np.concatenate([Ztr, Zva], axis=0), np.concatenate([Ytr, Yva], axis=0))
proba = base.predict_proba(Zte).astype(np.float32)
pred_graph = proba.argmax(axis=1).copy()

groups = group_indices_by_patch(Mte)
for _, idx in groups.items():
    if len(idx) <= 2:
        continue
    xy = Mte.loc[idx, ["x", "y"]].to_numpy(dtype=np.float32)
    n_neighbors = min(9, len(idx))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(xy)
    _, ind = nn.kneighbors(xy)
    P = proba[idx].copy()
    for _ in range(3):
        P_new = P.copy()
        for i in range(len(idx)):
            nb = ind[i, 1:] if n_neighbors > 1 else ind[i]
            if len(nb) == 0:
                continue
            P_new[i] = 0.65 * P[i] + 0.35 * P[nb].mean(axis=0)
            P_new[i] /= max(P_new[i].sum(), 1e-8)
        P = P_new
    pred_graph[idx] = P.argmax(axis=1)

# ----------------------------
# Patch-level comparison
# ----------------------------
df = Mte.copy()
df["y_true"] = Yte
df["pred_cell"] = pred_cell
df["pred_patch"] = pred_patch
df["pred_graph"] = pred_graph

rows = []
for pid, g in df.groupby("patch_id", sort=False):
    y = g["y_true"].to_numpy()
    pc = g["pred_cell"].to_numpy()
    pp = g["pred_patch"].to_numpy()
    pg = g["pred_graph"].to_numpy()
    rows.append({
        "patch_id": pid,
        "n_cells": len(g),
        "macro_f1_cell": f1_score(y, pc, average="macro"),
        "macro_f1_patch": f1_score(y, pp, average="macro"),
        "macro_f1_graph": f1_score(y, pg, average="macro"),
        "delta_patch_minus_cell": f1_score(y, pp, average="macro") - f1_score(y, pc, average="macro"),
        "delta_graph_minus_cell": f1_score(y, pg, average="macro") - f1_score(y, pc, average="macro"),
    })

patch_perf = pd.DataFrame(rows).sort_values("delta_patch_minus_cell", ascending=False)
patch_perf.to_csv(OUT_DIR / f"{ENCODER}_patch_level_qualitative_summary.csv", index=False)

print("\nTop rescue patches (patch-context better than cell-only):")
print(patch_perf.head(10).to_string(index=False))

print("\nWorst patches (patch-context worse than cell-only):")
print(patch_perf.sort_values('delta_patch_minus_cell').head(10).to_string(index=False))

# ----------------------------
# Image helpers
# ----------------------------
def find_existing_image_path(repo_root: Path, rel_or_patch_id: str) -> Path:
    # try exact patch stem match in images
    cand = list((repo_root / "images").glob(rel_or_patch_id + ".*"))
    if len(cand) > 0:
        return cand[0]
    # fallback broader search
    for p in (repo_root / "images").glob("*.png"):
        if p.stem == rel_or_patch_id:
            return p
    raise FileNotFoundError(rel_or_patch_id)

def draw_overlay(img_path: Path, g: pd.DataFrame, label_col: str, out_path: Path, radius=7):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for _, r in g.iterrows():
        x = int(r["x"])
        y = int(r["y"])
        cls = int(r[label_col])
        fill = CLASS_COLORS[cls] + (190,)

        # outer black ring
        draw.ellipse(
            (x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1),
            fill=(0, 0, 0, 210)
        )

        # inner colored dot
        draw.ellipse(
            (x - radius + 1, y - radius + 1, x + radius - 1, y + radius - 1),
            fill=fill
        )

    img.save(out_path)

def make_panel_for_patch(patch_id: str):
    g = df[df["patch_id"] == patch_id].copy()
    img_path = find_existing_image_path(REPO_ROOT, patch_id)

    patch_dir = OUT_DIR / "patch_panels" / patch_id
    patch_dir.mkdir(parents=True, exist_ok=True)

    raw_copy = patch_dir / "raw.png"
    Image.open(img_path).convert("RGB").save(raw_copy)

    draw_overlay(img_path, g, "y_true", patch_dir / "gt.png")
    draw_overlay(img_path, g, "pred_cell", patch_dir / "pred_cell.png")
    draw_overlay(img_path, g, "pred_patch", patch_dir / "pred_patch.png")
    draw_overlay(img_path, g, "pred_graph", patch_dir / "pred_graph.png")

    # build 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    panel_files = [
        ("Ground truth", patch_dir / "gt.png"),
        ("Cell-only", patch_dir / "pred_cell.png"),
        ("Patch-context", patch_dir / "pred_patch.png"),
        ("Graph-context", patch_dir / "pred_graph.png"),
    ]

    for ax, (title, path) in zip(axes.ravel(), panel_files):
        ax.imshow(Image.open(path))
        ax.set_title(title, fontsize=24, fontweight="bold", pad=14)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.04, hspace=0.10)
    plt.savefig(patch_dir / "panel.png", dpi=260, bbox_inches="tight")
    plt.close()

# create panels for the strongest rescue and failure patches
selected = list(patch_perf.head(ARGS.num_rescue)["patch_id"]) + list(
    patch_perf.sort_values("delta_patch_minus_cell").head(ARGS.num_failure)["patch_id"]
)
selected = list(dict.fromkeys(selected))

for pid in selected:
    make_panel_for_patch(pid)

with open(OUT_DIR / f"{ENCODER}_selected_patches.json", "w") as f:
    json.dump(selected, f, indent=2)

print(f"\nSaved outputs to: {OUT_DIR}")
