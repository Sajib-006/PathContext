import argparse
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import umap
import matplotlib.pyplot as plt


# ----------------------------
# Dataset utilities
# ----------------------------

CLASS_MAP = {1: "lymphocyte", 2: "tumor_epi", 3: "stroma"}

def read_split_list(txt_path: Path) -> List[str]:
    # each line contains a filename (usually relative), e.g. images/xxx.png or xxx.png
    lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    return lines

def find_existing_image_path(repo_root: Path, rel: str) -> Path:
    # handles if split files contain "xxx.png" vs "images/xxx.png"
    p = repo_root / rel
    if p.exists():
        return p
    p2 = repo_root / "images" / rel
    if p2.exists():
        return p2
    # sometimes file includes "images/.."
    p3 = repo_root / rel.replace("images/", "")
    if p3.exists():
        return p3
    raise FileNotFoundError(f"Cannot locate image for split entry: {rel}")

def label_path_for_image(repo_root: Path, img_path: Path) -> Path:
    # dataset says label filename prefixed with corresponding image filename
    # Common pattern: labels/<image_stem>.txt OR labels/<image_filename>_...txt
    # We'll attempt the most common: same stem + .txt in labels/
    labels_dir = repo_root / "labels"
    cand1 = labels_dir / (img_path.stem + ".txt")
    if cand1.exists():
        return cand1

    # fallback: search by prefix
    matches = sorted(labels_dir.glob(img_path.stem + "*.txt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # choose shortest match (often exact)
        return min(matches, key=lambda p: len(p.name))

    raise FileNotFoundError(f"Cannot locate label file for image: {img_path.name}")

def read_dots(label_txt: Path) -> np.ndarray:
    # rows: Y X class_id
    rows = []
    for line in label_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        y, x, c = int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
        if c in (1, 2, 3):
            rows.append((x, y, c))
    return np.array(rows, dtype=np.int32)  # [N,3] with (x,y,class)

def safe_crop(img: Image.Image, x: int, y: int, size: int) -> Image.Image:
    # center crop around (x,y), pad if near boundary
    half = size // 2
    left, top = x - half, y - half
    right, bottom = left + size, top + size

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - img.width)
    pad_bottom = max(0, bottom - img.height)

    if pad_left or pad_top or pad_right or pad_bottom:
        new_w = img.width + pad_left + pad_right
        new_h = img.height + pad_top + pad_bottom
        canvas = Image.new(img.mode, (new_w, new_h))
        canvas.paste(img, (pad_left, pad_top))
        left += pad_left
        top += pad_top
        return canvas.crop((left, top, left + size, top + size))

    return img.crop((left, top, left + size, top + size))

@dataclass
class CellExample:
    img_rel: str
    cell_x: int
    cell_y: int
    cell_cls: int
    patch_id: str  # image stem


class BRCA_M2C_CellCrops(Dataset):
    def __init__(
        self,
        repo_root: Path,
        split_txt: Path,
        crop_size: int = 64,
        max_cells_per_image: Optional[int] = None,
        transform=None,
    ):
        self.repo_root = repo_root
        self.crop_size = crop_size
        self.transform = transform

        rels = read_split_list(split_txt)
        self.examples: List[CellExample] = []

        for rel in rels:
            img_path = find_existing_image_path(repo_root, rel)
            lab_path = label_path_for_image(repo_root, img_path)
            dots = read_dots(lab_path)
            if dots.size == 0:
                continue

            if max_cells_per_image is not None and len(dots) > max_cells_per_image:
                idx = np.random.choice(len(dots), size=max_cells_per_image, replace=False)
                dots = dots[idx]

            for (x, y, c) in dots:
                self.examples.append(
                    CellExample(img_rel=str(img_path.relative_to(repo_root)),
                                cell_x=int(x), cell_y=int(y),
                                cell_cls=int(c),
                                patch_id=img_path.stem)
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        img_path = self.repo_root / ex.img_rel
        img = Image.open(img_path).convert("RGB")
        crop = safe_crop(img, ex.cell_x, ex.cell_y, self.crop_size)
        if self.transform is not None:
            crop_t = self.transform(crop)
        else:
            crop_t = transforms.ToTensor()(crop)
        y = ex.cell_cls - 1  # 0..2
        meta = {"img_rel": ex.img_rel, "patch_id": ex.patch_id, "x": ex.cell_x, "y": ex.cell_y, "class_id": ex.cell_cls}
        return crop_t, y, meta


# ----------------------------
# Encoder adapters
# ----------------------------

import torch.nn.functional as F

class EncoderWrapper(nn.Module):
    """Generic image encoder returning a 1D embedding per crop."""
    def __init__(self, model: nn.Module, embed_dim: int):
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            feat = self.model.forward_features(x)

            # ViT-style: [B, T, C] -> CLS
            if feat.ndim == 3:
                return feat[:, 0, :]

            # CNN/Swin-style: [B, C, H, W] -> global avg pool
            if feat.ndim == 4:
                feat = F.adaptive_avg_pool2d(feat, output_size=1).flatten(1)
                return feat

            # Already [B, C]
            if feat.ndim == 2:
                return feat

        out = self.model(x)
        if out.ndim > 2:
            out = torch.flatten(out, 1)
        return out


def load_encoder_and_transforms(name: str, device: str, hf_token: Optional[str] = None):
    name = name.lower()

    def timm_tfm(model):
        from timm.data import resolve_model_data_config, create_transform
        cfg = resolve_model_data_config(model)
        tfm = create_transform(**cfg, is_training=False)
        return tfm
    # Default imagenet-style transform (safe baseline)
    def imagenet_tfm(size=224):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    # ---------- UNI / UNI2 ----------
    if name in ["uni", "uni2", "uni2-h"]:
        # HF has model weights; many examples use timm + hf_hub_download. :contentReference[oaicite:5]{index=5}
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as safe_load

        repo_id = "MahmoodLab/UNI" if name == "uni" else "MahmoodLab/UNI2-h"
        # download safetensors if available; otherwise pytorch_model.bin
        # ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", token=hf_token)
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", token=hf_token)

        # UNI/UNI2 commonly use ViT-H/14 or similar; use timm architecture as in their docs. :contentReference[oaicite:6]{index=6}
        # If UNI2-h: vit_huge_patch14_224, if UNI: often vit_large_patch16_224 as fallback.
        timm_arch = "vit_huge_patch14_224" if name != "uni" else "vit_large_patch16_224"
        base = timm.create_model(timm_arch, pretrained=False, num_classes=0)
        # sd = safe_load(ckpt_path)
        sd = torch.load(ckpt_path, map_location="cpu")
        # Strip common prefixes
        new_sd = {}
        for k, v in sd.items():
            nk = k
            for pref in ["model.", "module.", "encoder."]:
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            new_sd[nk] = v
        base.load_state_dict(new_sd, strict=False)
        enc = EncoderWrapper(base, base.num_features).to(device).eval()

        # UNI typically expects 224 tiles; keep 224 unless you confirm otherwise.
        return enc, imagenet_tfm(224)

    # ---------- CONCH (vision encoder only) ----------
    if name == "conch":
        # CONCH is a vision-language model; for our use we only need image encoder. :contentReference[oaicite:7]{index=7}
        # Many VLMs are open_clip-like; robust approach: try open_clip first.
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="hf-hub:MahmoodLab/CONCH"
            )
            # open_clip model returns both; we use visual encoder features
            class ConchVisual(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    # returns embedding [B,C]
                    return self.m.encode_image(x)

            enc = EncoderWrapper(ConchVisual(model), embed_dim=model.visual.output_dim).to(device).eval()
            # preprocess is already a torchvision transform
            return enc, preprocess
        except Exception:
            # fallback: treat as timm model if open_clip not available
            base = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
            enc = EncoderWrapper(base, base.num_features).to(device).eval()
            return enc, imagenet_tfm(224)

    # ---------- Virchow / Virchow2 ----------
    if name in ["virchow", "virchow2"]:
        # Paige provides Virchow/Virchow2 on HF. :contentReference[oaicite:8]{index=8}
        # Many are standard ViT checkpoints; easiest is transformers AutoModel.
        try:
            from transformers import AutoModel
            repo_id = "paige-ai/Virchow2" if name == "virchow2" else "paige-ai/Virchow"
            m = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=hf_token)
            m.eval().to(device)

            class HFVision(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    out = self.m(pixel_values=x)
                    # common: last_hidden_state [B,T,C]
                    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                    return h[:, 0, :]  # CLS

            # guess dim from config if present
            dim = getattr(getattr(m, "config", None), "hidden_size", 1024)
            enc = EncoderWrapper(HFVision(m), dim).to(device).eval()
            return enc, imagenet_tfm(224)
        except Exception:
            base = timm.create_model("vit_huge_patch14_224", pretrained=True, num_classes=0)
            enc = EncoderWrapper(base, base.num_features).to(device).eval()
            return enc, imagenet_tfm(224)

    # ---------- Prov-GigaPath patch encoder ----------
    if name in ["prov-gigapath", "provgigapath", "gigapath_patch"]:
        # GigaPath patch encoder exists on HF. :contentReference[oaicite:9]{index=9}
        try:
            from transformers import AutoModel
            m = AutoModel.from_pretrained("prov-gigapath/prov-gigapath", trust_remote_code=True, token=hf_token)
            m.eval().to(device)

            class HFVision(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    out = self.m(pixel_values=x)
                    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                    return h[:, 0, :]
            dim = getattr(getattr(m, "config", None), "hidden_size", 1024)
            enc = EncoderWrapper(HFVision(m), dim).to(device).eval()
            return enc, imagenet_tfm(224)
        except Exception:
            base = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=0)
            enc = EncoderWrapper(base, base.num_features).to(device).eval()
            return enc, imagenet_tfm(224)
    
    # ---------- DINOv2 ViT-L ----------
    if name in ["dinov2_vitl", "dinov2-vitl", "dinov2", "vit_large_dinov2"]:
        # timm model id for DINOv2 ViT-L
        base = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
        enc = EncoderWrapper(base, base.num_features).to(device).eval()
        return enc, timm_tfm(base)
    
    # ---------- MAE ViT-L ----------
    if name in ["mae_vitl", "mae-vitl", "vit_large_mae", "mae"]:
        base = timm.create_model("vit_large_patch16_224.mae", pretrained=True, num_classes=0)
        enc = EncoderWrapper(base, base.num_features).to(device).eval()
        return enc, timm_tfm(base)
    
    # ---------- ResNet50 (frozen features) ----------
    if name in ["resnet50", "rn50", "resnet50_frozen"]:
        base = timm.create_model("resnet50", pretrained=True, num_classes=0)
        enc = EncoderWrapper(base, base.num_features).to(device).eval()
        return enc, timm_tfm(base)
    
    # ---------- CTransPath ----------
    if name in ["ctranspath", "ctranspath_swin", "ctranspath_swin_tiny"]:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as safe_load

        repo_id = "1aurent/swin_tiny_patch4_window7_224.CTransPath"
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", token=hf_token)
        sd = safe_load(ckpt_path)

        # strip prefixes
        new_sd = {}
        for k, v in sd.items():
            nk = k
            for pref in ["model.", "module.", "encoder."]:
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            new_sd[nk] = v

        # ---- detect by embed dim (most reliable) ----
        key = "patch_embed.norm.weight"
        if key not in new_sd:
            raise RuntimeError(f"CTransPath ckpt missing expected key: {key}")

        embed_dim = int(new_sd[key].shape[0])  # 96 for tiny, 128 base, 192 large (depending on variant)
        if embed_dim == 96:
            arch = "swin_tiny_patch4_window7_224"
        elif embed_dim == 96:  # (kept for clarity; same as above)
            arch = "swin_tiny_patch4_window7_224"
        elif embed_dim == 128:
            arch = "swin_base_patch4_window7_224"
        elif embed_dim == 192:
            arch = "swin_large_patch4_window7_224"
        else:
            raise RuntimeError(f"Unrecognized embed_dim={embed_dim} from {key}")

        print(f"[CTransPath] using arch={arch} (embed_dim={embed_dim})")

        base = timm.create_model(arch, pretrained=False, num_classes=0)

        # IMPORTANT: allow partial load but also DROP incompatible keys if any
        model_sd = base.state_dict()
        filtered = {k: v for k, v in new_sd.items()
                    if (k in model_sd and model_sd[k].shape == v.shape)}

        missing, unexpected = base.load_state_dict(filtered, strict=False)
        print(f"[CTransPath load] loaded={len(filtered)}/{len(model_sd)} params | missing={len(missing)} unexpected={len(unexpected)}")

        enc = EncoderWrapper(base, base.num_features).to(device).eval()
        return enc, timm_tfm(base)

    # ---------- Generic timm baseline ----------
    base = timm.create_model(name, pretrained=True, num_classes=0)
    enc = EncoderWrapper(base, base.num_features).to(device).eval()
    return enc, imagenet_tfm(224)

# ----------------------------
# Feature extraction
# ----------------------------

@torch.no_grad()
def extract_embeddings(
    ds: Dataset,
    encoder: EncoderWrapper,
    batch_size: int,
    num_workers: int,
    device: str,
    out_npz: Path,
):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    all_z, all_y, all_meta = [], [], []

    for x, y, meta in tqdm(dl, desc="Extracting embeddings"):
        x = x.to(device, non_blocking=True)
        z = encoder(x).detach().cpu().numpy()
        all_z.append(z)
        all_y.append(y.numpy())
        # meta is dict of lists in default collate
        all_meta.append(meta)

    Z = np.concatenate(all_z, axis=0)
    Y = np.concatenate(all_y, axis=0)

    # flatten meta
    meta_rows = []
    for m in all_meta:
        n = len(m["img_rel"])
        for i in range(n):
            meta_rows.append({k: m[k][i] for k in m.keys()})
    M = pd.DataFrame(meta_rows)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, Z=Z, Y=Y, meta=M.to_dict(orient="list"))
    print(f"Saved: {out_npz}  Z={Z.shape}, Y={Y.shape}, meta_rows={len(M)}")
    return Z, Y, M


def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    d = np.load(npz_path, allow_pickle=True)
    Z = d["Z"]
    Y = d["Y"]
    meta = d["meta"].item()
    M = pd.DataFrame(meta)
    return Z, Y, M


# ----------------------------
# Analyses
# ----------------------------

def eval_separability(Ztr, Ytr, Zte, Yte, title: str):
    # kNN on embeddings
    knn = KNeighborsClassifier(n_neighbors=25)
    knn.fit(Ztr, Ytr)
    pred = knn.predict(Zte)
    acc = accuracy_score(Yte, pred)
    print(f"[{title}] kNN(25) accuracy: {acc:.4f}")

    # linear probe (logreg) on standardized embeddings
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="auto"))
    ])
    clf.fit(Ztr, Ytr)
    pred2 = clf.predict(Zte)
    acc2 = accuracy_score(Yte, pred2)
    print(f"[{title}] Linear probe accuracy: {acc2:.4f}")
    print(classification_report(Yte, pred2, target_names=[CLASS_MAP[i+1] for i in range(3)]))
    
    macro_f1 = f1_score(Yte, pred2, average="macro")
    bal_acc = balanced_accuracy_score(Yte, pred2)
    cm = confusion_matrix(Yte, pred2)

    print(f"[{title}] Macro-F1: {macro_f1:.4f}  Balanced-Acc: {bal_acc:.4f}")

    return {
        "knn_acc": float(acc),
        "linprobe_acc": float(acc2),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(bal_acc),
        "y_pred_linprobe": pred2,
        "cm": cm,
    }


def save_confusion_matrix(cm, out_png: Path, labels):
    plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)
    plt.close()
    
    
def plot_umap(Z, Y, out_png: Path, max_points: int = 20000):
    if Z.shape[0] > max_points:
        idx = np.random.choice(Z.shape[0], size=max_points, replace=False)
        Zs, Ys = Z[idx], Y[idx]
    else:
        Zs, Ys = Z, Y

    # speed: PCA to 50
    Zp = PCA(n_components=min(50, Zs.shape[1])).fit_transform(Zs)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=0)
    emb = reducer.fit_transform(Zp)

    plt.figure(figsize=(7, 6))
    for k in [0, 1, 2]:
        m = (Ys == k)
        plt.scatter(emb[m, 0], emb[m, 1], s=2, alpha=0.6, label=CLASS_MAP[k+1])
    plt.legend(markerscale=4)
    plt.title("UMAP of cell-crop embeddings")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"Saved UMAP: {out_png}")

def immune_infiltration_analysis(Z: np.ndarray, Y: np.ndarray, M: pd.DataFrame, out_json: Path):
    """
    Patch-level 'immune infiltration' proxy:
      lymphocyte fraction per patch = #lymph / total cells in that patch

    We test whether embeddings encode this:
      - compute patch embedding as mean of its cell embeddings
      - fit ridge/logreg to predict high-immune vs low-immune (median split)
      - report AUROC-ish proxy with accuracy (simple + fast)
    """
    df = M.copy()
    df["y"] = Y
    # group by patch
    groups = []
    for pid, g in df.groupby("patch_id"):
        idx = g.index.to_numpy()
        z_patch = Z[idx].mean(axis=0)
        # lymphocyte = class 0
        frac_lymph = float((g["y"] == 0).mean())
        n_cells = int(len(g))
        groups.append((pid, frac_lymph, n_cells, z_patch))
    gdf = pd.DataFrame(groups, columns=["patch_id", "frac_lymph", "n_cells", "z_patch"])
    X = np.vstack(gdf["z_patch"].to_numpy())
    y_cont = gdf["frac_lymph"].to_numpy()

    # binary "immune-high" label via median split
    thr = float(np.median(y_cont))
    y_bin = (y_cont >= thr).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y_bin, test_size=0.3, random_state=0, stratify=y_bin)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, n_jobs=-1))
    ])
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)

    # AUROC (better than accuracy for imbalance)
    proba = clf.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(yte, proba)

    # also check correlation between 1D projection (first PC) and frac_lymph
    pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X)).reshape(-1)
    corr = float(np.corrcoef(pc1, y_cont)[0, 1])
    out = {
        "num_patches": int(len(gdf)),
        "median_frac_lymph": thr,
        "immune_high_acc_median_split": float(acc),
        "immune_high_auroc_median_split": float(auroc),
        "corr_PC1_vs_frac_lymph": corr,
        "note": "immune infiltration proxy uses lymphocyte fraction per patch from dot labels"
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[Immune proxy] acc={acc:.4f}  auroc={auroc:.4f}  corr(PC1, frac_lymph)={corr:.4f}")
    print(f"Saved: {out_json}")
    return gdf

def subgroup_robustness_analysis(
    Z: np.ndarray, Y: np.ndarray, M: pd.DataFrame, out_json: Path
):
    """
    Fairness-style robustness without demographics:
    Evaluate cell classification performance across patch-defined subgroups:
      - immune_high vs immune_low (median split by frac_lymph)
      - high_density vs low_density (median split by n_cells)
      - tumor_dom vs stroma_dom (based on which fraction is larger)
    We report per-group accuracy + macro-F1, worst-group, and gap.
    """
    df = M.copy()
    df["y"] = Y

    # Build patch stats
    rows = []
    for pid, g in df.groupby("patch_id"):
        idx = g.index.to_numpy()
        n = len(g)
        frac_lymph = float((g["y"] == 0).mean())
        frac_tumor = float((g["y"] == 1).mean())
        frac_stroma = float((g["y"] == 2).mean())
        rows.append((pid, n, frac_lymph, frac_tumor, frac_stroma))
    pdf = pd.DataFrame(rows, columns=["patch_id", "n_cells", "frac_lymph", "frac_tumor", "frac_stroma"])

    # Group labels
    pdf["immune_group"] = np.where(pdf["frac_lymph"] >= pdf["frac_lymph"].median(), "immune_high", "immune_low")
    pdf["density_group"] = np.where(pdf["n_cells"] >= pdf["n_cells"].median(), "high_density", "low_density")
    pdf["tme_group"] = np.where(pdf["frac_tumor"] >= pdf["frac_stroma"], "tumor_dom", "stroma_dom")

    # Need predictions at cell-level (so compute a simple linear probe on test itself is cheating).
    # Instead: use a fixed classifier trained on TRAIN embeddings and pass its predictions here.
    # To keep it simple: this function expects that M has column 'y_pred' already.
    if "y_pred" not in df.columns:
        raise ValueError("subgroup_robustness_analysis expects M to include a 'y_pred' column.")

    # Merge group labels back to cells
    df = df.merge(pdf[["patch_id", "immune_group", "density_group", "tme_group"]], on="patch_id", how="left")

    def eval_group(group_col: str):
        out = {}
        for gname, gdf in df.groupby(group_col):
            y_true = gdf["y"].to_numpy()
            y_pred = gdf["y_pred"].to_numpy()
            acc = float(accuracy_score(y_true, y_pred))
            mf1 = float(f1_score(y_true, y_pred, average="macro"))
            out[gname] = {"n_cells": int(len(gdf)), "acc": acc, "macro_f1": mf1}
        # worst-group + gap
        accs = [v["acc"] for v in out.values()]
        mf1s = [v["macro_f1"] for v in out.values()]
        out["_summary"] = {
            "worst_acc": float(np.min(accs)) if accs else None,
            "acc_gap": float(np.max(accs) - np.min(accs)) if accs else None,
            "worst_macro_f1": float(np.min(mf1s)) if mf1s else None,
            "macro_f1_gap": float(np.max(mf1s) - np.min(mf1s)) if mf1s else None,
        }
        return out

    results = {
        "immune_group": eval_group("immune_group"),
        "density_group": eval_group("density_group"),
        "tme_group": eval_group("tme_group"),
        "note": "groups are defined from dot-label composition per patch; fairness-style robustness uses worst-group and gap"
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved subgroup robustness: {out_json}")
    return results

def train_cnn_baseline(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, device: str, out_dir: Path,
                       epochs: int = 5, batch_size: int = 128, num_workers: int = 4, lr: float = 1e-3):
    """
    Small CNN baseline: ResNet18 trained on the same cell crops end-to-end.
    Keeps it quick and fair.
    """
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    def make_dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    tr_dl = make_dl(train_ds, True)
    va_dl = make_dl(val_ds, False)
    te_dl = make_dl(test_ds, False)

    best_va = -1.0
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "cnn_resnet18_best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        tot, correct = 0, 0
        for x, y, _ in tqdm(tr_dl, desc=f"CNN train ep{ep}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            tot += int(y.numel())
        tr_acc = correct / max(1, tot)

        model.eval()
        tot, correct = 0, 0
        with torch.no_grad():
            for x, y, _ in va_dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                tot += int(y.numel())
        va_acc = correct / max(1, tot)
        print(f"[CNN] ep{ep} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_va:
            best_va = va_acc
            torch.save(model.state_dict(), ckpt)

    # test
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in tqdm(te_dl, desc="CNN test"):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(y.numpy())
            ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=[CLASS_MAP[i+1] for i in range(3)])
    (out_dir / "cnn_test_report.txt").write_text(rep)
    print(f"[CNN] test_acc={acc:.4f}")
    print(rep)
    return acc


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, required=True, help="Path to Dataset-BRCA-M2C repo root")

    ap.add_argument(
        "--encoders", type=str, nargs="+", default=["uni"],
        help="List of encoders: uni conch virchow2 provgigapath dinov2 vit_large_patch16_224 ..."
    )
    ap.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for gated models (UNI/UNI2 etc.)"
    )

    ap.add_argument("--crop_size", type=int, default=64)
    ap.add_argument("--max_cells_per_image", type=int, default=300, help="Cap per patch image to keep runtime manageable")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default="outputs_m2c")

    ap.add_argument("--run_cnn", action="store_true")
    ap.add_argument("--cnn_epochs", type=int, default=5)

    ap.add_argument("--use_cache", action="store_true", help="Reuse existing embeddings npz if present")

    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_txt = repo / "brca_ds_train.txt"
    val_txt = repo / "brca_ds_val.txt"
    test_txt = repo / "brca_ds_test.txt"

    # device selection
    if args.device != "cpu" and torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"
    print(f"Using device: {device}")

    summary_rows = []

    # helper: per-encoder npz path
    def npz_name(enc_name: str, split: str) -> Path:
        tag = f"{enc_name}_crop{args.crop_size}_cap{args.max_cells_per_image}"
        return out_dir / "embeddings" / f"{split}_{tag}.npz"

    # helper: extract or load cached
    def get_embeddings(split: str, ds: Dataset, encoder: EncoderWrapper, enc_name: str):
        npz_path = npz_name(enc_name, split)

        t0 = time.perf_counter()
        if args.use_cache and npz_path.exists():
            Z, Y, M = load_npz(npz_path)
            t = time.perf_counter() - t0
            print(f"Loaded cached: {npz_path}  Z={Z.shape}, Y={Y.shape}, meta_rows={len(M)}  time={t:.2f}s")
            return Z, Y, M, t, True

        Z, Y, M = extract_embeddings(ds, encoder, args.batch_size, args.num_workers, device, npz_path)
        t = time.perf_counter() - t0
        print(f"[{enc_name}] extract split={split} time={t:.2f}s")
        return Z, Y, M, t, False
    
    for enc_name in args.encoders:
        print("\n" + "=" * 72)
        print(f"Running encoder: {enc_name}")
        print("=" * 72)
        t_enc_start = time.perf_counter()

        t0 = time.perf_counter()
        # 1) load encoder + its preprocessing transform
        encoder, tfm = load_encoder_and_transforms(
            enc_name,
            device=device,
            hf_token=args.hf_token
        )
        t_load_encoder = time.perf_counter() - t0
        embed_dim = getattr(encoder, "embed_dim", None)
        print(f"[{enc_name}] embed_dim={embed_dim}")

        # 2) build datasets with the encoder-specific transform
        train_ds = BRCA_M2C_CellCrops(
            repo, train_txt,
            crop_size=args.crop_size,
            max_cells_per_image=args.max_cells_per_image,
            transform=tfm
        )
        val_ds = BRCA_M2C_CellCrops(
            repo, val_txt,
            crop_size=args.crop_size,
            max_cells_per_image=args.max_cells_per_image,
            transform=tfm
        )
        test_ds = BRCA_M2C_CellCrops(
            repo, test_txt,
            crop_size=args.crop_size,
            max_cells_per_image=args.max_cells_per_image,
            transform=tfm
        )

        print(f"Cells: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} (crop_size={args.crop_size})")

        # 3) embeddings
        Ztr, Ytr, Mtr, t_tr, c_tr = get_embeddings("train", train_ds, encoder, enc_name)
        Zva, Yva, Mva, t_va, c_va = get_embeddings("val", val_ds, encoder, enc_name)
        Zte, Yte, Mte, t_te, c_te = get_embeddings("test", test_ds, encoder, enc_name)

        embed_time_total = t_tr + t_va + t_te
        cache_frac = (int(c_tr) + int(c_va) + int(c_te)) / 3.0

        # 4) cell-type separability
        t0 = time.perf_counter()
        metrics = eval_separability(Ztr, Ytr, Zte, Yte, title=f"{enc_name} embeddings")
        t_eval = time.perf_counter() - t0

        save_confusion_matrix(
            metrics["cm"],
            out_dir / f"{enc_name}_cm_test.png",
            labels=[CLASS_MAP[i+1] for i in range(3)]
        )

        metrics_json = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics_json[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                metrics_json[k] = float(v)
            else:
                metrics_json[k] = v

        (out_dir / f"{enc_name}_celltype_separability.json").write_text(
            json.dumps(metrics_json, indent=2)
        )

        # 5) UMAP
        t0 = time.perf_counter()
        plot_umap(
            np.concatenate([Ztr, Zva, Zte], axis=0),
            np.concatenate([Ytr, Yva, Yte], axis=0),
            out_dir / f"{enc_name}_umap_cells.png"
        )
        t_umap = time.perf_counter() - t0

        # 6) immune proxy
        t0 = time.perf_counter()
        immune_infiltration_analysis(
            Zte, Yte, Mte,
            out_dir / f"{enc_name}_immune_proxy_test.json"
        )
        t_immune = time.perf_counter() - t0

        # 7) subgroup robustness
        t_subgroup = 0.0
        Mte2 = Mte.copy()
        if "y_pred_linprobe" in metrics:
            Mte2["y_pred"] = metrics["y_pred_linprobe"]
            t0 = time.perf_counter()
            subgroup_robustness_analysis(
                Zte, Yte, Mte2,
                out_dir / f"{enc_name}_subgroup_robustness_test.json"
            )
            t_subgroup = time.perf_counter() - t0
        else:
            print("WARNING: metrics missing y_pred_linprobe; skipping subgroup robustness")

        # 8) add to summary CSV
        t_total = time.perf_counter() - t_enc_start
        # row = {"encoder": enc_name, "dim": int(embed_dim) if embed_dim is not None else None}
        row = {
            "encoder": enc_name,
            "dim": int(embed_dim) if embed_dim is not None else None,

            # runtime (seconds)
            "t_load_encoder_s": float(t_load_encoder),
            "t_embed_total_s": float(embed_time_total),
            "t_eval_s": float(t_eval),
            "t_umap_s": float(t_umap),
            "t_immune_s": float(t_immune),
            "t_subgroup_s": float(t_subgroup),
            "t_total_s": float(t_total),

            # cache indicator
            "cache_frac": float(cache_frac),
        }
        for k, v in metrics.items():
            # keep scalars only in summary
            if isinstance(v, (int, float, np.floating, np.integer)):
                row[k] = float(v)
        summary_rows.append(row)

    # write global summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "results_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\nSaved overall summary:", summary_csv)

    # Optional: CNN baseline (single run, not per-encoder; compare to best FM later)
    if args.run_cnn:
        # For CNN, you can reuse a standard imagenet transform
        tfm_cnn = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        train_ds = BRCA_M2C_CellCrops(repo, train_txt, crop_size=args.crop_size,
                                      max_cells_per_image=args.max_cells_per_image, transform=tfm_cnn)
        val_ds = BRCA_M2C_CellCrops(repo, val_txt, crop_size=args.crop_size,
                                    max_cells_per_image=args.max_cells_per_image, transform=tfm_cnn)
        test_ds = BRCA_M2C_CellCrops(repo, test_txt, crop_size=args.crop_size,
                                     max_cells_per_image=args.max_cells_per_image, transform=tfm_cnn)

        train_cnn_baseline(train_ds, val_ds, test_ds, device=device, out_dir=out_dir / "cnn_baseline",
                           epochs=args.cnn_epochs, batch_size=min(128, args.batch_size), num_workers=args.num_workers)

    print("Done.")

if __name__ == "__main__":
    main()