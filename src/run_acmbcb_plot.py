#!/usr/bin/env python3
"""
Generate paper-ready figures from results_all_methods_and_ablations.csv
for the BRCA-M2C frozen PFM context paper.

Figures generated
-----------------
Fig1  performance_heatmap_macro_f1.png
      Encoder x method heatmap (main methods, macro-F1)

Fig2  context_gain_barplot.png
      Patch-context gain over cell-only MLP per encoder

Fig3  pareto_runtime_vs_macrof1.png
      Accuracy-efficiency tradeoff (runtime vs macro-F1)

Fig4  method_rank_lines.png
      Per-encoder method ranking trends

Fig5  encoder_best_method_heatmap.png
      Best score per encoder/method family

Fig6  ablation_feature_mode.png
      Cell-only vs patch-only vs cell+patch vs cell+patch+residual

Fig7  ablation_patch_agg.png
      Mean vs max aggregation

Fig8  ablation_patch_budget.png
      Effect of patch cell budget (100/200/300)

Fig9  ablation_graph_k.png
      Graph neighborhood size sensitivity

Fig10 ablation_localctx_k.png
      Local-context neighborhood size sensitivity

Also writes CSV summaries used for tables.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ENCODER_COLORS = {
#     "provgigapath": "#08306b",
#     "virchow2": "#08519c",
#     "uni": "#2171b5",
#     "ctranspath": "#4292c6",
#     "dinov2_vitl": "#6baed6",
#     "mae_vitl": "#9ecae1",
#     "vit_large_patch16_224": "#c6dbef",
#     "resnet50": "#4d4d4d",
#     "conch": "#969696",
# }

ENCODER_COLORS = {
    "provgigapath": "#1f77b4",          # blue
    "uni": "#ff7f0e",                   # orange
    "dinov2_vitl": "#2ca02c",           # green
    "conch": "#d62728",                 # red
    "virchow2": "#9467bd",              # purple
    "mae_vitl": "#8c564b",              # brown
    "resnet50": "#e377c2",              # pink
    "ctranspath": "#7f7f7f",            # gray
    "vit_large_patch16_224": "#bcbd22", # yellow-green
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_heatmap(df_plot: pd.DataFrame, title: str, out_png: Path, fmt: str = ".3f", cmap_name: str = "Blues"):
    arr = df_plot.to_numpy(dtype=float)
    fig_w = max(8, 0.8 * len(df_plot.columns))
    fig_h = max(5, 0.55 * len(df_plot.index))

    cmap = cm.get_cmap(cmap_name)
    finite_vals = arr[np.isfinite(arr)]
    vmin = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
    vmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(len(df_plot.columns)), df_plot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(df_plot.index)), df_plot.index)

    norm = Normalize(vmin=vmin, vmax=vmax)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                txt_color = "white" if norm(arr[i, j]) > 0.58 else "black"
                plt.text(j, i, format(arr[i, j], fmt), ha="center", va="center", fontsize=8, color=txt_color)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


def grouped_barplot(df_plot: pd.DataFrame, x: str, y: str, hue: str, title: str, out_png: Path,
                    ylabel: str = None, legend_outside: bool = False):
    xcats = list(df_plot[x].astype(str).unique())
    hcats = list(df_plot[hue].astype(str).unique())

    width = 0.8 / max(1, len(hcats))
    base = np.arange(len(xcats))

    # sequential blue shades, dark to light
    palette = ["#163a5f", "#2f6c99", "#5b8fb9", "#9ecae1", "#c6dbef"]

    plt.figure(figsize=(max(8, 0.9 * len(xcats)), 5))

    for j, hc in enumerate(hcats):
        sub = df_plot[df_plot[hue].astype(str) == hc].set_index(x).reindex(xcats)
        vals = sub[y].to_numpy(dtype=float)
        xpos = base + j * width - 0.4 + width / 2
        plt.bar(
            xpos, vals,
            width=width,
            label=hc,
            color=palette[j % len(palette)],
            edgecolor="black",
            linewidth=0.4
        )

    plt.xticks(base, xcats, rotation=45, ha="right")
    plt.ylabel(ylabel or y)
    plt.title(title)

    if legend_outside:
        plt.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plt.tight_layout(rect=[0, 0, 0.88, 1])
    else:
        plt.legend(frameon=False)
        plt.tight_layout()

    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


def lineplot_multi(df_plot: pd.DataFrame, x: str, y: str, group: str, title: str, out_png: Path,
                   ylabel: str = None):
    # plt.figure(figsize=(max(8, 0.9 * len(df_plot[x].astype(str).unique())), 5))
    plt.figure(figsize=(max(12, 1.6 * len(df_plot[x].astype(str).unique())), 6))

    xcats = list(df_plot[x].astype(str).unique())
    xmap = {v: i for i, v in enumerate(xcats)}

    palette = [
        "#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6",
        "#9ecae1", "#c6dbef", "#4d4d4d", "#969696"
    ]

    # for idx, g in enumerate(df_plot[group].astype(str).unique()):
    groups = sorted(df_plot[group].astype(str).unique())
    for g in groups:
        sub = df_plot[df_plot[group].astype(str) == g].copy()
        xs = [xmap[v] for v in sub[x].astype(str)]
        ys = sub[y].to_numpy(dtype=float)
        # plt.plot(xs, ys, marker="o", linewidth=2.0, label=g, color=palette[idx % len(palette)])
        plt.plot(
            xs,
            ys,
            marker='o',
            markersize=7,
            linewidth=2.2,
            alpha=0.9,
            label=g,
            color=ENCODER_COLORS.get(g, "#333333")
        )

    plt.xticks(range(len(xcats)), xcats, rotation=35, ha="right")
    plt.ylabel(ylabel or y)
    plt.title(title)
    plt.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


def scatter_runtime_tradeoff(df_plot: pd.DataFrame, runtime_col: str, score_col: str, label_col: str,
                             title: str, out_png: Path):

    # Fixed color mapping (consistent across figures)
    METHOD_COLORS = {
        "mlp_head": "#1f77b4",                  # blue
        "patch_context_mlp": "#2ca02c",         # green
        "hybrid_morph_context": "#17becf",      # cyan
        "linear_probe": "#ff7f0e",              # orange
        "graph_label_propagation": "#9467bd",   # purple
        "residual_boosting": "#d62728",         # red
    }

    plot_df = df_plot.sort_values(runtime_col).reset_index(drop=True)

    plt.figure(figsize=(8, 5.8))

    xs = plot_df[runtime_col].to_numpy(dtype=float)
    ys = plot_df[score_col].to_numpy(dtype=float)
    labels = plot_df[label_col].tolist()

    # Assign colors per method
    colors = [METHOD_COLORS.get(lbl, "#333333") for lbl in labels]

    # Scatter with per-point colors
    plt.scatter(xs, ys, s=150, c=colors, edgecolors='black', linewidths=0.6)

    # Offsets to avoid overlap
    offsets = {
        "mlp_head": (0.6, -0.0008),
        "patch_context_mlp": (0.6, -0.0012),
        "hybrid_morph_context": (0.6, 0.0012),
        "linear_probe": (0.6, -0.0010),
        "graph_label_propagation": (0.6, 0.0010),
        "residual_boosting": (0.6, 0.0008),
    }

    # Annotate with matching colors
    for _, r in plot_df.iterrows():
        lbl = r[label_col]
        dx, dy = offsets.get(lbl, (0.6, 0.0005))
        color = METHOD_COLORS.get(lbl, "#333333")

        plt.text(
            float(r[runtime_col]) + dx,
            float(r[score_col]) + dy,
            lbl,
            fontsize=10,
            color=color,
            va="center"
        )

    # Pareto frontier (optional)
    pareto_points = []
    best_so_far = -np.inf
    for x, y in sorted(zip(xs, ys), key=lambda t: t[0]):
        if y > best_so_far:
            pareto_points.append((x, y))
            best_so_far = y

    if len(pareto_points) >= 2:
        px, py = zip(*pareto_points)
        plt.plot(px, py, linestyle="--", linewidth=1.5, color="#9ecae1")

    # Axes and formatting
    plt.xlabel("Runtime (s)")
    plt.ylabel("Macro-F1")
    plt.title(title)

    plt.yticks(np.round(np.linspace(min(ys) - 0.002, max(ys) + 0.002, 5), 3))
    plt.ylim(min(ys) - 0.004, max(ys) + 0.004)  # slightly expanded for readability
    plt.xlim(0, max(xs) + 8)

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='paper_figures_brca_m2c')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / 'figures'
    tab_dir = out_dir / 'tables'
    ensure_dir(fig_dir)
    ensure_dir(tab_dir)

    df = pd.read_csv(csv_path)

    # Main methods for paper
    main_methods = [
        'linear_probe',
        'mlp_head',
        'residual_boosting',
        'patch_context_mlp',
        'graph_label_propagation',
        'hybrid_morph_context',
        'multi_encoder_context_fusion',
    ]
    negative_methods = ['local_context_mlp', 'multiscale_proxy_mlp']

    # Preferred encoder order
    enc_order = [
        'provgigapath', 'virchow2', 'uni', 'ctranspath', 'dinov2_vitl',
        'mae_vitl', 'vit_large_patch16_224', 'resnet50', 'conch', 'fusion'
    ]
    method_order = [m for m in main_methods if m in df['method'].unique()] + [m for m in negative_methods if m in df['method'].unique()]

    # ------------------------------------------------------------------
    # Main results tables
    # ------------------------------------------------------------------
    main_df = df[df['method'].isin(method_order)].copy()
    main_df['encoder'] = pd.Categorical(main_df['encoder'], categories=[e for e in enc_order if e in main_df['encoder'].unique()], ordered=True)
    main_df['method'] = pd.Categorical(main_df['method'], categories=method_order, ordered=True)
    main_df = main_df.sort_values(['encoder', 'method']).reset_index(drop=True)
    main_df.to_csv(tab_dir / 'main_methods_table.csv', index=False)

    # best per encoder
    best_per_encoder = main_df[main_df['encoder'] != 'fusion'].sort_values(['encoder', 'balanced_acc', 'macro_f1', 'acc'], ascending=[True, False, False, False]).groupby('encoder').head(1)
    best_per_encoder.to_csv(tab_dir / 'best_per_encoder.csv', index=False)

    # average main methods across encoders (excluding fusion)
    avg_main = main_df[(main_df['encoder'] != 'fusion') & (main_df['method'].isin(main_methods))].groupby('method')[['acc', 'macro_f1', 'balanced_acc', 'runtime_s']].mean().reset_index()
    avg_main = avg_main.sort_values(['balanced_acc', 'macro_f1', 'acc'], ascending=False)
    avg_main.to_csv(tab_dir / 'average_main_methods.csv', index=False)

    # ------------------------------------------------------------------
    # Fig 1: Encoder x method heatmap (macro-F1)
    # ------------------------------------------------------------------
    heat = main_df[main_df['method'].isin(main_methods)].pivot_table(index='encoder', columns='method', values='macro_f1', aggfunc='mean')
    heat = heat.loc[[e for e in enc_order if e in heat.index], [m for m in main_methods if m in heat.columns]]
    # save_heatmap(heat, 'Macro-F1 across encoders and methods', fig_dir / 'performance_heatmap_macro_f1.png')

    # Also balanced acc heatmap
    heat_bal = main_df[main_df['method'].isin(main_methods)].pivot_table(index='encoder', columns='method', values='balanced_acc', aggfunc='mean')
    heat_bal = heat_bal.loc[[e for e in enc_order if e in heat_bal.index], [m for m in main_methods if m in heat_bal.columns]]
    # save_heatmap(heat_bal, 'Balanced accuracy across encoders and methods', fig_dir / 'performance_heatmap_balanced_acc.png')

    # ------------------------------------------------------------------
    # Fig 2: Context gain over cell-only MLP baseline
    # ------------------------------------------------------------------
    # Prefer ablation cell_only when available, else fallback to mlp_head
    cell_only = df[df['method'] == 'ablate_patchfeat_cell_only'][['encoder', 'macro_f1', 'acc', 'balanced_acc']].rename(columns={'macro_f1': 'macro_f1_cell', 'acc': 'acc_cell', 'balanced_acc': 'bal_cell'})
    if len(cell_only) == 0:
        cell_only = df[df['method'] == 'mlp_head'][['encoder', 'macro_f1', 'acc', 'balanced_acc']].rename(columns={'macro_f1': 'macro_f1_cell', 'acc': 'acc_cell', 'balanced_acc': 'bal_cell'})
    patch_ctx = df[df['method'] == 'patch_context_mlp'][['encoder', 'macro_f1', 'acc', 'balanced_acc']].rename(columns={'macro_f1': 'macro_f1_patch', 'acc': 'acc_patch', 'balanced_acc': 'bal_patch'})
    gain = pd.merge(cell_only, patch_ctx, on='encoder', how='inner')
    gain['delta_macro_f1'] = gain['macro_f1_patch'] - gain['macro_f1_cell']
    gain['delta_acc'] = gain['acc_patch'] - gain['acc_cell']
    gain['delta_bal'] = gain['bal_patch'] - gain['bal_cell']
    gain = gain.sort_values('delta_macro_f1', ascending=False)
    gain.to_csv(tab_dir / 'context_gain_vs_cell_only.csv', index=False)
    plt.figure(figsize=(max(8, 0.9 * len(gain)), 5))
    xs = np.arange(len(gain))
    vals = gain['delta_macro_f1'].to_numpy(dtype=float)
    plt.bar(xs, vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(xs, gain['encoder'].astype(str).tolist(), rotation=45, ha='right')
    plt.ylabel('Δ Macro-F1 (patch context - cell only)')
    plt.title('Patch-level context gain across encoders')
    plt.tight_layout()
    plt.savefig(fig_dir / 'context_gain_barplot.png', dpi=250)
    plt.close()

    # ------------------------------------------------------------------
    # Fig 3: Pareto plot runtime vs macro-F1 (average over encoders)
    # ------------------------------------------------------------------
    pareto = avg_main.copy()
    pareto['label'] = pareto['method']
    scatter_runtime_tradeoff(
        pareto,
        runtime_col="runtime_s",
        score_col="macro_f1",
        label_col="label",
        title="Performance-efficiency tradeoff across main methods",
        out_png=fig_dir / "pareto_runtime_vs_macrof1.png",
    )
    # ------------------------------------------------------------------
    # Fig 4: Method lines across encoders
    # ------------------------------------------------------------------
    rank_df = main_df[(main_df['encoder'] != 'fusion') & (main_df['method'].isin(['mlp_head', 'patch_context_mlp', 'graph_label_propagation', 'hybrid_morph_context']))].copy()
    rank_df['encoder'] = rank_df['encoder'].astype(str)
    rank_df = rank_df.sort_values(['encoder', 'method'])
    lineplot_multi(rank_df, 'encoder', 'macro_f1', 'method', 'Method performance trends across encoders', fig_dir / 'method_rank_lines.png', ylabel='Macro-F1')

    # ------------------------------------------------------------------
    # Fig 5: Best score per encoder/method family heatmap
    # ------------------------------------------------------------------
    family_map = {
        'linear_probe': 'baseline',
        'mlp_head': 'baseline',
        'residual_boosting': 'baseline',
        'patch_context_mlp': 'patch_context',
        'hybrid_morph_context': 'hybrid_context',
        'graph_label_propagation': 'graph_context',
        'local_context_mlp': 'local_context',
        'multiscale_proxy_mlp': 'multiscale_proxy',
        'multi_encoder_context_fusion': 'fusion',
    }
    fam_df = main_df.copy()
    fam_df['family'] = fam_df['method'].map(family_map)
    best_family = fam_df.groupby(['encoder', 'family'])['macro_f1'].max().reset_index()
    fam_heat = best_family.pivot_table(index='encoder', columns='family', values='macro_f1')
    fam_order = ['baseline', 'patch_context', 'hybrid_context', 'graph_context', 'local_context', 'multiscale_proxy', 'fusion']
    fam_heat = fam_heat.loc[[e for e in enc_order if e in fam_heat.index], [f for f in fam_order if f in fam_heat.columns]]
    # save_heatmap(fam_heat, 'Best Macro-F1 by encoder and method family', fig_dir / 'encoder_best_method_heatmap.png')

    save_heatmap(
        heat,
        "Macro-F1 across encoders and methods",
        fig_dir / "performance_heatmap_macro_f1.png",
        cmap_name="Blues",
    )

    save_heatmap(
        heat_bal,
        "Balanced accuracy across encoders and methods",
        fig_dir / "performance_heatmap_balanced_acc.png",
        cmap_name="Blues",
    )

    save_heatmap(
        fam_heat,
        "Best Macro-F1 by encoder and method family",
        fig_dir / "encoder_best_method_heatmap.png",
        cmap_name="Blues",
    )
    # ------------------------------------------------------------------
    # Ablation figures and tables
    # ------------------------------------------------------------------
    # Feature mode
    feat = df[df['method'].astype(str).str.startswith('ablate_patchfeat_')].copy()
    if len(feat):
        feat.to_csv(tab_dir / 'ablation_feature_mode.csv', index=False)
        feat_plot = feat.sort_values(['encoder', 'feature_mode'])
        # grouped_barplot(feat_plot, 'encoder', 'macro_f1', 'feature_mode', 'Feature composition ablation', fig_dir / 'ablation_feature_mode.png', ylabel='Macro-F1')
        grouped_barplot(
            feat_plot,
            "encoder",
            "macro_f1",
            "feature_mode",
            "Feature composition ablation",
            fig_dir / "ablation_feature_mode.png",
            ylabel="Macro-F1",
            legend_outside=True,
        )

    # Patch agg
    agg = df[df['method'].astype(str).str.startswith('ablate_patchagg_')].copy()
    if len(agg):
        agg.to_csv(tab_dir / 'ablation_patch_agg.csv', index=False)
        # grouped_barplot(agg.sort_values(['encoder', 'patch_agg']), 'encoder', 'macro_f1', 'patch_agg', 'Patch aggregation ablation', fig_dir / 'ablation_patch_agg.png', ylabel='Macro-F1')
        grouped_barplot(
            agg.sort_values(["encoder", "patch_agg"]),
            "encoder",
            "macro_f1",
            "patch_agg",
            "Patch aggregation ablation",
            fig_dir / "ablation_patch_agg.png",
            ylabel="Macro-F1",
            legend_outside=True,
        )
        
    # Patch budget
    budget = df[df['method'].astype(str).str.startswith('ablate_patchbudget_')].copy()
    if len(budget):
        budget.to_csv(tab_dir / 'ablation_patch_budget.csv', index=False)
        lineplot_multi(budget.sort_values(['patch_budget']), 'patch_budget', 'macro_f1', 'encoder', 'Patch cell budget sensitivity', fig_dir / 'ablation_patch_budget.png', ylabel='Macro-F1')

    # Graph k
    gk = df[df['method'].astype(str).str.startswith('ablate_graph_k')].copy()
    if len(gk):
        gk.to_csv(tab_dir / 'ablation_graph_k.csv', index=False)
        lineplot_multi(gk.sort_values(['k']), 'k', 'macro_f1', 'encoder', 'Graph neighborhood size sensitivity', fig_dir / 'ablation_graph_k.png', ylabel='Macro-F1')

    # Localctx k
    lk = df[df['method'].astype(str).str.startswith('ablate_localctx_k')].copy()
    if len(lk):
        lk.to_csv(tab_dir / 'ablation_localctx_k.csv', index=False)
        lineplot_multi(lk.sort_values(['k']), 'k', 'macro_f1', 'encoder', 'Local-context neighborhood size sensitivity', fig_dir / 'ablation_localctx_k.png', ylabel='Macro-F1')

    # ------------------------------------------------------------------
    # Extra useful tables for paper writing
    # ------------------------------------------------------------------
    # Table: strongest model overall
    strongest = df.sort_values(['balanced_acc', 'macro_f1', 'acc'], ascending=False).head(20)
    strongest.to_csv(tab_dir / 'top20_all_runs.csv', index=False)

    # Table: best patch-context per encoder
    patch_best = df[df['method'].isin(['patch_context_mlp', 'hybrid_morph_context', 'graph_label_propagation'])].sort_values(['encoder', 'balanced_acc', 'macro_f1', 'acc'], ascending=[True, False, False, False]).groupby('encoder').head(1)
    patch_best.to_csv(tab_dir / 'best_context_method_per_encoder.csv', index=False)

    print('\nSaved figures to:', fig_dir)
    print('Saved tables to:', tab_dir)
    print('\nMost useful files:')
    print(fig_dir / 'performance_heatmap_macro_f1.png')
    print(fig_dir / 'context_gain_barplot.png')
    print(fig_dir / 'pareto_runtime_vs_macrof1.png')
    print(tab_dir / 'main_methods_table.csv')
    print(tab_dir / 'best_per_encoder.csv')
    print(tab_dir / 'average_main_methods.csv')


if __name__ == '__main__':
    main()
