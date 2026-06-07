from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback for minimal reproduction environments.
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "evidence_package"
if not EVIDENCE.exists():
    EVIDENCE = PROJECT_ROOT / "review-stage" / "cross_market_modern_audit" / "evidence_package"
FIG_DIR = PROJECT_ROOT / "source" / "figures"
FIGURE_BASENAME = "Figure1_RMSE_Ratio_Distribution"

MODEL_ORDER = [
    "LSTM",
    "GRU",
    "DLinear",
    "PatchTST-style",
    "iTransformer-style",
    "TimeMixer-style",
    "Chronos-T5-small zero-shot",
]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EVIDENCE / "combined_model_results.csv")
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["RMSE_ratio_vs_latest"] = df["RMSE_ratio_vs_latest"].astype(float)

    # Keep the axis readable while still showing extreme failures through a capped marker.
    df["plot_ratio"] = np.minimum(df["RMSE_ratio_vs_latest"], 20.0)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    else:
        plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    palette = ["#557AA7", "#6C8FB1", "#7A7A7A", "#B5835A", "#A36C6C", "#5D9A83", "#C45A4B"]

    if sns is not None:
        sns.boxplot(
            data=df,
            x="model",
            y="plot_ratio",
            hue="model",
            order=MODEL_ORDER,
            hue_order=MODEL_ORDER,
            palette=palette,
            legend=False,
            width=0.62,
            fliersize=0.6,
            linewidth=0.9,
            ax=ax,
        )
    else:
        grouped = [df.loc[df["model"] == model, "plot_ratio"].to_numpy() for model in MODEL_ORDER]
        box = ax.boxplot(
            grouped,
            widths=0.62,
            patch_artist=True,
            showfliers=True,
            flierprops={"markersize": 0.6, "marker": "o"},
            medianprops={"color": "#111111", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )
        for patch, color in zip(box["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.9)
    ax.axhline(1.0, color="#111111", linestyle="--", linewidth=1.1)
    ax.text(4.6, 1.07, "persistence gate", ha="left", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.85, 22)
    ax.set_ylabel("RMSE ratio vs latest-close persistence (log scale)")
    ax.set_xlabel("")
    ax.set_title("Persistence-gate distribution across audited models", pad=10)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(
        [
            "LSTM",
            "GRU",
            "DLinear",
            "PatchTST-\nstyle",
            "iTransformer-\nstyle",
            "TimeMixer-\nstyle",
            "Chronos-T5-small\nzero-shot",
        ],
        rotation=0,
    )
    ax.grid(True, which="major", axis="y", linewidth=0.55, alpha=0.55)
    ax.grid(True, which="minor", axis="y", linewidth=0.25, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.0,
        -0.24,
        "Values above 20 are clipped for readability; full rows are retained in the evidence package.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{FIGURE_BASENAME}.eps", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{FIGURE_BASENAME}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{FIGURE_BASENAME}.png", dpi=220, bbox_inches="tight")
    print(FIG_DIR / f"{FIGURE_BASENAME}.eps")


if __name__ == "__main__":
    main()
