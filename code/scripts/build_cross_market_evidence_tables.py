from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = PROJECT_ROOT / "review-stage" / "cross_market_modern_audit"
DATA_RUN = AUDIT_ROOT / "cross_market_local_20260513_2255"
STAGE2 = AUDIT_ROOT / "model_runs" / "models_stage2_20260514_093210"
STAGE3 = AUDIT_ROOT / "model_runs" / "models_stage3_modern_20260514_205937"
CHRONOS = AUDIT_ROOT / "model_runs" / "chronos_zero_shot_20260515_155918"
OUT = PROJECT_ROOT / "evidence_package"
if not OUT.exists():
    OUT = AUDIT_ROOT / "evidence_package"
SOURCE_TABLE_DIR = PROJECT_ROOT / "source" / "tables"


MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "dlinear": "DLinear",
    "patchtst": "PatchTST-style",
    "itransformer": "iTransformer-style",
    "timemixer": "TimeMixer-style",
    "chronos-t5-small": "Chronos-T5-small zero-shot",
}

DATASET_LABELS = {
    "dow30": "Dow 30",
    "nasdaq_large_tech": "Nasdaq large-tech",
    "sp100_largecap": "S&P 100 subset",
    "hang_seng_largecap": "Hang Seng large-cap",
    "etf_basket": "ETF basket",
}


def pct(x: float) -> str:
    return f"{100 * x:.2f}\\%"


def fmt(x: float, ndigits: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{ndigits}f}"


def latex_escape(s: object) -> str:
    text = str(s)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def bootstrap_ci(values: np.ndarray, func=np.median, n: int = 5000, seed: int = 7) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n, len(values)), replace=True)
    estimates = np.apply_along_axis(func, 1, samples)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def wilcoxon_greater_than_one(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan
    diffs = values - 1.0
    if np.allclose(diffs, 0):
        return 1.0
    return float(stats.wilcoxon(diffs, alternative="greater", zero_method="wilcox").pvalue)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        ratios = group["RMSE_ratio_vs_latest"].astype(float).to_numpy()
        lo, hi = bootstrap_ci(ratios, np.median)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "rows": len(group),
                "passes": int(group["passes_latest_rmse_gate"].sum()),
                "pass_rate": float(group["passes_latest_rmse_gate"].mean()),
                "mean_rmse_ratio": float(np.mean(ratios)),
                "median_rmse_ratio": float(np.median(ratios)),
                "median_ci_low": float(lo),
                "median_ci_high": float(hi),
                "q1_rmse_ratio": float(np.quantile(ratios, 0.25)),
                "q3_rmse_ratio": float(np.quantile(ratios, 0.75)),
                "min_rmse_ratio": float(np.min(ratios)),
                "p_wilcoxon_ratio_gt_1": wilcoxon_greater_than_one(ratios),
            }
        )
    return pd.DataFrame(rows)


def load_supervised() -> pd.DataFrame:
    frames = []
    for path in [
        STAGE2 / "worker0_dow_nasdaq_etf" / "model_results.csv",
        STAGE2 / "worker1_sp100_hangseng" / "model_results.csv",
        STAGE3 / "worker_modern_all" / "model_results.csv",
    ]:
        df = pd.read_csv(path)
        df["source_file"] = str(path.relative_to(PROJECT_ROOT))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "protocol": "full rolling-origin supervised audit",
            "dataset": df["dataset"],
            "ticker": df["ticker"],
            "price_mode": df["price_mode"],
            "horizon": df["horizon"].astype(int),
            "feature_set": df["feature_set"],
            "model_type": df["model_type"],
            "model": df["model_type"].map(MODEL_LABELS),
            "seed": df["seed"],
            "test_units": df["test_samples"],
            "test_RMSE": df["test_RMSE"],
            "test_MAE": df["test_MAE"],
            "test_MAPE": df["test_MAPE"],
            "latest_RMSE": df["latest_RMSE"],
            "latest_MAE": df["latest_MAE"],
            "latest_MAPE": df["latest_MAPE"],
            "RMSE_ratio_vs_latest": df["RMSE_ratio_vs_latest"],
            "MAE_ratio_vs_latest": df["MAE_ratio_vs_latest"],
            "MAPE_ratio_vs_latest": df["MAPE_ratio_vs_latest"],
            "passes_latest_rmse_gate": df["passes_latest_rmse_gate"].astype(int),
        }
    )
    return out


def load_chronos() -> pd.DataFrame:
    df = pd.read_csv(CHRONOS / "worker_chronos_all" / "chronos_zero_shot_results.csv")
    return pd.DataFrame(
        {
            "protocol": "sampled-origin zero-shot foundation audit",
            "dataset": df["dataset"],
            "ticker": df["ticker"],
            "price_mode": df["price_mode"],
            "horizon": df["horizon"].astype(int),
            "feature_set": "univariate close",
            "model_type": df["model_type"],
            "model": df["model_type"].map(MODEL_LABELS),
            "seed": np.nan,
            "test_units": df["test_origins"],
            "test_RMSE": df["test_RMSE"],
            "test_MAE": df["test_MAE"],
            "test_MAPE": df["test_MAPE"],
            "latest_RMSE": df["latest_RMSE"],
            "latest_MAE": df["latest_MAE"],
            "latest_MAPE": df["latest_MAPE"],
            "RMSE_ratio_vs_latest": df["RMSE_ratio_vs_latest"],
            "MAE_ratio_vs_latest": df["MAE_ratio_vs_latest"],
            "MAPE_ratio_vs_latest": df["MAPE_ratio_vs_latest"],
            "passes_latest_rmse_gate": df["passes_latest_rmse_gate"].astype(int),
        }
    )


def write_latex_tables(dataset_summary: pd.DataFrame, model_summary: pd.DataFrame, horizon_summary: pd.DataFrame) -> None:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-market dataset coverage.}")
    lines.append(r"\label{tab:dataset_coverage}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Assets & Date span \\")
    lines.append(r"\midrule")
    for _, row in dataset_summary.iterrows():
        lines.append(
            f"{latex_escape(row['dataset_label'])} & {int(row['assets'])} & "
            f"{latex_escape(row['start'])}--{latex_escape(row['end'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    core_order = [
        "LSTM",
        "GRU",
        "DLinear",
        "PatchTST-style",
        "iTransformer-style",
        "TimeMixer-style",
        "Chronos-T5-small zero-shot",
    ]
    ms = model_summary.set_index("model").loc[core_order].reset_index()
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Persistence-gate results across supervised and zero-shot models. Ratios below 1.0 beat latest-close persistence; values above 1.0 underperform it. Chronos uses the sampled-origin zero-shot protocol, while the other models use the full rolling-origin supervised protocol.}")
    lines.append(r"\label{tab:model_gate_summary}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Rows & Passes & Pass rate & Median RMSE ratio & 95\% CI \\")
    lines.append(r"\midrule")
    for _, row in ms.iterrows():
        ci = f"[{fmt(row['median_ci_low'])}, {fmt(row['median_ci_high'])}]"
        lines.append(
            f"{latex_escape(row['model'])} & {int(row['rows'])} & {int(row['passes'])} & "
            f"{pct(row['pass_rate'])} & {fmt(row['median_rmse_ratio'])} & {ci} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("")

    hs = horizon_summary.copy()
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Median RMSE ratio by forecast horizon.}")
    lines.append(r"\label{tab:horizon_summary}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model group & 1-day & 5-day & 10-day \\")
    lines.append(r"\midrule")
    for model in core_order:
        sub = hs[hs["model"] == model].set_index("horizon")
        vals = [fmt(sub.loc[h, "median_rmse_ratio"]) if h in sub.index else "--" for h in [1, 5, 10]]
        lines.append(f"{latex_escape(model)} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    core_table = "\n".join(lines)
    (OUT / "core_tables.tex").write_text(core_table, encoding="utf-8")
    SOURCE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (SOURCE_TABLE_DIR / "crossmarket_core_tables.tex").write_text(core_table, encoding="utf-8")


def write_summary_md(model_summary: pd.DataFrame, dataset_model_summary: pd.DataFrame) -> None:
    top = model_summary.sort_values("median_rmse_ratio")
    md = [
        "# Cross-Market Evidence Package",
        "",
        "The evidence package combines the no-training baselines, supervised neural models, compact modern time-series models, and Chronos-T5-small zero-shot audit.",
        "",
        "## Main finding",
        "",
        "Latest-close persistence remains a hard gate. Supervised neural models almost never beat it, and Chronos-T5-small zero-shot has more local wins but still has median RMSE ratio above 1.0.",
        "",
        "## Model summary",
        "",
        "```",
        top[
            [
                "model",
                "rows",
                "passes",
                "pass_rate",
                "median_rmse_ratio",
                "median_ci_low",
                "median_ci_high",
                "p_wilcoxon_ratio_gt_1",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Dataset-model pass summary",
        "",
        "```",
        dataset_model_summary[
            ["dataset_label", "model", "rows", "passes", "pass_rate", "median_rmse_ratio"]
        ].to_string(index=False),
        "```",
    ]
    (OUT / "EVIDENCE_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(DATA_RUN / "dataset_download_summary.csv")
    dataset_summary = (
        dataset.groupby("dataset")
        .agg(assets=("ticker", "nunique"), start=("start", "min"), end=("end", "max"))
        .reset_index()
    )
    dataset_summary["dataset_label"] = dataset_summary["dataset"].map(DATASET_LABELS)

    combined = pd.concat([load_supervised(), load_chronos()], ignore_index=True)
    combined["dataset_label"] = combined["dataset"].map(DATASET_LABELS)
    combined.to_csv(OUT / "combined_model_results.csv", index=False)
    dataset_summary.to_csv(OUT / "dataset_coverage.csv", index=False)

    model_summary = summarize(combined, ["model"])
    model_summary.to_csv(OUT / "model_gate_summary.csv", index=False)
    horizon_summary = summarize(combined, ["model", "horizon"])
    horizon_summary.to_csv(OUT / "model_horizon_summary.csv", index=False)
    dataset_model_summary = summarize(combined, ["dataset_label", "model"])
    dataset_model_summary.to_csv(OUT / "dataset_model_summary.csv", index=False)
    protocol_summary = summarize(combined, ["protocol", "model"])
    protocol_summary.to_csv(OUT / "protocol_model_summary.csv", index=False)

    best = combined.sort_values("RMSE_ratio_vs_latest").head(100)
    worst = combined.sort_values("RMSE_ratio_vs_latest", ascending=False).head(100)
    best.to_csv(OUT / "best_100_model_rows.csv", index=False)
    worst.to_csv(OUT / "worst_100_model_rows.csv", index=False)

    write_latex_tables(dataset_summary, model_summary, horizon_summary)
    write_summary_md(model_summary, dataset_model_summary)
    print(OUT)


if __name__ == "__main__":
    main()
