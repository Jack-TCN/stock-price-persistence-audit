from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT_ROOT / "review-stage" / "cross_market_modern_audit" / "evidence_package"
OUT = PROJECT_ROOT / "crossmarket_supplement_tables.tex"


def esc(value: object) -> str:
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}\\%"


def main() -> None:
    dataset_model = pd.read_csv(EVIDENCE / "dataset_model_summary.csv")
    protocol = pd.read_csv(EVIDENCE / "protocol_model_summary.csv")
    best = pd.read_csv(EVIDENCE / "best_100_model_rows.csv").head(20)
    worst = pd.read_csv(EVIDENCE / "worst_100_model_rows.csv").head(20)

    lines: list[str] = []
    lines.append(r"\begin{center}")
    lines.append(r"\small")
    lines.append(r"\textbf{Table 1. Protocol-level model summary.}\\[0.5em]")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Protocol & Model & Rows & Passes & Pass rate & Median ratio \\")
    lines.append(r"\midrule")
    for _, row in protocol.iterrows():
        lines.append(
            f"{esc(row['protocol'])} & {esc(row['model'])} & {int(row['rows'])} & "
            f"{int(row['passes'])} & {pct(row['pass_rate'])} & {fmt(row['median_rmse_ratio'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{center}")
    lines.append("")

    lines.append(r"\begin{center}")
    lines.append(r"\small")
    lines.append(r"\textbf{Table 2. Dataset-by-model persistence-gate summary.}\\[0.5em]")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Model & Rows & Passes & Pass rate & Median ratio \\")
    lines.append(r"\midrule")
    for _, row in dataset_model.iterrows():
        lines.append(
            f"{esc(row['dataset_label'])} & {esc(row['model'])} & {int(row['rows'])} & "
            f"{int(row['passes'])} & {pct(row['pass_rate'])} & {fmt(row['median_rmse_ratio'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{center}")
    lines.append("")

    for label, frame, table_id in [
        ("Best model rows by RMSE ratio", best, "tab:supp_best_rows"),
        ("Worst model rows by RMSE ratio", worst, "tab:supp_worst_rows"),
    ]:
        lines.append(r"\begin{center}")
        lines.append(r"\small")
        lines.append(rf"\textbf{{{label}.}}\\[0.5em]")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{llllrrr}")
        lines.append(r"\toprule")
        lines.append(r"Dataset & Ticker & Model & Horizon & Price & Ratio & Pass \\")
        lines.append(r"\midrule")
        for _, row in frame.iterrows():
            lines.append(
                f"{esc(row['dataset_label'])} & {esc(row['ticker'])} & {esc(row['model'])} & "
                f"{int(row['horizon'])} & {esc(row['price_mode'])} & {fmt(row['RMSE_ratio_vs_latest'])} & "
                f"{int(row['passes_latest_rmse_gate'])} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(r"\end{center}")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
