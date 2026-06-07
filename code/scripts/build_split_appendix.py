from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RUN = PROJECT_ROOT / "review-stage" / "cross_market_modern_audit" / "cross_market_local_20260513_2255"
EVIDENCE = PROJECT_ROOT / "evidence_package"
if not EVIDENCE.exists():
    EVIDENCE = PROJECT_ROOT / "review-stage" / "cross_market_modern_audit" / "evidence_package"
OUT_TEX = PROJECT_ROOT / "source" / "tables" / "crossmarket_split_appendix_tables.tex"

DATASET_LABELS = {
    "dow30": "Dow 30",
    "nasdaq_large_tech": "Nasdaq large-tech",
    "sp100_largecap": "S&P 100 subset",
    "hang_seng_largecap": "Hang Seng large-cap",
    "etf_basket": "ETF basket",
}


def esc(value: object) -> str:
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def resolve_raw_path(raw_file: str) -> Path:
    p = Path(raw_file)
    if p.exists():
        return p
    name_parts = p.parts
    if "raw" in name_parts:
        idx = name_parts.index("raw")
        candidate = DATA_RUN / Path(*name_parts[idx:])
        if candidate.exists():
            return candidate
    candidate = PROJECT_ROOT / raw_file
    if candidate.exists():
        return candidate
    raise FileNotFoundError(raw_file)


def split_dates_for_file(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, parse_dates=["Date"])
    n = len(df)
    train_end = int(n * 0.72)
    val_end = int(n * 0.80)
    return {
        "first_date": df["Date"].iloc[0].date().isoformat(),
        "train_val_boundary": df["Date"].iloc[train_end].date().isoformat(),
        "val_test_boundary": df["Date"].iloc[val_end].date().isoformat(),
        "last_date": df["Date"].iloc[-1].date().isoformat(),
        "rows": n,
    }


def main() -> None:
    summary = pd.read_csv(DATA_RUN / "dataset_download_summary.csv")
    rows = []
    for dataset, group in summary.groupby("dataset"):
        first = group.iloc[0]
        dates = split_dates_for_file(resolve_raw_path(str(first["raw_file"])))
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "assets": group["ticker"].nunique(),
                **dates,
            }
        )
    split_summary = pd.DataFrame(rows).sort_values("dataset_label")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    split_summary.to_csv(EVIDENCE / "split_boundary_summary.csv", index=False)

    lines = [
        r"\begin{center}",
        r"\small",
        r"\textbf{Table S5. Nominal chronological split boundaries by dataset.}\\[0.5em]",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & Assets & First date & Train/validation boundary & Validation/test boundary & Last date \\",
        r"\midrule",
    ]
    for _, row in split_summary.iterrows():
        lines.append(
            f"{esc(row['dataset_label'])} & {int(row['assets'])} & {row['first_date']} & "
            f"{row['train_val_boundary']} & {row['val_test_boundary']} & {row['last_date']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{center}",
        "",
        r"\noindent Exact split indices can vary slightly after indicator construction and horizon-specific target shifting. The CSV file \texttt{split\_boundary\_summary.csv} records the raw-download nominal split boundaries, while model-result CSV files record train, validation, and test sample counts for each row.",
    ]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_TEX)


if __name__ == "__main__":
    main()
