from __future__ import annotations

import argparse
import shutil
import json
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


CONFIG_PATH = Path("configs/cross_market_modern_audit_protocol.json")
DEFAULT_RUN_ROOT = Path("review-stage/cross_market_modern_audit")


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    script_root = Path(__file__).resolve().parents[1]
    return cwd if (cwd / "scripts").exists() and (cwd / "configs").exists() else script_root


def load_protocol(project_root: Path) -> dict:
    return json.loads((project_root / CONFIG_PATH).read_text(encoding="utf-8"))


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_download(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def download_one(ticker: str, start: str, end: str, retries: int, sleep: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                timeout=30,
            )
            if not df.empty:
                return normalize_download(df)
            last_error = ValueError("empty dataframe")
        except Exception as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(sleep * attempt)
    raise ValueError(f"{ticker} download failed after {retries} attempts: {last_error}")


def load_project_cache(project_root: Path, ticker: str, end: str) -> pd.DataFrame | None:
    cache_path = project_root / "data" / "raw" / f"{ticker.replace('/', '_')}.csv"
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if required - set(df.columns):
            return None
        requested_end = pd.to_datetime(end)
        max_date = pd.to_datetime(df["Date"]).max()
        if max_date < requested_end - pd.Timedelta(days=45):
            return None
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return None


def close_column(price_mode: str) -> str:
    if price_mode == "raw_close":
        return "Close"
    if price_mode == "adjusted_close":
        return "Adj Close"
    raise ValueError(f"Unknown price mode: {price_mode}")


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_true.astype(float) - y_pred.astype(float)
    return {
        "RMSE": float((err.pow(2).mean()) ** 0.5),
        "MAE": float(err.abs().mean()),
        "MAPE": float((err.abs() / y_true.abs().clip(lower=1e-8)).mean() * 100),
    }


def compute_baselines_for_series(
    close: pd.Series,
    horizon: int,
    split_ratios: dict,
) -> list[dict]:
    frame = pd.DataFrame({"close_t": close.astype(float)})
    frame["target"] = frame["close_t"].shift(-horizon)
    frame = frame.dropna().reset_index(drop=True)
    train_end = int(len(frame) * float(split_ratios["train"]))
    val_end = int(len(frame) * (float(split_ratios["train"]) + float(split_ratios["validation"])))
    if train_end <= 20 or val_end >= len(frame) - 20:
        raise ValueError(f"not enough rows after horizon={horizon}: {len(frame)}")

    train = frame.iloc[:train_end].copy()
    test = frame.iloc[val_end:].copy()
    drift = float(train["close_t"].diff(horizon).dropna().mean())
    train_mean = float(train["close_t"].mean())

    preds = {
        "latest_close": test["close_t"],
        "drift_adjusted": test["close_t"] + drift,
        "rolling_mean_5": frame["close_t"].rolling(5, min_periods=1).mean().iloc[val_end:],
        "rolling_mean_20": frame["close_t"].rolling(20, min_periods=1).mean().iloc[val_end:],
        "train_mean": pd.Series([train_mean] * len(test), index=test.index),
    }
    rows = []
    for name, pred in preds.items():
        m = metrics(test["target"], pred)
        rows.append(
            {
                "baseline": name,
                "test_RMSE": m["RMSE"],
                "test_MAE": m["MAE"],
                "test_MAPE": m["MAPE"],
                "test_samples": len(test),
                "train_samples": len(train),
                "validation_samples": int(val_end - train_end),
            }
        )
    return rows


def sign_p(all_failures: int, n: int) -> str:
    if all_failures != n:
        return "not_all_fail"
    log10_p = -n * math.log10(2)
    return "<1e-99" if log10_p < -99 else f"{10 ** log10_p:.3g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download cross-market data and compute no-training baseline gates.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--download-delay", type=float, default=8.0)
    parser.add_argument("--no-project-cache", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    project_root = resolve_project_root()
    protocol = load_protocol(project_root)
    run_dir = args.run_dir or (project_root / DEFAULT_RUN_ROOT / f"cross_market_{run_id()}")
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol_snapshot.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    raw_root = run_dir / "raw"
    summary_rows: list[dict] = []
    failure_rows: list[dict] = []
    baseline_rows: list[dict] = []

    start = protocol["date_range"]["start"]
    end = protocol["date_range"]["end"]

    for dataset_name, dataset in protocol["datasets"].items():
        dataset_raw = raw_root / dataset_name
        dataset_raw.mkdir(parents=True, exist_ok=True)
        for ticker in dataset["tickers"]:
            csv_path = dataset_raw / f"{ticker.replace('/', '_')}.csv"
            try:
                cached = None if args.no_project_cache else load_project_cache(project_root, ticker, end)
                if args.skip_download and csv_path.exists():
                    df = pd.read_csv(csv_path, parse_dates=["Date"])
                elif cached is not None:
                    df = cached
                    shutil.copyfile(project_root / "data" / "raw" / f"{ticker.replace('/', '_')}.csv", csv_path)
                else:
                    df = download_one(ticker, start, end, args.retries, args.retry_sleep)
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    if args.download_delay > 0:
                        time.sleep(args.download_delay)

                required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
                missing = sorted(required - set(df.columns))
                if missing:
                    raise ValueError(f"missing columns: {missing}")
                if len(df) < 800:
                    raise ValueError(f"too few rows: {len(df)}")

                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "ticker": ticker,
                        "rows": len(df),
                        "start": str(pd.to_datetime(df["Date"]).min().date()),
                        "end": str(pd.to_datetime(df["Date"]).max().date()),
                        "missing_values": int(df[list(required)].isna().sum().sum()),
                        "raw_file": csv_path.relative_to(project_root).as_posix(),
                    }
                )

                for price_mode in protocol["price_modes"]:
                    col = close_column(price_mode)
                    close = df[col].dropna().reset_index(drop=True)
                    for horizon in protocol["horizons"]:
                        for row in compute_baselines_for_series(close, int(horizon), protocol["split_ratios"]):
                            baseline_rows.append(
                                {
                                    "dataset": dataset_name,
                                    "market": dataset["market"],
                                    "ticker": ticker,
                                    "price_mode": price_mode,
                                    "horizon": int(horizon),
                                    **row,
                                }
                            )
            except Exception as exc:
                failure_rows.append({"dataset": dataset_name, "ticker": ticker, "error": str(exc)})
                print(f"[WARN] {dataset_name}/{ticker}: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    failures_df = pd.DataFrame(failure_rows)
    baselines_df = pd.DataFrame(baseline_rows)
    summary_df.to_csv(run_dir / "dataset_download_summary.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(run_dir / "download_failures.csv", index=False, encoding="utf-8-sig")
    baselines_df.to_csv(run_dir / "naive_baseline_results.csv", index=False, encoding="utf-8-sig")

    if not baselines_df.empty:
        latest = baselines_df[baselines_df["baseline"] == "latest_close"][
            ["dataset", "ticker", "price_mode", "horizon", "test_RMSE", "test_MAE", "test_MAPE"]
        ].rename(
            columns={
                "test_RMSE": "latest_RMSE",
                "test_MAE": "latest_MAE",
                "test_MAPE": "latest_MAPE",
            }
        )
        merged = baselines_df.merge(latest, on=["dataset", "ticker", "price_mode", "horizon"], how="left")
        for metric in ["RMSE", "MAE", "MAPE"]:
            merged[f"{metric}_ratio_vs_latest"] = merged[f"test_{metric}"] / merged[f"latest_{metric}"]
        merged.to_csv(run_dir / "naive_baseline_with_gate.csv", index=False, encoding="utf-8-sig")

        gate = (
            merged.groupby(["dataset", "price_mode", "horizon", "baseline"])
            .agg(
                assets=("ticker", "nunique"),
                mean_rmse_ratio=("RMSE_ratio_vs_latest", "mean"),
                median_rmse_ratio=("RMSE_ratio_vs_latest", "median"),
                min_rmse_ratio=("RMSE_ratio_vs_latest", "min"),
                max_rmse_ratio=("RMSE_ratio_vs_latest", "max"),
                pass_count=("RMSE_ratio_vs_latest", lambda s: int((s <= 1.0).sum())),
            )
            .reset_index()
            .sort_values(["dataset", "price_mode", "horizon", "mean_rmse_ratio"])
        )
        gate.to_csv(run_dir / "naive_baseline_gate_summary.csv", index=False, encoding="utf-8-sig")

    readme = [
        "# Cross-Market Modern Audit Run",
        "",
        f"Run directory: `{run_dir.relative_to(project_root).as_posix()}`",
        f"Started/finished: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "This first-stage run downloads cross-market OHLCV data and computes no-training baseline gates.",
        "It intentionally does not claim neural or foundation-model results yet.",
        "",
        "Generated files:",
        "- `dataset_download_summary.csv`",
        "- `download_failures.csv`",
        "- `naive_baseline_results.csv`",
        "- `naive_baseline_with_gate.csv`",
        "- `naive_baseline_gate_summary.csv`",
    ]
    (run_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Cross-market data audit completed: {run_dir}")
    print(f"Downloaded assets: {summary_df['ticker'].nunique() if not summary_df.empty else 0}")
    print(f"Failures: {len(failure_rows)}")


if __name__ == "__main__":
    main()
