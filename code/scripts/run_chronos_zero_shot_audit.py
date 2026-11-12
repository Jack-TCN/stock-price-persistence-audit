from __future__ import annotations

import argparse
import csv
import itertools
import json
import hashlib
import random
import platform
import importlib.metadata
import time
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from scripts.run_cross_market_model_audit import adjusted_ohlcv, resolve_raw_path


DEFAULT_DATA_RUN = Path("review-stage/cross_market_modern_audit/cross_market_local_20260513_2255")
DEFAULT_OUT_ROOT = Path("review-stage/cross_market_modern_audit/model_runs")


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    script_root = Path(__file__).resolve().parents[1]
    return cwd if (cwd / "scripts").exists() and (cwd / "configs").exists() else script_root


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_completed(path: Path) -> set[tuple[str, ...]]:
    fields = ["dataset", "ticker", "price_mode", "horizon", "model_type", "model_id", "max_origins"]
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    return {tuple(str(row.get(field, "")) for field in fields) for _, row in df.iterrows()}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def make_eval_frame(raw_df: pd.DataFrame, price_mode: str, horizon: int) -> pd.DataFrame:
    df = raw_df.copy()
    dates = pd.to_datetime(df['Date'])
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Input dates must be valid, unique and increasing')
    if price_mode == "adjusted_close":
        df = adjusted_ohlcv(df)
    elif price_mode != "raw_close":
        raise ValueError(f"Unknown price mode: {price_mode}")
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["Date"]),
            "Close": df["Close"].astype(float),
        }
    )
    out["Target"] = out["Close"].shift(-int(horizon))
    out['TargetDate'] = out['Date'].shift(-int(horizon))
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(out) < 300:
        raise ValueError(f"Too few usable rows: {len(out)}")
    return out


def select_test_origins(n_rows: int, split_ratios: dict, max_origins: int) -> np.ndarray:
    val_end = int(n_rows * (float(split_ratios["train"]) + float(split_ratios["validation"])))
    origins = np.arange(val_end, n_rows, dtype=int)
    if len(origins) == 0:
        raise ValueError(f"No test origins for n_rows={n_rows}")
    if len(origins) <= max_origins:
        return origins
    take = np.linspace(0, len(origins) - 1, max_origins).round().astype(int)
    return origins[np.unique(take)]


def batched(iterable: list[torch.Tensor], size: int):
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size], start


def evaluate_asset(pipe, frame: pd.DataFrame, origins: np.ndarray, horizon: int, args, device: str) -> dict[str, float]:
    contexts: list[torch.Tensor] = []
    y_true: list[float] = []
    latest: list[float] = []
    for idx in origins:
        start = max(0, int(idx) - int(args.context_length) + 1)
        context = frame["Close"].iloc[start : int(idx) + 1].to_numpy(dtype=np.float32)
        contexts.append(torch.tensor(context, dtype=torch.float32))
        y_true.append(float(frame["Target"].iloc[int(idx)]))
        latest.append(float(frame["Close"].iloc[int(idx)]))

    preds: list[np.ndarray] = []
    for batch, _ in batched(contexts, int(args.batch_size)):
        with torch.no_grad():
            forecast = pipe.predict(
                batch,
                prediction_length=int(horizon),
                num_samples=int(args.num_samples),
                limit_prediction_length=False,
            )
        point = forecast.median(dim=1).values[:, int(horizon) - 1].detach().cpu().numpy()
        preds.append(point)
    y_pred = np.concatenate(preds).astype(float)
    y_true_arr = np.asarray(y_true, dtype=float)
    latest_arr = np.asarray(latest, dtype=float)

    if getattr(args, "prediction_path", None):
        args.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"origin_date": frame["Date"].iloc[origins].to_numpy(), "target_date": frame['TargetDate'].iloc[origins].to_numpy(), "target": y_true_arr,
                      "prediction": y_pred, "persistence": latest_arr}).to_csv(args.prediction_path, index=False)
    model_m = metrics(y_true_arr, y_pred)
    latest_m = metrics(y_true_arr, latest_arr)
    return {
        "test_origins": int(len(origins)),
        "test_RMSE": model_m["RMSE"],
        "test_MAE": model_m["MAE"],
        "test_MAPE": model_m["MAPE"],
        "latest_RMSE": latest_m["RMSE"],
        "latest_MAE": latest_m["MAE"],
        "latest_MAPE": latest_m["MAPE"],
        "RMSE_ratio_vs_latest": model_m["RMSE"] / latest_m["RMSE"] if latest_m["RMSE"] > 0 else math.nan,
        "MAE_ratio_vs_latest": model_m["MAE"] / latest_m["MAE"] if latest_m["MAE"] > 0 else math.nan,
        "MAPE_ratio_vs_latest": model_m["MAPE"] / latest_m["MAPE"] if latest_m["MAPE"] > 0 else math.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Chronos zero-shot persistence-gate audit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--model-revision', required=True, help='Immutable Hugging Face model commit SHA')
    parser.add_argument("--data-run", type=Path, default=DEFAULT_DATA_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--price-modes", nargs="+", default=["raw_close", "adjusted_close"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--model-id", default="amazon/chronos-t5-small")
    parser.add_argument("--model-type", default="chronos-t5-small")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--max-origins", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-assets", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    data_run = args.data_run if args.data_run.is_absolute() else project_root / args.data_run
    out_dir = args.out_dir or (project_root / DEFAULT_OUT_ROOT / f"chronos_zero_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "chronos_zero_shot_results.csv"
    error_path = out_dir / "chronos_zero_shot_errors.csv"
    status_path = out_dir / "status.json"

    from chronos import ChronosPipeline

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.device == "auto" and device == "cuda":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    if len(args.model_revision) != 40 or any(c not in '0123456789abcdef' for c in args.model_revision):
        raise ValueError('An immutable 40-character model commit is required')

    protocol = json.loads((data_run / "protocol_snapshot.json").read_text(encoding="utf-8"))
    split_ratios = protocol["split_ratios"]
    summary = pd.read_csv(data_run / "dataset_download_summary.csv")
    if args.datasets:
        summary = summary[summary["dataset"].isin(args.datasets)].copy()
    if args.limit_assets:
        summary = summary.head(int(args.limit_assets)).copy()
    if summary.empty or summary.duplicated(['dataset','ticker']).any():
        raise ValueError('Empty or duplicate asset selection')

    manifest = out_dir / "run_manifest.json"
    signature = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "helper_sha256": hashlib.sha256(Path(__file__).with_name('run_cross_market_model_audit.py').read_bytes()).hexdigest(),
                 "input_sha256": {f"{r.dataset}/{r.ticker}": hashlib.sha256(resolve_raw_path(project_root,data_run,r.raw_file).read_bytes()).hexdigest() for r in summary.itertuples()},
                 "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, "protocol": protocol,
                 "python": platform.python_version(), "device": device, "dtype": str(torch_dtype),
                 "packages": {k: importlib.metadata.version(k) for k in ['torch','numpy','pandas','chronos-forecasting','transformers','huggingface-hub']}}
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Resume refused: source/config changed")
    if result_path.exists() and not manifest.exists():
        raise ValueError("Cannot append to legacy output")
    manifest.write_text(json.dumps(signature, indent=2))
    pipe = ChronosPipeline.from_pretrained(args.model_id, revision=args.model_revision, device_map=device, torch_dtype=torch_dtype)
    completed = load_completed(result_path)
    combos = list(
        itertools.product(
            summary.to_dict("records"),
            args.price_modes,
            args.horizons,
        )
    )

    errors = 0
    for idx, (asset, price_mode, horizon) in enumerate(combos, start=1):
        row = {
            "dataset": asset["dataset"],
            "ticker": asset["ticker"],
            "price_mode": price_mode,
            "horizon": int(horizon),
            "model_type": args.model_type,
            "model_id": args.model_id,
            "max_origins": int(args.max_origins),
        }
        key = tuple(str(row[field]) for field in ["dataset", "ticker", "price_mode", "horizon", "model_type", "model_id", "max_origins"])
        if key in completed:
            continue
        print(f"[{idx}/{len(combos)}] {row}", flush=True)
        try:
            started = time.monotonic()
            raw_path = resolve_raw_path(project_root, data_run, str(asset["raw_file"]))
            raw_df = pd.read_csv(raw_path, parse_dates=["Date"])
            frame = make_eval_frame(raw_df, price_mode, int(horizon))
            origins = select_test_origins(len(frame), split_ratios, int(args.max_origins))
            row_seed = int.from_bytes(hashlib.sha256((str(args.seed)+repr(key)).encode()).digest()[:4], "little")
            random.seed(row_seed); np.random.seed(row_seed); torch.manual_seed(row_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(row_seed)
            args.prediction_path = out_dir / "predictions" / (hashlib.sha256('_'.join(key).encode()).hexdigest() + '.csv')
            result = evaluate_asset(pipe, frame, origins, int(horizon), args, device)
            out = {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                **row,
                "context_length": int(args.context_length),
                "num_samples": int(args.num_samples),
                "device": str(device),
                "row_seed": row_seed,
                "elapsed_seconds": time.monotonic() - started,
                **result,
            }
            out["passes_latest_rmse_gate"] = int(out["RMSE_ratio_vs_latest"] < 1.0)
            append_csv(result_path, out)
            completed.add(key)
        except Exception as exc:
            errors += 1
            append_csv(error_path, {**row, "error": repr(exc), "failed_at": datetime.now().isoformat(timespec="seconds")})
            print(f"[ERROR] {row}: {exc}", flush=True)
        if idx % 1 == 0:
            status_path.write_text(
                json.dumps(
                    {
                        "completed": len(completed),
                        "errors": errors,
                        "total": len(combos),
                        "device": str(device),
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    status_path.write_text(
        json.dumps(
            {
                "completed": len(completed),
                "errors": errors,
                "total": len(combos),
                "device": str(device),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Chronos zero-shot audit finished under {out_dir}")
    if len(completed) != len(combos):
        raise RuntimeError('Incomplete zero-shot run; inspect error CSV')


if __name__ == "__main__":
    main()
