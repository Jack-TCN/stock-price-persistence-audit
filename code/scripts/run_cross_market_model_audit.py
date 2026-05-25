from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


DEFAULT_DATA_RUN = Path("review-stage/cross_market_modern_audit/cross_market_local_20260513_2255")
DEFAULT_OUT_ROOT = Path("review-stage/cross_market_modern_audit/model_runs")
BASE_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
TECH_FEATURES = [
    "RSI_14",
    "MACD",
    "MACD_SIGNAL",
    "MACD_DIFF",
    "BB_MAVG",
    "BB_HIGH",
    "BB_LOW",
]


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


def row_key(row: dict, fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def load_completed(path: Path, fields: list[str]) -> set[tuple[str, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path)
    return {row_key(row.to_dict(), fields) for _, row in df.iterrows()}


def adjusted_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ratio = (out["Adj Close"].astype(float) / out["Close"].astype(float)).replace([np.inf, -np.inf], np.nan)
    for col in ["Open", "High", "Low", "Close"]:
        out[col] = out[col].astype(float) * ratio
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    out["RSI_14"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    out["MACD"] = macd.macd()
    out["MACD_SIGNAL"] = macd.macd_signal()
    out["MACD_DIFF"] = macd.macd_diff()
    bb = BollingerBands(close=close, window=20, window_dev=2)
    out["BB_MAVG"] = bb.bollinger_mavg()
    out["BB_HIGH"] = bb.bollinger_hband()
    out["BB_LOW"] = bb.bollinger_lband()
    return out


@dataclass
class WindowBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    target_mean: float
    target_std: float
    n_features: int
    train_samples: int
    validation_samples: int
    test_samples: int


def make_windows(
    raw_df: pd.DataFrame,
    price_mode: str,
    horizon: int,
    feature_set: str,
    window_size: int,
    split_ratios: dict,
) -> WindowBundle:
    df = raw_df.copy()
    if price_mode == "adjusted_close":
        df = adjusted_ohlcv(df)
    elif price_mode != "raw_close":
        raise ValueError(f"Unknown price mode: {price_mode}")

    df = add_indicators(df)
    df["Target"] = df["Close"].astype(float).shift(-int(horizon))
    feature_cols = BASE_FEATURES if feature_set == "ohlcv" else BASE_FEATURES + TECH_FEATURES
    needed = ["Date", "Target", *feature_cols]
    frame = df[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(frame) < window_size + 200:
        raise ValueError(f"Too few usable rows: {len(frame)}")

    train_end = int(len(frame) * float(split_ratios["train"]))
    val_end = int(len(frame) * (float(split_ratios["train"]) + float(split_ratios["validation"])))
    if train_end <= window_size or val_end >= len(frame) - 20:
        raise ValueError(f"Invalid split for usable rows={len(frame)}")

    scaler = MinMaxScaler()
    features = frame[feature_cols].astype(float).copy()
    features.iloc[:train_end] = scaler.fit_transform(features.iloc[:train_end])
    features.iloc[train_end:] = scaler.transform(features.iloc[train_end:])
    values = features.to_numpy(dtype=np.float32)
    target = frame["Target"].astype(float).to_numpy(dtype=np.float32)
    train_targets = target[window_size - 1 : train_end]
    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std())
    if target_std <= 1e-8:
        target_std = 1.0

    xs, ys, label_idx = [], [], []
    for idx in range(window_size - 1, len(frame)):
        xs.append(values[idx - window_size + 1 : idx + 1])
        ys.append(target[idx])
        label_idx.append(idx)
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    label_idx_arr = np.asarray(label_idx)
    train_mask = label_idx_arr < train_end
    val_mask = (label_idx_arr >= train_end) & (label_idx_arr < val_end)
    test_mask = label_idx_arr >= val_end

    def xt(mask: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x[mask])

    def yt(mask: np.ndarray, scaled: bool) -> torch.Tensor:
        raw = y[mask]
        if scaled:
            raw = (raw - target_mean) / target_std
        return torch.from_numpy(raw.astype(np.float32)).view(-1, 1)

    return WindowBundle(
        x_train=xt(train_mask),
        y_train=yt(train_mask, True),
        x_val=xt(val_mask),
        y_val=yt(val_mask, True),
        x_test=xt(test_mask),
        y_test=yt(test_mask, False),
        target_mean=target_mean,
        target_std=target_std,
        n_features=len(feature_cols),
        train_samples=int(train_mask.sum()),
        validation_samples=int(val_mask.sum()),
        test_samples=int(test_mask.sum()),
    )


class RecurrentRegressor(nn.Module):
    def __init__(self, model_type: str, n_features: int, hidden_size: int, dropout: float):
        super().__init__()
        cls = nn.LSTM if model_type == "lstm" else nn.GRU
        self.rnn = cls(n_features, hidden_size, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(self.dropout(out[:, -1, :]))


class DLinearRegressor(nn.Module):
    def __init__(self, window_size: int, n_features: int):
        super().__init__()
        self.time = nn.Linear(window_size, 1)
        self.head = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        per_feature = self.time(x.transpose(1, 2)).squeeze(-1)
        return self.head(per_feature)


class PatchTransformerRegressor(nn.Module):
    def __init__(
        self,
        window_size: int,
        n_features: int,
        d_model: int = 64,
        patch_len: int = 10,
        stride: int = 5,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = min(patch_len, window_size)
        self.stride = max(1, stride)
        self.n_features = n_features
        n_patches = 1 + max(0, (window_size - self.patch_len) // self.stride)
        self.proj = nn.Linear(self.patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_features * n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, L, C -> B, C, num_patches, patch_len -> B, tokens, patch_len
        patches = x.transpose(1, 2).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        tokens = patches.reshape(x.shape[0], -1, self.patch_len)
        h = self.proj(tokens) + self.pos[:, : tokens.shape[1], :]
        h = self.encoder(h).mean(dim=1)
        return self.head(h)


class InvertedTransformerRegressor(nn.Module):
    def __init__(
        self,
        window_size: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.value_proj = nn.Linear(window_size, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # iTransformer-style audit proxy: variables are tokens, each token contains its full temporal trace.
        tokens = self.value_proj(x.transpose(1, 2)) + self.var_embed
        h = self.encoder(tokens).mean(dim=1)
        return self.head(h)


class TimeMixerRegressor(nn.Module):
    def __init__(self, window_size: int, n_features: int, hidden_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scales = [1, 2, 4]
        in_dim = 0
        for scale in self.scales:
            in_dim += math.ceil(window_size / scale) * n_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps = []
        for scale in self.scales:
            if scale == 1:
                pooled = x
            else:
                pooled = nn.functional.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale,
                    ceil_mode=True,
                ).transpose(1, 2)
            reps.append(pooled.reshape(x.shape[0], -1))
        return self.net(torch.cat(reps, dim=1))


def build_model(model_type: str, window_size: int, n_features: int, hidden_size: int, dropout: float) -> nn.Module:
    if model_type in {"lstm", "gru"}:
        return RecurrentRegressor(model_type, n_features, hidden_size, dropout)
    if model_type == "dlinear":
        return DLinearRegressor(window_size, n_features)
    if model_type == "patchtst":
        return PatchTransformerRegressor(window_size, n_features, d_model=hidden_size, dropout=dropout)
    if model_type == "itransformer":
        return InvertedTransformerRegressor(window_size, n_features, d_model=hidden_size, dropout=dropout)
    if model_type == "timemixer":
        return TimeMixerRegressor(window_size, n_features, hidden_size=max(128, hidden_size * 2), dropout=dropout)
    raise ValueError(f"Unsupported model: {model_type}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true.astype(float) - y_pred.astype(float)
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err) / np.clip(np.abs(y_true.astype(float)), 1e-8, None)) * 100),
    }


def train_eval(bundle: WindowBundle, row: dict, args: argparse.Namespace, device: torch.device) -> dict:
    model = build_model(
        row["model_type"],
        int(row["window_size"]),
        bundle.n_features,
        int(row["hidden_size"]),
        float(row["dropout"]),
    ).to(device)
    train_ds = TensorDataset(bundle.x_train, bundle.y_train)
    generator = torch.Generator().manual_seed(int(row["seed"]))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(row["learning_rate"]), weight_decay=float(row["weight_decay"]))
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(bundle.x_val.to(device))
            val_loss = float(loss_fn(val_pred, bundle.y_val.to(device)).item())
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(bundle.x_test.to(device)).cpu().numpy().reshape(-1)
    pred = pred_scaled * bundle.target_std + bundle.target_mean
    y_true = bundle.y_test.numpy().reshape(-1)
    test = regression_metrics(y_true, pred)
    with torch.no_grad():
        val_pred = model(bundle.x_val.to(device)).cpu().numpy().reshape(-1) * bundle.target_std + bundle.target_mean
    val_true = bundle.y_val.numpy().reshape(-1) * bundle.target_std + bundle.target_mean
    val = regression_metrics(val_true, val_pred)
    return {
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_val_loss_scaled": best_val,
        "best_val_RMSE": val["RMSE"],
        "best_val_MAE": val["MAE"],
        "best_val_MAPE": val["MAPE"],
        "test_RMSE": test["RMSE"],
        "test_MAE": test["MAE"],
        "test_MAPE": test["MAPE"],
    }


def load_latest_baselines(data_run: Path) -> pd.DataFrame:
    df = pd.read_csv(data_run / "naive_baseline_results.csv")
    latest = df[df["baseline"] == "latest_close"][
        ["dataset", "ticker", "price_mode", "horizon", "test_RMSE", "test_MAE", "test_MAPE"]
    ].rename(
        columns={
            "test_RMSE": "latest_RMSE",
            "test_MAE": "latest_MAE",
            "test_MAPE": "latest_MAPE",
        }
    )
    return latest


def resolve_raw_path(project_root: Path, data_run: Path, raw_file: str) -> Path:
    rel = Path(str(raw_file).replace("\\", "/"))
    candidates = [
        project_root / rel,
        data_run / rel.name,
        data_run / "raw" / rel.parent.name / rel.name,
    ]
    parts = rel.parts
    if parts and parts[0] == project_root.name:
        candidates.append(project_root / Path(*parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot resolve raw file {raw_file}; tried: {[str(p) for p in candidates]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-market model audit against latest-close persistence.")
    parser.add_argument("--data-run", type=Path, default=DEFAULT_DATA_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=["lstm", "gru", "dlinear"])
    parser.add_argument("--feature-sets", nargs="+", default=["ohlcv", "technical"])
    parser.add_argument("--price-modes", nargs="+", default=["raw_close", "adjusted_close"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[20, 60])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777, 2025, 3407])
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    data_run = args.data_run if args.data_run.is_absolute() else project_root / args.data_run
    out_dir = args.out_dir or (project_root / DEFAULT_OUT_ROOT / f"models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "model_results.csv"
    error_path = out_dir / "model_errors.csv"
    status_path = out_dir / "status.json"

    protocol = json.loads((data_run / "protocol_snapshot.json").read_text(encoding="utf-8"))
    split_ratios = protocol["split_ratios"]
    summary = pd.read_csv(data_run / "dataset_download_summary.csv")
    if args.datasets:
        summary = summary[summary["dataset"].isin(args.datasets)].copy()
    latest = load_latest_baselines(data_run)

    key_fields = [
        "dataset",
        "ticker",
        "price_mode",
        "horizon",
        "feature_set",
        "model_type",
        "window_size",
        "seed",
    ]
    completed = load_completed(result_path, key_fields)
    errored = load_completed(error_path, key_fields)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    combos = []
    for _, asset in summary.iterrows():
        for price_mode, horizon, feature_set, model_type, window_size, seed in itertools.product(
            args.price_modes,
            args.horizons,
            args.feature_sets,
            args.models,
            args.window_sizes,
            args.seeds,
        ):
            combos.append(
                {
                    "dataset": asset["dataset"],
                    "ticker": asset["ticker"],
                    "price_mode": price_mode,
                    "horizon": int(horizon),
                    "feature_set": feature_set,
                    "model_type": model_type,
                    "window_size": int(window_size),
                    "seed": int(seed),
                    "hidden_size": int(args.hidden_size),
                    "learning_rate": float(args.learning_rate),
                    "dropout": float(args.dropout if model_type != "dlinear" else 0.0),
                    "weight_decay": float(args.weight_decay),
                }
            )

    for idx, row in enumerate(combos, start=1):
        key = row_key(row, key_fields)
        if key in completed or key in errored:
            continue
        print(f"[{idx}/{len(combos)}] {row}", flush=True)
        try:
            torch.manual_seed(int(row["seed"]))
            np.random.seed(int(row["seed"]))
            raw_file = summary[
                (summary["dataset"] == row["dataset"]) & (summary["ticker"] == row["ticker"])
            ].iloc[0]["raw_file"]
            raw_path = resolve_raw_path(project_root, data_run, str(raw_file))
            raw_df = pd.read_csv(raw_path, parse_dates=["Date"])
            bundle = make_windows(
                raw_df=raw_df,
                price_mode=row["price_mode"],
                horizon=int(row["horizon"]),
                feature_set=row["feature_set"],
                window_size=int(row["window_size"]),
                split_ratios=split_ratios,
            )
            result = train_eval(bundle, row, args, device)
            base = latest[
                (latest["dataset"] == row["dataset"])
                & (latest["ticker"] == row["ticker"])
                & (latest["price_mode"] == row["price_mode"])
                & (latest["horizon"] == int(row["horizon"]))
            ].iloc[0]
            out = {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                **row,
                "epochs": args.epochs,
                "patience": args.patience,
                "train_samples": bundle.train_samples,
                "validation_samples": bundle.validation_samples,
                "test_samples": bundle.test_samples,
                "target_mean_train": bundle.target_mean,
                "target_std_train": bundle.target_std,
                **result,
                "latest_RMSE": float(base["latest_RMSE"]),
                "latest_MAE": float(base["latest_MAE"]),
                "latest_MAPE": float(base["latest_MAPE"]),
            }
            out["RMSE_ratio_vs_latest"] = out["test_RMSE"] / out["latest_RMSE"]
            out["MAE_ratio_vs_latest"] = out["test_MAE"] / out["latest_MAE"]
            out["MAPE_ratio_vs_latest"] = out["test_MAPE"] / out["latest_MAPE"]
            out["passes_latest_rmse_gate"] = int(out["RMSE_ratio_vs_latest"] <= 1.0)
            append_csv(result_path, out)
            completed.add(key)
        except Exception as exc:
            append_csv(error_path, {**row, "error": repr(exc), "failed_at": datetime.now().isoformat(timespec="seconds")})
            errored.add(key)
            print(f"[ERROR] {row}: {exc}", flush=True)

        if idx % 25 == 0:
            status_path.write_text(
                json.dumps(
                    {
                        "completed": len(completed),
                        "errors": len(errored),
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
                "errors": len(errored),
                "total": len(combos),
                "device": str(device),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Model audit finished under {out_dir}")


if __name__ == "__main__":
    main()
