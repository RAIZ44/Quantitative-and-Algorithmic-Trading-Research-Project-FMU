#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd  # type: ignore

try:
    import databento as db  # type: ignore
except Exception as exc:
    raise SystemExit(
        "Missing dependency: databento\n"
        f"Import error: {exc}\n"
        "Install with: pip install databento"
    )


def normalize_ohlcv(df_raw: pd.DataFrame, local_tz: str) -> pd.DataFrame:
    df = df_raw.copy()
    if "ts_event" not in df.columns:
        df = df.reset_index()
    if "ts_event" not in df.columns:
        raise ValueError("Could not locate 'ts_event' in Databento dataframe.")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    # Fixed-point guard
    close_abs_med = df["close"].abs().median()
    if pd.notna(close_abs_med) and close_abs_med > 1_000_000:
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col] / 1_000_000_000.0

    ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Found {int(ts.isna().sum())} rows with invalid ts_event.")

    dt_local = ts.dt.tz_convert(local_tz).dt.tz_localize(None)

    out = (
        df.assign(datetime=dt_local)[["datetime", "open", "high", "low", "close", "volume"]]
        .dropna(subset=["datetime", "open", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="first")
        .set_index("datetime")
    )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Download Databento OHLCV-1h for a date range.")
    p.add_argument("--dataset", default="GLBX.MDP3")
    p.add_argument("--symbol", default="NQ.c.0")
    p.add_argument("--stype-in", default="continuous")
    p.add_argument("--start", required=True, help='UTC start, e.g. "2016-01-01T00:00:00Z"')
    p.add_argument("--end", required=True, help='UTC end, e.g. "2026-01-01T00:00:00Z"')
    p.add_argument("--local-tz", default="America/Chicago")
    p.add_argument(
        "--out",
        default="data/databento/nq_ohlcv_1h_10y.parquet",
        help="Output path. Supported extensions: .parquet, .pq, .csv",
    )
    p.add_argument("--api-key-env", default="DATABENTO_API_KEY")
    return p.parse_args()


def save_ohlcv(df: pd.DataFrame, out_path: Path) -> None:
    ext = out_path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        df.to_parquet(out_path)
        return
    if ext == ".csv":
        df.to_csv(out_path, index=True, index_label="datetime")
        return
    raise SystemExit(
        f"Unsupported --out extension: '{out_path.suffix}'. "
        "Use .parquet, .pq, or .csv."
    )


def main() -> int:
    args = parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = db.Historical(api_key)
    store = client.timeseries.get_range(
        dataset=args.dataset,
        schema="ohlcv-1h",
        symbols=args.symbol,
        stype_in=args.stype_in,
        start=args.start,
        end=args.end,
    )

    raw = store.to_df()
    if raw.empty:
        print("No rows returned.")
        return 1

    h1 = normalize_ohlcv(raw, args.local_tz)
    save_ohlcv(h1, out_path)
    print(f"Saved: {out_path} ({len(h1):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
