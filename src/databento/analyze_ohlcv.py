#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore


REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close"]
WEEKDAY_ORDER = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Analyze Databento OHLCV data and export summary tables.")
    p.add_argument(
        "--input",
        default="data/databento/nq_ohlcv_1m_5y.csv",
        help="Input OHLCV file (.csv, .parquet, .pq).",
    )
    p.add_argument(
        "--outdir",
        default="results/databento",
        help="Output directory for analysis artifacts.",
    )
    return p.parse_args()


def load_ohlcv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input extension: {suffix}")

    if "datetime" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            pass
        elif len(df.columns) > 0:
            first_col = str(df.columns[0])
            df = df.rename(columns={first_col: "datetime"})
        else:
            raise ValueError("Unable to identify datetime column.")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        bad_dt = int(df["datetime"].isna().sum())
        if bad_dt > 0:
            df = df.dropna(subset=["datetime"]).copy()
        df = df.set_index("datetime")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex after loading input file.")

    missing_price_cols = [c for c in REQUIRED_PRICE_COLUMNS if c not in df.columns]
    if missing_price_cols:
        raise ValueError(f"Input data missing required columns: {missing_price_cols}")

    for col in REQUIRED_PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    pre_drop_rows = len(df)
    df = df.dropna(subset=REQUIRED_PRICE_COLUMNS).copy()
    dropped_na_rows = pre_drop_rows - len(df)

    pre_dupe_rows = len(df)
    df = df[~df.index.duplicated(keep="first")].copy()
    dropped_duplicate_rows = pre_dupe_rows - len(df)

    df = df.sort_index()

    meta = {
        "dropped_na_rows": int(dropped_na_rows),
        "dropped_duplicate_rows": int(dropped_duplicate_rows),
    }
    return df, meta


def infer_expected_bar_minutes(idx: pd.DatetimeIndex) -> float:
    diffs = idx.to_series().diff().dt.total_seconds().div(60.0).dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 0.0

    mode_vals = diffs.mode()
    if not mode_vals.empty:
        return float(mode_vals.iloc[0])
    return float(diffs.median())


def classify_gap(gap_minutes: float) -> str:
    if gap_minutes >= 2_000:
        return "weekend_or_holiday_gap"
    if gap_minutes >= 60:
        return "session_or_holiday_gap"
    return "short_unexpected_gap"


def build_gap_report(idx: pd.DatetimeIndex, expected_bar_minutes: float) -> pd.DataFrame:
    columns = [
        "previous_bar",
        "next_bar",
        "gap_minutes",
        "missing_bars_est",
        "gap_type",
    ]
    if expected_bar_minutes <= 0 or len(idx) < 2:
        return pd.DataFrame(columns=columns)

    diffs = idx.to_series().diff().dt.total_seconds().div(60.0)
    mask = diffs > (expected_bar_minutes * 1.5)
    if not bool(mask.any()):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for next_bar, gap_minutes in diffs[mask].items():
        loc = int(idx.get_loc(next_bar))
        prev_bar = idx[loc - 1] if loc > 0 else pd.NaT
        missing_bars = max(int(round(float(gap_minutes) / expected_bar_minutes)) - 1, 0)
        rows.append(
            {
                "previous_bar": prev_bar,
                "next_bar": next_bar,
                "gap_minutes": float(gap_minutes),
                "missing_bars_est": int(missing_bars),
                "gap_type": classify_gap(float(gap_minutes)),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def build_overall_stats(
    df: pd.DataFrame,
    expected_bar_minutes: float,
    gap_report: pd.DataFrame,
    ingest_meta: dict[str, Any],
) -> pd.DataFrame:
    close = df["close"]
    bar_returns = close.pct_change()
    range_bp = ((df["high"] - df["low"]) / df["open"]).replace([pd.NA], 0.0) * 10_000.0
    up_bar_rate = (df["close"] > df["open"]).mean()

    span_seconds = (df.index.max() - df.index.min()).total_seconds() if len(df) > 1 else 0.0
    span_days = span_seconds / 86_400.0 if span_seconds > 0 else 0.0
    rows_per_day = (len(df) / span_days) if span_days > 0 else 0.0
    annualized_vol_pct = (
        float(bar_returns.std()) * (rows_per_day * 365.25) ** 0.5 * 100.0
        if rows_per_day > 0 and pd.notna(bar_returns.std())
        else 0.0
    )

    out = pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "start": df.index.min(),
                "end": df.index.max(),
                "expected_bar_minutes": float(expected_bar_minutes),
                "span_days": float(span_days),
                "rows_per_day_observed": float(rows_per_day),
                "close_first": float(close.iloc[0]),
                "close_last": float(close.iloc[-1]),
                "total_return_pct": float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0),
                "mean_return_bp": float(bar_returns.mean() * 10_000.0),
                "std_return_bp": float(bar_returns.std() * 10_000.0),
                "annualized_vol_pct": float(annualized_vol_pct),
                "up_bar_rate": float(up_bar_rate),
                "avg_range_bp": float(range_bp.mean()),
                "median_volume": float(df["volume"].median()),
                "sum_volume": float(df["volume"].sum()),
                "gap_count": int(len(gap_report)),
                "gap_missing_bars_est_sum": int(
                    gap_report["missing_bars_est"].sum() if not gap_report.empty else 0
                ),
                "dropped_na_rows": int(ingest_meta["dropped_na_rows"]),
                "dropped_duplicate_rows": int(ingest_meta["dropped_duplicate_rows"]),
            }
        ]
    )
    return out


def build_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["month"] = frame.index.to_period("M").astype(str)
    frame["bar_return_bp"] = frame["close"].pct_change() * 10_000.0
    frame["range_bp"] = ((frame["high"] - frame["low"]) / frame["open"]) * 10_000.0

    monthly = (
        frame.groupby("month")
        .agg(
            rows=("close", "size"),
            open_first=("open", "first"),
            high_max=("high", "max"),
            low_min=("low", "min"),
            close_last=("close", "last"),
            volume_sum=("volume", "sum"),
            mean_return_bp=("bar_return_bp", "mean"),
            std_return_bp=("bar_return_bp", "std"),
            mean_range_bp=("range_bp", "mean"),
        )
        .reset_index()
    )
    monthly["month_return_pct"] = (monthly["close_last"] / monthly["open_first"] - 1.0) * 100.0
    return monthly


def build_hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["hour"] = frame.index.hour
    frame["up_bar"] = (frame["close"] > frame["open"]).astype(int)
    frame["bar_return_bp"] = frame["close"].pct_change() * 10_000.0
    frame["range_bp"] = ((frame["high"] - frame["low"]) / frame["open"]) * 10_000.0

    hourly = (
        frame.groupby("hour")
        .agg(
            rows=("close", "size"),
            up_bar_rate=("up_bar", "mean"),
            mean_return_bp=("bar_return_bp", "mean"),
            std_return_bp=("bar_return_bp", "std"),
            mean_range_bp=("range_bp", "mean"),
            mean_volume=("volume", "mean"),
            sum_volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values("hour")
        .reset_index(drop=True)
    )
    return hourly


def build_weekday_stats(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["weekday"] = frame.index.day_name()
    frame["up_bar"] = (frame["close"] > frame["open"]).astype(int)
    frame["bar_return_bp"] = frame["close"].pct_change() * 10_000.0
    frame["range_bp"] = ((frame["high"] - frame["low"]) / frame["open"]) * 10_000.0

    weekday = (
        frame.groupby("weekday")
        .agg(
            rows=("close", "size"),
            up_bar_rate=("up_bar", "mean"),
            mean_return_bp=("bar_return_bp", "mean"),
            std_return_bp=("bar_return_bp", "std"),
            mean_range_bp=("range_bp", "mean"),
            mean_volume=("volume", "mean"),
            sum_volume=("volume", "sum"),
        )
        .reset_index()
    )
    weekday["weekday_order"] = weekday["weekday"].map(WEEKDAY_ORDER).fillna(99).astype(int)
    weekday = weekday.sort_values("weekday_order").drop(columns=["weekday_order"]).reset_index(drop=True)
    return weekday


def write_summary_text(
    path: Path,
    input_path: Path,
    overall: pd.DataFrame,
    monthly: pd.DataFrame,
    gaps: pd.DataFrame,
) -> None:
    row = overall.iloc[0]
    lines = [
        f"Input: {input_path}",
        f"Rows: {int(row['rows']):,}",
        f"Range: {row['start']} -> {row['end']}",
        f"Expected bar size (min): {row['expected_bar_minutes']}",
        f"Total return (%): {row['total_return_pct']:.4f}",
        f"Annualized vol (%): {row['annualized_vol_pct']:.4f}",
        f"Up-bar rate: {row['up_bar_rate']:.4f}",
        f"Gap count: {int(row['gap_count'])}",
        f"Estimated missing bars from gaps: {int(row['gap_missing_bars_est_sum'])}",
        f"Months analyzed: {len(monthly)}",
    ]
    if not gaps.empty:
        largest = gaps.sort_values("gap_minutes", ascending=False).head(3)
        lines.append("Top 3 largest gaps (minutes): " + ", ".join(f"{v:.1f}" for v in largest["gap_minutes"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)

    df, ingest_meta = load_ohlcv(input_path)
    if df.empty:
        raise SystemExit("Input data contains zero valid rows after cleaning.")

    expected_bar_minutes = infer_expected_bar_minutes(df.index)
    gap_report = build_gap_report(df.index, expected_bar_minutes)
    overall = build_overall_stats(df, expected_bar_minutes, gap_report, ingest_meta)
    monthly = build_monthly_stats(df)
    hourly = build_hourly_stats(df)
    weekday = build_weekday_stats(df)

    outdir.mkdir(parents=True, exist_ok=True)

    overall_path = outdir / "overall_stats.csv"
    monthly_path = outdir / "monthly_stats.csv"
    hourly_path = outdir / "hourly_stats.csv"
    weekday_path = outdir / "weekday_stats.csv"
    gaps_path = outdir / "gap_report.csv"
    summary_json_path = outdir / "analysis_summary.json"
    summary_txt_path = outdir / "analysis_summary.txt"

    overall.to_csv(overall_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    hourly.to_csv(hourly_path, index=False)
    weekday.to_csv(weekday_path, index=False)
    gap_report.to_csv(gaps_path, index=False)

    summary_payload = {
        "input_path": str(input_path),
        "output_dir": str(outdir),
        "rows": int(overall.iloc[0]["rows"]),
        "start": str(overall.iloc[0]["start"]),
        "end": str(overall.iloc[0]["end"]),
        "expected_bar_minutes": float(overall.iloc[0]["expected_bar_minutes"]),
        "gap_count": int(overall.iloc[0]["gap_count"]),
        "estimated_missing_bars_from_gaps": int(overall.iloc[0]["gap_missing_bars_est_sum"]),
        "files": {
            "overall_stats_csv": str(overall_path),
            "monthly_stats_csv": str(monthly_path),
            "hourly_stats_csv": str(hourly_path),
            "weekday_stats_csv": str(weekday_path),
            "gap_report_csv": str(gaps_path),
            "analysis_summary_txt": str(summary_txt_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    write_summary_text(summary_txt_path, input_path, overall, monthly, gap_report)

    print(f"Saved: {overall_path}")
    print(f"Saved: {monthly_path}")
    print(f"Saved: {hourly_path}")
    print(f"Saved: {weekday_path}")
    print(f"Saved: {gaps_path}")
    print(f"Saved: {summary_json_path}")
    print(f"Saved: {summary_txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
