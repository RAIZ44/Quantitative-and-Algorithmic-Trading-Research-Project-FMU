"""
Pivot Retest Strategy (Hourly strong candle -> 5-min pivot touch -> TP stats)

This version includes:
- SAME-DIRECTION filter for strong candles:
  * Strong bull only counts if current candle is bullish AND previous candle is bullish.
  * Strong bear only counts if current candle is bearish AND previous candle is bearish.
- Mixed CSV delimiter support (';' or ',') per-file auto detection.
- Robust datetime parsing for DD/MM/YYYY HH:MM(/:SS).
- Output filenames tagged with RUN_TAG.

Outputs (per run):
- results/pivot_retest_5min_same_direction/events_<MONTH>.csv
- results/pivot_retest_5min_same_direction/summary_<MONTH>.csv
- results/pivot_retest_5min_same_direction/hourly_stats_<MONTH>.csv
"""

from __future__ import annotations

import os
import csv
import pandas as pd  # type: ignore
import numpy as np   # type: ignore


# ----------------------------------
# 1) CONFIG
# ----------------------------------
HOURLY_PATH = "data/nq-1h_bk_new.csv"
MINUTE_PATH = "data/nq-5m_bk.csv"

# Optional month filter in format "YYYY-MM" (e.g., "2026-01").
# Set to None to disable month filtering and load all data.
MONTH_FILTER: str | None = None

# Expected raw columns (no header in file)
COLS = ["date", "time", "open", "high", "low", "close", "volume"]

# Output directory and run tag
RUN_TAG = "pivot_retest_5min_same_direction"
RESULTS_DIR = os.path.join("results", RUN_TAG)

# Minimum distance filters (set to None to disable)
MIN_DISTANCE_TP1 = None
MIN_DISTANCE_TP2 = None


# ----------------------------------
# 2) LOAD & PREP HELPERS
# ----------------------------------
def detect_delimiter(path: str) -> str:
    """
    Auto-detect delimiter for a CSV-like file.
    Prefers csv.Sniffer; falls back to a simple count heuristic.
    """
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except Exception:
        semis = sample.count(";")
        commas = sample.count(",")
        return ";" if semis >= commas else ","


def parse_datetime_series(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    """
    Parse date+time robustly:
      1) %d/%m/%Y %H:%M
      2) %d/%m/%Y %H:%M:%S
      3) final fallback with dayfirst=True
    """
    dt_text = date_s.astype(str).str.strip() + " " + time_s.astype(str).str.strip()

    dt = pd.to_datetime(dt_text, format="%d/%m/%Y %H:%M", errors="coerce")
    missing = dt.isna()

    if missing.any():
        dt2 = pd.to_datetime(dt_text[missing], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        dt.loc[missing] = dt2
        missing = dt.isna()

    if missing.any():
        dt3 = pd.to_datetime(dt_text[missing], dayfirst=True, errors="coerce")
        dt.loc[missing] = dt3

    return dt


def load_month_subset(path: str, month_filter: str | None) -> pd.DataFrame:
    """
    Load CSV with auto delimiter, parse datetime, optional month filter.
    """
    sep = detect_delimiter(path)
    print(f"[INFO] Loading {path} with delimiter: '{sep}'")

    df = pd.read_csv(path, names=COLS, header=None, sep=sep, engine="python")

    df["datetime"] = parse_datetime_series(df["date"], df["time"])

    before = len(df)
    df = df.dropna(subset=["datetime"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped:,} rows with invalid datetime in {os.path.basename(path)}")

    if month_filter is not None:
        df = df[df["datetime"].dt.strftime("%Y-%m") == month_filter]

    return df


def coerce_ohlc_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLC columns are numeric."""
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def finalize_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Drop unused columns, remove bad OHLC rows, sort/index by datetime, de-duplicate timestamps.
    """
    df = df.drop(columns=["date", "time", "volume"], errors="ignore")

    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped:,} {name} rows with invalid OHLC")

    df = df.sort_values("datetime")
    df = df.set_index("datetime")
    df = df[~df.index.duplicated(keep="first")]
    return df


# Load data
h = load_month_subset(HOURLY_PATH, MONTH_FILTER)
m = load_month_subset(MINUTE_PATH, MONTH_FILTER)

print(f"Loaded {len(h):,} hourly rows and {len(m):,} 5-min rows (MONTH_FILTER={MONTH_FILTER})")

# Clean / prep
h = finalize_index(coerce_ohlc_numeric(h), "hourly")
m = finalize_index(coerce_ohlc_numeric(m), "5-min")

if h.empty:
    raise ValueError("Hourly dataset is empty after parsing/filtering.")
if m.empty:
    raise ValueError("5-minute dataset is empty after parsing/filtering.")


# ----------------------------------
# 3) STRONG CANDLE DETECTION (hourly)
# ----------------------------------
# Previous hour levels
h["prev_high"] = h["high"].shift(1)
h["prev_low"] = h["low"].shift(1)

# Candle polarity
h["is_bull"] = h["close"] > h["open"]
h["is_bear"] = h["close"] < h["open"]

# Previous candle polarity
h["prev_is_bull"] = h["is_bull"].shift(1)
h["prev_is_bear"] = h["is_bear"].shift(1)

# Strong definition + SAME-DIRECTION filter
h["strong_bull"] = (
    (h["close"] > h["prev_high"]) &
    h["is_bull"] &
    h["prev_is_bull"]
)

h["strong_bear"] = (
    (h["close"] < h["prev_low"]) &
    h["is_bear"] &
    h["prev_is_bear"]
)


def compute_levels(row: pd.Series) -> pd.Series:
    """
    For a strong bull:
      - pivot = prev_high
      - tp1   = open
      - tp2   = high

    For a strong bear:
      - pivot = prev_low
      - tp1   = open
      - tp2   = low
    """
    if bool(row.get("strong_bull", False)):
        return pd.Series(
            {"direction": "bull", "pivot": row["prev_high"], "tp1": row["open"], "tp2": row["high"]}
        )
    if bool(row.get("strong_bear", False)):
        return pd.Series(
            {"direction": "bear", "pivot": row["prev_low"], "tp1": row["open"], "tp2": row["low"]}
        )
    return pd.Series({"direction": np.nan, "pivot": np.nan, "tp1": np.nan, "tp2": np.nan})


strong = h[h["strong_bull"] | h["strong_bear"]].copy()

if strong.empty:
    print("No strong candles detected (with same-direction filter).")
    events_df = pd.DataFrame(
        columns=[
            "datetime", "direction", "pivot_hit_time",
            "tp1_hit", "tp2_hit", "tp1_within_2h", "tp2_within_2h",
            "distance_pivot_to_tp1", "distance_pivot_to_tp2",
        ]
    )
else:
    strong = strong.join(strong.apply(compute_levels, axis=1))

    strong["hour_start"] = strong.index
    strong["next_hour_start"] = strong.index + pd.Timedelta(hours=1)
    strong["next_hour_end"] = strong.index + pd.Timedelta(hours=2)
    strong["twoh_end"] = strong.index + pd.Timedelta(hours=3)
    strong["first20_end_next"] = strong["next_hour_start"] + pd.Timedelta(minutes=20)


# ----------------------------------
# 4) 5-MIN HELPERS
# ----------------------------------
def first_hit_time(level: float, df_min: pd.DataFrame) -> pd.Timestamp | None:
    """
    First timestamp in df_min where low <= level <= high.
    """
    if df_min.empty:
        return None
    touched = (df_min["low"] <= level) & (level <= df_min["high"])
    arr = touched.to_numpy()
    if not arr.any():
        return None
    return df_min.index[int(np.argmax(arr))]


def was_hit_in_slice(level: float, df_min: pd.DataFrame) -> bool:
    """
    True if any bar in df_min touches level (low <= level <= high).
    """
    if df_min.empty:
        return False
    return bool(((df_min["low"] <= level) & (level <= df_min["high"])).any())


# ----------------------------------
# 5) SCAN 5-MIN BARS PER STRONG HOUR
# ----------------------------------
events: list[dict] = []
pivot_touch_count = 0
eps = pd.Timedelta(microseconds=1)

if not strong.empty:
    total = len(strong)

    for i, (ts, row) in enumerate(strong.iterrows(), 1):
        if i % 50 == 0:
            print(f"Processed {i}/{total} strong candles...")

        direction = row["direction"]
        pivot = float(row["pivot"])
        tp1 = float(row["tp1"])
        tp2 = float(row["tp2"])

        next_hour_start = row["next_hour_start"]
        next_hour_end = row["next_hour_end"]
        twoh_end = row["twoh_end"]
        first20_end_next = row["first20_end_next"]

        # 5-min window: first 20 minutes of next hour
        m_first20 = m.loc[next_hour_start:first20_end_next - eps]

        # Condition A: pivot touched in first 20 minutes
        pivot_time = first_hit_time(pivot, m_first20)
        if pivot_time is None:
            continue

        # Condition B: tp2 NOT hit before pivot touch
        pivot_touch_count += 1

        m_before_pivot = m.loc[next_hour_start:pivot_time - eps]
        if was_hit_in_slice(tp2, m_before_pivot):
            continue

        # After pivot touch windows
        m_after_pivot_hour = m.loc[pivot_time:next_hour_end - eps]
        m_after_pivot_2h = m.loc[pivot_time:twoh_end - eps]

        tp1_hour = was_hit_in_slice(tp1, m_after_pivot_hour)
        tp2_hour = was_hit_in_slice(tp2, m_after_pivot_hour)
        tp1_2h = was_hit_in_slice(tp1, m_after_pivot_2h)
        tp2_2h = was_hit_in_slice(tp2, m_after_pivot_2h)

        dist_tp1 = float(abs(pivot - tp1))
        dist_tp2 = float(abs(pivot - tp2))

        if MIN_DISTANCE_TP1 is not None and dist_tp1 < MIN_DISTANCE_TP1:
            continue
        if MIN_DISTANCE_TP2 is not None and dist_tp2 < MIN_DISTANCE_TP2:
            continue

        events.append(
            {
                "datetime": ts,
                "direction": direction,
                "pivot_hit_time": pivot_time,
                "tp1_hit": bool(tp1_hour),
                "tp2_hit": bool(tp2_hour),
                "tp1_within_2h": bool(tp1_2h),
                "tp2_within_2h": bool(tp2_2h),
                "distance_pivot_to_tp1": dist_tp1,
                "distance_pivot_to_tp2": dist_tp2,
            }
        )

# Build events_df robustly even if no events were found
if events:
    events_df = pd.DataFrame(events).sort_values("datetime").reset_index(drop=True)
else:
    events_df = pd.DataFrame(
        columns=[
            "datetime", "direction", "pivot_hit_time",
            "tp1_hit", "tp2_hit", "tp1_within_2h", "tp2_within_2h",
            "distance_pivot_to_tp1", "distance_pivot_to_tp2",
        ]
    )


# ----------------------------------
# 6) SUMMARY (BY DIRECTION)
# ----------------------------------
def summarize(events_df_in: pd.DataFrame, direction: str) -> dict:
    df = events_df_in[events_df_in["direction"] == direction]
    n = len(df)

    if n == 0:
        return {
            "direction": direction,
            "setups": 0,
            "tp1_same_hour_pct": np.nan,
            "tp2_same_hour_pct": np.nan,
            "tp1_within_2h_pct": np.nan,
            "tp2_within_2h_pct": np.nan,
        }

    return {
        "direction": direction,
        "setups": int(n),
        "tp1_same_hour_pct": 100.0 * df["tp1_hit"].mean(),
        "tp2_same_hour_pct": 100.0 * df["tp2_hit"].mean(),
        "tp1_within_2h_pct": 100.0 * df["tp1_within_2h"].mean(),
        "tp2_within_2h_pct": 100.0 * df["tp2_within_2h"].mean(),
    }


summary_df = pd.DataFrame([summarize(events_df, "bull"), summarize(events_df, "bear")])
summary_df["strong_candles"] = len(strong)
summary_df["pivot_touch_first20"] = pivot_touch_count
summary_df["pivot_touch_rate_pct"] = (
    100.0 * pivot_touch_count / len(strong) if len(strong) > 0 else np.nan
)


# ----------------------------------
# 6b) HOUR-OF-DAY STATS
# ----------------------------------
if not events_df.empty:
    events_df["hour_of_day"] = pd.to_datetime(events_df["datetime"]).dt.hour

    hourly_stats = (
        events_df
        .groupby(["direction", "hour_of_day"])[["tp1_hit", "tp2_hit", "tp1_within_2h", "tp2_within_2h"]]
        .mean()
        .mul(100.0)
        .reset_index()
        .sort_values(["direction", "hour_of_day"])
    )
else:
    hourly_stats = pd.DataFrame(
        columns=["direction", "hour_of_day", "tp1_hit", "tp2_hit", "tp1_within_2h", "tp2_within_2h"]
    )


# ----------------------------------
# 7) OUTPUTS
# ----------------------------------
print(f"\n=== Strategy Results for {MONTH_FILTER} ===")
print(summary_df.to_string(index=False))

print("\n=== Hour-of-Day Stats (percentages) ===")
if hourly_stats.empty:
    print("No valid pivot-retest setups for this month.")
else:
    print(hourly_stats.to_string(index=False))

print("\n=== Event Log (first 10) ===")
print(events_df.head(10).to_string(index=False))

os.makedirs(RESULTS_DIR, exist_ok=True)
month_tag = str(MONTH_FILTER) if MONTH_FILTER is not None else "ALL"

events_path = os.path.join(RESULTS_DIR, f"events_{month_tag}.csv")
summary_path = os.path.join(RESULTS_DIR, f"summary_{month_tag}.csv")
hod_path = os.path.join(RESULTS_DIR, f"hourly_stats_{month_tag}.csv")

events_df.to_csv(events_path, index=False)
summary_df.to_csv(summary_path, index=False)
hourly_stats.to_csv(hod_path, index=False)

print("\nSaved:")
print(f"- {events_path}")
print(f"- {summary_path}")
print(f"- {hod_path}")
