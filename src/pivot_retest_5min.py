import os
import csv
import pandas as pd  # type: ignore
import numpy as np   # type: ignore

# ----------------------------------
# 1) CONFIG
# ----------------------------------
hourly_path = "data/nq-1h_bk_new.csv"   # semicolon file
minute_path = "data/nq-5m_bk.csv"       # comma file

MONTH_FILTER = None  # e.g. "2026-01" or None for all

cols = ["date", "time", "open", "high", "low", "close", "volume"]

# Output tag + directory
RUN_TAG = "pivot_retest_5min"
RESULTS_DIR = os.path.join("results", RUN_TAG)

# Minimum distance filters (set to None to disable)
MIN_DISTANCE_TP1 = None
MIN_DISTANCE_TP2 = None


# ----------------------------------
# 2) CSV + DATETIME HELPERS
# ----------------------------------
def detect_delimiter(path: str) -> str:
    """
    Auto-detect CSV delimiter from file sample.
    Supports ';' and ',' primarily.
    """
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(4096)

    # Try robust sniff first
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except Exception:
        # Fallback heuristic
        semis = sample.count(";")
        commas = sample.count(",")
        return ";" if semis >= commas else ","


def parse_datetime_series(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    """
    Parse DD/MM/YYYY + HH:MM (or HH:MM:SS) safely.
    First try strict/common formats, then fallback.
    """
    dt_text = date_s.astype(str).str.strip() + " " + time_s.astype(str).str.strip()

    # Try strict format first (fast + consistent)
    dt = pd.to_datetime(dt_text, format="%d/%m/%Y %H:%M", errors="coerce")
    missing = dt.isna()

    # Try with seconds
    if missing.any():
        dt2 = pd.to_datetime(dt_text[missing], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        dt.loc[missing] = dt2
        missing = dt.isna()

    # Final fallback (rare weird rows)
    if missing.any():
        dt3 = pd.to_datetime(dt_text[missing], dayfirst=True, errors="coerce")
        dt.loc[missing] = dt3

    return dt


def load_month_subset(path: str) -> pd.DataFrame:
    """
    Load CSV with auto delimiter, parse datetime, optional month filtering.
    """
    sep = detect_delimiter(path)
    print(f"[INFO] Loading {path} with delimiter: '{sep}'")

    df = pd.read_csv(path, names=cols, header=None, sep=sep, engine="python")

    # Build deterministic datetime
    df["datetime"] = parse_datetime_series(df["date"], df["time"])

    # Drop rows with bad datetime
    before = len(df)
    df = df.dropna(subset=["datetime"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped:,} rows with invalid datetime in {os.path.basename(path)}")

    if MONTH_FILTER is not None:
        df = df[df["datetime"].dt.strftime("%Y-%m") == MONTH_FILTER]

    return df


# ----------------------------------
# 3) LOAD DATA
# ----------------------------------
h = load_month_subset(hourly_path)
m = load_month_subset(minute_path)

print(f"Loaded {len(h):,} hourly rows and {len(m):,} 5-min rows (MONTH_FILTER={MONTH_FILTER})")

# Convert OHLC
for df in (h, m):
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Clean/sort/index
for name, df in (("hourly", h), ("5min", m)):
    df.drop(columns=["date", "time", "volume"], inplace=True, errors="ignore")
    before = len(df)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    if before - len(df) > 0:
        print(f"[WARN] Dropped {before - len(df):,} {name} rows with bad OHLC")
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)

# Remove duplicate timestamps
h = h[~h.index.duplicated(keep="first")]
m = m[~m.index.duplicated(keep="first")]

if h.empty:
    raise ValueError("Hourly dataset is empty after parsing/filtering.")
if m.empty:
    raise ValueError("5-minute dataset is empty after parsing/filtering.")

# -----------------------------
# 4) STRONG CLOSED CANDLES (hourly)
# -----------------------------
h["prev_high"] = h["high"].shift(1)
h["prev_low"] = h["low"].shift(1)

h["strong_bull"] = h["close"] > h["prev_high"]
h["strong_bear"] = h["close"] < h["prev_low"]


def _levels(row: pd.Series) -> pd.Series:
    if row["strong_bull"]:
        return pd.Series({
            "direction": "bull",
            "pivot": row["prev_high"],
            "tp1": row["open"],
            "tp2": row["high"],
        })
    elif row["strong_bear"]:
        return pd.Series({
            "direction": "bear",
            "pivot": row["prev_low"],
            "tp1": row["open"],
            "tp2": row["low"],
        })
    return pd.Series({"direction": np.nan, "pivot": np.nan, "tp1": np.nan, "tp2": np.nan})


strong = h[h["strong_bull"] | h["strong_bear"]].copy()

if strong.empty:
    print("No strong candles detected for current filter.")
else:
    strong = strong.join(strong.apply(_levels, axis=1))
    strong["hour_start"] = strong.index
    strong["next_hour_start"] = strong.index + pd.Timedelta(hours=1)
    strong["next_hour_end"] = strong.index + pd.Timedelta(hours=2)
    strong["twoh_end"] = strong.index + pd.Timedelta(hours=3)
    strong["first20_end_next"] = strong["next_hour_start"] + pd.Timedelta(minutes=20)

# -----------------------------
# 5) Helpers on 5-min data
# -----------------------------
def first_hit_time(level: float, df_5m: pd.DataFrame):
    if df_5m.empty:
        return None
    touched = (df_5m["low"] <= level) & (level <= df_5m["high"])
    arr = touched.to_numpy()
    if not arr.any():
        return None
    return df_5m.index[np.argmax(arr)]


def was_hit_in_slice(level: float, df_5m: pd.DataFrame) -> bool:
    if df_5m.empty:
        return False
    return bool(((df_5m["low"] <= level) & (level <= df_5m["high"])).any())

# -----------------------------
# 6) Scan events
# -----------------------------
events = []
pivot_touch_count = 0

if not strong.empty:
    total = len(strong)

    for i, (ts, row) in enumerate(strong.iterrows(), 1):
        if i % 100 == 0:
            print(f"Processed {i}/{total} strong candles...")

        direction = row["direction"]
        pivot = row["pivot"]
        tp1 = row["tp1"]
        tp2 = row["tp2"]
        next_hour_start = row["next_hour_start"]
        next_hour_end = row["next_hour_end"]
        twoh_end = row["twoh_end"]
        first20_end_next = row["first20_end_next"]

        eps = pd.Timedelta(microseconds=1)

        m_first20 = m.loc[next_hour_start:first20_end_next - eps]
        pivot_time = first_hit_time(pivot, m_first20)
        if pivot_time is None:
            continue

        pivot_touch_count += 1

        m_before_pivot = m.loc[next_hour_start:pivot_time - eps]
        if was_hit_in_slice(tp2, m_before_pivot):
            continue

        m_after_pivot_hour = m.loc[pivot_time:next_hour_end - eps]
        m_after_pivot_2h = m.loc[pivot_time:twoh_end - eps]

        dist_tp1 = float(abs(pivot - tp1))
        dist_tp2 = float(abs(pivot - tp2))

        if MIN_DISTANCE_TP1 is not None and dist_tp1 < MIN_DISTANCE_TP1:
            continue
        if MIN_DISTANCE_TP2 is not None and dist_tp2 < MIN_DISTANCE_TP2:
            continue

        events.append({
            "datetime": ts,
            "direction": direction,
            "pivot_hit_time": pivot_time,
            "tp1_hit": was_hit_in_slice(tp1, m_after_pivot_hour),
            "tp2_hit": was_hit_in_slice(tp2, m_after_pivot_hour),
            "tp1_within_2h": was_hit_in_slice(tp1, m_after_pivot_2h),
            "tp2_within_2h": was_hit_in_slice(tp2, m_after_pivot_2h),
            "distance_pivot_to_tp1": dist_tp1,
            "distance_pivot_to_tp2": dist_tp2,
        })

# Build events df
if events:
    events_df = pd.DataFrame(events).sort_values("datetime").reset_index(drop=True)
else:
    events_df = pd.DataFrame(columns=[
        "datetime", "direction", "pivot_hit_time",
        "tp1_hit", "tp2_hit", "tp1_within_2h", "tp2_within_2h",
        "distance_pivot_to_tp1", "distance_pivot_to_tp2"
    ])

# -----------------------------
# 7) Summary
# -----------------------------
def summarize(direction: str) -> dict:
    df = events_df[events_df["direction"] == direction]
    n = len(df)
    if n == 0:
        return {
            "direction": direction, "setups": 0,
            "tp1_same_hour_pct": np.nan, "tp2_same_hour_pct": np.nan,
            "tp1_within_2h_pct": np.nan, "tp2_within_2h_pct": np.nan,
        }
    return {
        "direction": direction,
        "setups": n,
        "tp1_same_hour_pct": 100.0 * df["tp1_hit"].mean(),
        "tp2_same_hour_pct": 100.0 * df["tp2_hit"].mean(),
        "tp1_within_2h_pct": 100.0 * df["tp1_within_2h"].mean(),
        "tp2_within_2h_pct": 100.0 * df["tp2_within_2h"].mean(),
    }

summary_df = pd.DataFrame([summarize("bull"), summarize("bear")])
summary_df["strong_candles"] = len(strong)
summary_df["pivot_touch_first20"] = pivot_touch_count
summary_df["pivot_touch_rate_pct"] = (
    100.0 * pivot_touch_count / len(strong) if len(strong) > 0 else np.nan
)

if not events_df.empty:
    events_df["hour_of_day"] = events_df["datetime"].dt.hour
    hourly_stats = (
        events_df
        .groupby(["direction", "hour_of_day"])[["tp1_hit", "tp2_hit", "tp1_within_2h", "tp2_within_2h"]]
        .mean()
        .mul(100.0)
        .reset_index()
        .sort_values(["direction", "hour_of_day"])
    )
else:
    hourly_stats = pd.DataFrame(columns=[
        "direction", "hour_of_day", "tp1_hit", "tp2_hit", "tp1_within_2h", "tp2_within_2h"
    ])

# -----------------------------
# 8) Output
# -----------------------------
print(f"\n=== Strategy Results for MONTH_FILTER={MONTH_FILTER} ===")
print(summary_df.to_string(index=False))

print("\n=== Hour-of-Day Stats (percentages) ===")
if hourly_stats.empty:
    print("No valid pivot-retest setups found.")
else:
    print(hourly_stats.to_string(index=False))

print("\n=== Event Log (first 10) ===")
print(events_df.head(10).to_string(index=False))

os.makedirs(RESULTS_DIR, exist_ok=True)
suffix = MONTH_FILTER if MONTH_FILTER is not None else "ALL"

events_path = os.path.join(RESULTS_DIR, f"events_{suffix}.csv")
summary_path = os.path.join(RESULTS_DIR, f"summary_{suffix}.csv")
hourly_stats_path = os.path.join(RESULTS_DIR, f"hourly_stats_{suffix}.csv")

events_df.to_csv(events_path, index=False)
summary_df.to_csv(summary_path, index=False)
hourly_stats.to_csv(hourly_stats_path, index=False)

print("\nSaved:")
print(f"- {events_path}")
print(f"- {summary_path}")
print(f"- {hourly_stats_path}")
