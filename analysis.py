import pandas as pd # type: ignore
import numpy as np # type: ignore
 
# -----------------------------
# 1) CONFIG
# -----------------------------
hourly_path = "C:\\Users\\Owner\\Stock Market Predictor\\data\\nq-1h_bk.csv"
minute_path = "C:\\Users\\Owner\\Stock Market Predictor\\data\\nq-1m_bk.csv"

# Change this to whichever month you want (YYYY-MM)
MONTH_FILTER = "2025-05"

cols = ["date", "time", "open", "high", "low", "close", "volume"]

# -----------------------------
# 2) LOAD & FILTER DATA
# -----------------------------
def load_month_subset(path):
    """Load only rows matching MONTH_FILTER (YYYY-MM)."""
    df = pd.read_csv(path, names=cols, header=None, sep=";")
    # Combine early to allow filtering by month string
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), dayfirst=True, errors="coerce")
    # Filter to one month before doing any numeric conversion
    df = df[df["datetime"].dt.strftime("%Y-%m") == MONTH_FILTER]
    return df

h = load_month_subset(hourly_path)
m = load_month_subset(minute_path)

print(f"Loaded {len(h):,} hourly rows and {len(m):,} minute rows for {MONTH_FILTER}")

# Convert numeric columns
for df in (h, m):
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns, sort, index by datetime, and remove duplicates."""
    df = df.copy()
    df.drop(columns=["date", "time", "volume"], inplace=True, errors="ignore")
    df.sort_values("datetime", inplace=True)
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    return df

h = _prepare(h)
m = _prepare(m)

# -----------------------------
# 3) STRONG CLOSED CANDLES (hourly)
# -----------------------------
h["prev_high"] = h["high"].shift(1)
h["prev_low"]  = h["low"].shift(1)
h["strong_bull"] = h["close"] > h["prev_high"]
h["strong_bear"] = h["close"] < h["prev_low"]

def _levels(row):
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
    else:
        return pd.Series({"direction": np.nan, "pivot": np.nan, "tp1": np.nan, "tp2": np.nan})

strong = h[h["strong_bull"] | h["strong_bear"]].copy()
if strong.empty:
    print("No strong candles detected in this month.")
else:
    strong = strong.join(strong.apply(_levels, axis=1))
    strong["hour_start"]  = strong.index
    strong["hour_end"]    = strong.index + pd.Timedelta(hours=1)
    strong["twoh_end"]    = strong.index + pd.Timedelta(hours=2)
    strong["first20_end"] = strong.index + pd.Timedelta(minutes=20)

# -----------------------------
# 4) Minute helpers
# -----------------------------
def first_hit_time(level: float, df_min: pd.DataFrame):
    if df_min.empty:
        return None
    touched = (df_min["low"] <= level) & (level <= df_min["high"])
    arr = touched.to_numpy()
    if not arr.any():
        return None
    return df_min.index[np.argmax(arr)]

def was_hit_in_slice(level: float, df_min: pd.DataFrame) -> bool:
    if df_min.empty:
        return False
    return ((df_min["low"] <= level) & (level <= df_min["high"])).any()

# -----------------------------
# 5) Scan minutes per strong hour
# -----------------------------
events = []
total = len(strong)

for i, (ts, row) in enumerate(strong.iterrows(), 1):
    if i % 50 == 0:
        print(f"Processed {i}/{total} strong candles...")

    direction   = row["direction"]
    pivot       = row["pivot"]
    tp1         = row["tp1"]
    tp2         = row["tp2"]
    hour_start  = row["hour_start"]
    hour_end    = row["hour_end"]
    twoh_end    = row["twoh_end"]
    first20_end = row["first20_end"]

    eps = pd.Timedelta(milliseconds=1)
    m_first20 = m.loc[hour_start:first20_end - eps]
    m_hour    = m.loc[hour_start:hour_end - eps]
    m_2h      = m.loc[hour_start:twoh_end - eps]

    pivot_time = first_hit_time(pivot, m_first20)
    if pivot_time is None:
        continue

    m_before_pivot = m.loc[hour_start:pivot_time - eps]
    if was_hit_in_slice(tp2, m_before_pivot):
        continue

    m_after_pivot_hour = m.loc[pivot_time:hour_end - eps]
    m_after_pivot_2h   = m.loc[pivot_time:twoh_end - eps]

    tp1_hour = was_hit_in_slice(tp1, m_after_pivot_hour)
    tp2_hour = was_hit_in_slice(tp2, m_after_pivot_hour)
    tp1_2h   = was_hit_in_slice(tp1, m_after_pivot_2h)
    tp2_2h   = was_hit_in_slice(tp2, m_after_pivot_2h)

    events.append({
        "datetime": ts,
        "direction": direction,
        "pivot_hit_time": pivot_time,
        "tp1_hit": bool(tp1_hour),
        "tp2_hit": bool(tp2_hour),
        "tp1_within_2h": bool(tp1_2h),
        "tp2_within_2h": bool(tp2_2h),
        "distance_pivot_to_tp1": float(abs(pivot - tp1)),
        "distance_pivot_to_tp2": float(abs(pivot - tp2)),
    })

if events:
    events_df = pd.DataFrame(events).sort_values("datetime").reset_index(drop=True)
else:
    events_df = pd.DataFrame(
        columns=[
            "datetime",
            "direction",
            "pivot_hit_time",
            "tp1_hit",
            "tp2_hit",
            "tp1_within_2h",
            "tp2_within_2h",
            "distance_pivot_to_tp1",
            "distance_pivot_to_tp2",
        ]
    )

# -----------------------------
# 6) Summary
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

# -----------------------------
# 7) Outputs
# -----------------------------
print(f"\n=== Strategy Results for {MONTH_FILTER} ===")
print(summary_df.to_string(index=False))

print("\n=== Event Log (first 10) ===")
print(events_df.head(10).to_string(index=False))

# print("\n=== Event Log ===")
# print(events_df.to_string(index=False))

events_df.to_csv(f"pivot_retest_events_{MONTH_FILTER}.csv", index=False)
summary_df.to_csv(f"pivot_retest_summary_{MONTH_FILTER}.csv", index=False)
