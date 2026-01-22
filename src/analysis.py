import pandas as pd  # type: ignore
import numpy as np   # type: ignore

# ----------------------------------
# 1) CONFIG
# ----------------------------------
# Input CSV paths
hourly_path = "data/nq-1h_bk.csv"
minute_path = "data/nq-1m_bk.csv"

# Optional month filter in format "YYYY-MM" (e.g., "2023-11").
# Set to None to disable month filtering and load all data.
MONTH_FILTER = None

# Column names expected in the CSV files (semicolon-separated).
cols = ["date", "time", "open", "high", "low", "close", "volume"]

# ----------------------------------
# 2) LOAD & FILTER DATA
# ----------------------------------
"""
This method allows for the analysis of one or more months to limit execution time.

Set MONTH_FILTER to whatever month in YYYY-MM format to filter.
Set to None to disable month filter.
"""

def load_month_subset(path: str) -> pd.DataFrame:
	"""
	Load a CSV file into a DataFrame, build a datetime column, and optionally filter to a month.

	This function reads the raw data using the expected schema (cols), creates a unified
	`datetime` column from the date/time fields, and filters the dataset to a specific
	month if MONTH_FILTER is set.

	Args:
		path (str): Path to the CSV file to load.

	Returns:
		pd.DataFrame: A DataFrame containing:
			- OHLCV columns from the source
			- a parsed `datetime` column
			- only rows in MONTH_FILTER (if MONTH_FILTER is not None)
	"""
	# Read the CSV with explicit column names; data appears to be ";" delimited with no header.
	df = pd.read_csv(path, names=cols, header=None, sep=";")

	# Build a datetime column from date + time.
	# dayfirst=True assumes dates like DD/MM/YYYY. If your data is MM/DD/YYYY, change this.
	df["datetime"] = pd.to_datetime(
		df["date"].astype(str) + " " + df["time"].astype(str),
		dayfirst=True,
		errors="coerce",  # invalid parses become NaT
	)

	# If the month filter is enabled, keep only rows matching the given YYYY-MM.
	# Note: you could use "is not None" stylistically, but this keeps your original intent.
	if MONTH_FILTER != None:
		df = df[df["datetime"].dt.strftime("%Y-%m") == MONTH_FILTER]
		return df
	else:
		return df

# Load hourly and minute datasets (optionally month-filtered).
h = load_month_subset(hourly_path)
m = load_month_subset(minute_path)

print(f"Loaded {len(h):,} hourly rows and {len(m):,} minute rows and filtered for {MONTH_FILTER} month")

# Ensure OHLC columns are numeric (coerce invalid values to NaN).
for df in (h, m):
	for col in ["open", "high", "low", "close"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop unused columns, sort by datetime, and set datetime as the index
# so we can do fast time-slicing with df.loc[start:end].
for df in (h, m):
	df.drop(columns=["date", "time", "volume"], inplace=True)  # could add errors="ignore" if schema varies
	df.sort_values("datetime", inplace=True)
	df.set_index("datetime", inplace=True)

# Remove duplicate timestamps (keep the first occurrence).
h = h[~h.index.duplicated(keep="first")]
m = m[~m.index.duplicated(keep="first")]

# -----------------------------
# 3) STRONG CLOSED CANDLES (hourly)
# -----------------------------
# Compute previous hour's high/low to evaluate breakout closes.
h["prev_high"] = h["high"].shift(1)
h["prev_low"]  = h["low"].shift(1)

# "Strong" bull/bear definition:
# - strong_bull: hourly close breaks above previous hour high
# - strong_bear: hourly close breaks below previous hour low
h["strong_bull"] = h["close"] > h["prev_high"]
h["strong_bear"] = h["close"] < h["prev_low"]

def _levels(row: pd.Series) -> pd.Series:
    """
    Compute pivot and target levels for a strong candle row.

    For a strong bull:
      - pivot is previous high
      - tp1 is the candle open (pullback target)
      - tp2 is the candle high

    For a strong bear:
      - pivot is previous low
      - tp1 is the candle open
      - tp2 is the candle low

    Args:
        row (pd.Series): A row from the hourly DataFrame representing a single hour.

    Returns:
        pd.Series: A small series containing:
            - direction (str or NaN): "bull" / "bear"
            - pivot (float or NaN)
            - tp1 (float or NaN)
            - tp2 (float or NaN)
    """
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

# Extract only strong candles into a separate DataFrame.
strong = h[h["strong_bull"] | h["strong_bear"]].copy()

# If there are no strong candles, we can’t proceed with pivot-retest scanning.
if strong.empty:
    print("No strong candles detected in this month.")
else:
    # Add derived levels (direction, pivot, tp1, tp2) to each strong candle.
    strong = strong.join(strong.apply(_levels, axis=1))

    # Define time windows relative to the strong candle hour.
    strong["hour_start"]  = strong.index
    strong["hour_end"]    = strong.index + pd.Timedelta(hours=1)
    strong["twoh_end"]    = strong.index + pd.Timedelta(hours=2)
    strong["first20_end"] = strong.index + pd.Timedelta(minutes=20)

# -----------------------------
# 4) Minute helpers
# -----------------------------
def first_hit_time(level: float, df_min: pd.DataFrame):
    """
    Find the first timestamp in a minute slice where price touches a given level.

    A "touch" is defined as: low <= level <= high for that minute bar.

    Args:
        level (float): The price level we want to detect being touched.
        df_min (pd.DataFrame): Minute-bar slice (must contain 'low' and 'high').

    Returns:
        pd.Timestamp | None: The timestamp of the first bar that touches the level,
        or None if the slice is empty or never touches the level.
    """
    if df_min.empty:
        return None

    touched = (df_min["low"] <= level) & (level <= df_min["high"])
    arr = touched.to_numpy()

    if not arr.any():
        return None

    # np.argmax returns the index of the first True when arr contains at least one True.
    return df_min.index[np.argmax(arr)]

def was_hit_in_slice(level: float, df_min: pd.DataFrame) -> bool:
    """
    Determine whether a given price level is touched at least once in a minute slice.

    Args:
        level (float): The price level we want to detect being touched.
        df_min (pd.DataFrame): Minute-bar slice (must contain 'low' and 'high').

    Returns:
        bool: True if any bar touches the level, else False.
    """
    if df_min.empty:
        return False
    return ((df_min["low"] <= level) & (level <= df_min["high"])).any()

# -----------------------------
# 5) Scan minutes per strong hour
# -----------------------------
events = []

# NOTE: If strong is empty and you still want the script to continue safely,
# you should ensure strong is always defined. As written, this assumes strong exists.
total = len(strong)

for i, (ts, row) in enumerate(strong.iterrows(), 1):
    # Progress logging for large datasets
    if i % 50 == 0:
        print(f"Processed {i}/{total} strong candles...")

    # Extract strategy levels/time windows for this strong candle hour
    direction   = row["direction"]
    pivot       = row["pivot"]
    tp1         = row["tp1"]
    tp2         = row["tp2"]
    hour_start  = row["hour_start"]
    hour_end    = row["hour_end"]
    twoh_end    = row["twoh_end"]
    first20_end = row["first20_end"]

    # Use a tiny epsilon so that slicing behaves like a half-open interval:
    # [start, end) rather than [start, end] when using .loc time slicing.
    eps = pd.Timedelta(milliseconds=1)

    # Define minute windows:
    # - first 20 minutes of the hour
    # - full hour
    # - full 2-hour window
    m_first20 = m.loc[hour_start:first20_end - eps]
    m_hour    = m.loc[hour_start:hour_end - eps]
    m_2h      = m.loc[hour_start:twoh_end - eps]

    # Condition A: pivot must be touched within the first 20 minutes.
    pivot_time = first_hit_time(pivot, m_first20)
    if pivot_time is None:
        continue

    # Condition B: tp2 must NOT be hit before pivot touch within the hour.
    # (i.e., ensure the "retest" logic doesn't already run to tp2 prior to pivot.)
    m_before_pivot = m.loc[hour_start:pivot_time - eps]
    if was_hit_in_slice(tp2, m_before_pivot):
        continue

    # After pivot touch, measure whether tp1/tp2 get hit by:
    # - end of same hour
    # - end of next hour (2h window)
    m_after_pivot_hour = m.loc[pivot_time:hour_end - eps]
    m_after_pivot_2h   = m.loc[pivot_time:twoh_end - eps]

    tp1_hour = was_hit_in_slice(tp1, m_after_pivot_hour)
    tp2_hour = was_hit_in_slice(tp2, m_after_pivot_hour)
    tp1_2h   = was_hit_in_slice(tp1, m_after_pivot_2h)
    tp2_2h   = was_hit_in_slice(tp2, m_after_pivot_2h)

    # Record an event row for later analysis/summaries.
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

# Build events_df robustly even if no events were found.
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
# 6) Summary (by direction)
# -----------------------------
def summarize(direction: str) -> dict:
    """
    Compute hit-rate summary metrics for a given direction (bull/bear).

    Args:
        direction (str): Either "bull" or "bear".

    Returns:
        dict: A summary row containing:
            - direction (str)
            - setups (int): number of qualifying pivot-retest setups
            - tp1_same_hour_pct (float)
            - tp2_same_hour_pct (float)
            - tp1_within_2h_pct (float)
            - tp2_within_2h_pct (float)
    """
    df = events_df[events_df["direction"] == direction]
    n = len(df)

    if n == 0:
        return {
            "direction": direction, "setups": 0,
            "tp1_same_hour_pct": np.nan, "tp2_same_hour_pct": np.nan,
            "tp1_within_2h_pct": np.nan, "tp2_within_2h_pct": np.nan,
        }

    # Mean of boolean columns gives fraction True; multiply by 100 for percentage.
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
# 6b) Hour-of-day stats
# -----------------------------
# Group hit rates by the hour-of-day when the setup occurred.
if not events_df.empty:
    events_df["hour_of_day"] = events_df["datetime"].dt.hour

    hourly_stats = (
        events_df
        .groupby(["direction", "hour_of_day"])[
            ["tp1_hit", "tp2_hit", "tp1_within_2h", "tp2_within_2h"]
        ]
        .mean()
        .mul(100.0)
        .reset_index()
        .sort_values(["direction", "hour_of_day"])
    )
else:
    hourly_stats = pd.DataFrame(
        columns=["direction", "hour_of_day", "tp1_hit", "tp2_hit", "tp1_within_2h", "tp2_within_2h"]
    )

# -----------------------------
# 7) Outputs
# -----------------------------
print(f"\n=== Strategy Results for {MONTH_FILTER} ===")
print(summary_df.to_string(index=False))

print("\n=== Hour-of-Day Stats (percentages) ===")
if hourly_stats.empty:
    print("No valid pivot-retest setups for this month.")
else:
    print(hourly_stats.to_string(index=False))

print("\n=== Event Log (first 10) ===")
print(events_df.head(10).to_string(index=False))

# Persist outputs to CSV for downstream analysis.
events_df.to_csv(f"results/pivot_retest_events_{MONTH_FILTER}.csv", index=False)
summary_df.to_csv(f"results/pivot_retest_summary_{MONTH_FILTER}.csv", index=False)
hourly_stats.to_csv(f"results/pivot_retest_hourly_stats_{MONTH_FILTER}.csv", index=False)
