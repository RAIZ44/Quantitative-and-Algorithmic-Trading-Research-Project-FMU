## Stock Market Predictor (Pivot Retest Study)

This repo analyzes Nasdaq (NQ) price data to evaluate a "strong candle -> pivot retest" setup. It loads hourly and minute bars, finds strong breakouts, then checks whether the pivot is retested and whether targets (tp1/tp2) are hit within the same hour or within two hours.

### Features
- Parses semicolon-delimited OHLCV CSVs for hourly and minute data.
- Optional month filtering to speed up analysis.
- Identifies strong bull/bear candles using prior hour high/low.
- Scans minute data for pivot retests and target hits.
- Exports event logs and summary stats to CSV.

### Repository layout
- `src/analysis.py` - main analysis script.
- `data/` - place your input CSVs here (or update paths in the script).
- `results/` - output CSVs written by the script.
- `Notes.md` - project notes.

### Data requirements
The script expects two CSV files:
- Hourly bars: `nq-1h_bk.csv`
- Minute bars: `nq-1m_bk.csv`

Each file must be semicolon-delimited with columns in this order (no header):

```
date;time;open;high;low;close;volume
```

Date parsing uses `dayfirst=True` (DD/MM/YYYY). If your data is MM/DD/YYYY, update the parsing line in `src/analysis.py`.

### Setup
1. Create a virtual environment (optional but recommended).
2. Install dependencies:

```
pip install pandas numpy
```

3. Place the input CSVs in the repo root or update these paths in `src/analysis.py`:

```
hourly_path = "nq-1h_bk.csv"
minute_path = "nq-1m_bk.csv"
```

### Usage
From the repo root:

```
python src/analysis.py
```

To analyze a specific month, set this in `src/analysis.py`:

```
MONTH_FILTER = "2023-11"
```

### Outputs
The script writes CSVs to `results/`:
- `pivot_retest_events_<MONTH>.csv` - event log for each setup.
- `pivot_retest_summary_<MONTH>.csv` - summary hit rates by direction.
- `pivot_retest_hourly_stats_<MONTH>.csv` - hit rates by hour of day.

If `MONTH_FILTER = None`, `<MONTH>` will be `None` in the filenames.

### Notes
- If no strong candles are found for the month, the script prints a warning and outputs empty summary files.
- To change the strategy logic (pivot/targets), edit the `_levels` function in `src/analysis.py`.

