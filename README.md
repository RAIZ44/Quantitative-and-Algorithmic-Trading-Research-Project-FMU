# Quantitative and Algorithmic Trading Research Project (NQ Pivot Retest)

## Project objective

This project tests a specific NQ futures setup:

1. Detect a **strong hourly close**.
2. Wait for a **pivot retest** in the first 20 minutes of the next hour.
3. Measure whether TP levels are reached in the same hour and within 2 hours.

The repository includes 1-minute and 5-minute variants, plus an external Databento verification script.

## Strategy definition

For each hourly candle `t`, compare against candle `t-1`:

- **Strong bull candle**: `close_t > high_(t-1)`
- **Strong bear candle**: `close_t < low_(t-1)`

Level mapping used by the strategy:

| Direction |              Pivot |               TP1 |               TP2 |
| --------- | -----------------: | ----------------: | ----------------: |
| Bull      | Previous hour high | Current hour open | Current hour high |
| Bear      |  Previous hour low | Current hour open |  Current hour low |

## Event qualification logic (implemented)

For each strong candle:

1. Define windows:
   - `next_hour_start = t + 1h`
   - `first20_end = next_hour_start + 20m`
   - `next_hour_end = t + 2h`
   - `twoh_end = t + 3h`
2. Find the first pivot touch inside `[next_hour_start, first20_end)`.
3. Reject the setup if TP2 is touched before that pivot touch.
4. If valid, compute outcomes:
   - `tp1_hit`: TP1 touched from pivot time to `next_hour_end`
   - `tp2_hit`: TP2 touched from pivot time to `next_hour_end`
   - `tp1_within_2h`: TP1 touched from pivot time to `twoh_end`
   - `tp2_within_2h`: TP2 touched from pivot time to `twoh_end`
5. Save each event row to `events_*.csv`, aggregate summaries to `summary_*.csv` and `hourly_stats_*.csv`.

The 5-minute implementation is in `src/pivot_retest_5min.py`.

## Data and time assumptions

- Input CSV format: `date,time,open,high,low,close,volume` (headerless).
- Date parsing is `DD/MM/YYYY` with robust fallback parsing.
- Delimiter auto-detection supports `;` and `,`.
- Verification defaults to `America/Chicago` timezone (CME session clock).

## Results snapshot (5-minute variant, ALL data)

Source: `results/pivot_retest_5min/summary_ALL.csv`

| Metric                     |  Value |
| -------------------------- | -----: |
| Strong candles scanned     | 47,668 |
| Pivot touched in first 20m | 18,571 |
| Pivot-touch rate           | 38.96% |
| Valid bull setups          |  8,590 |
| Valid bear setups          |  7,475 |

| Direction | Setups | TP1 same hour | TP2 same hour | TP1 within 2h | TP2 within 2h |
| --------- | -----: | ------------: | ------------: | ------------: | ------------: |
| Bull      |  8,590 |        64.76% |        50.86% |        73.91% |        63.47% |
| Bear      |  7,475 |        64.58% |        47.30% |        74.54% |        58.58% |

## Bearish strong closed candles by hour (reference-style table)

This table is formatted as a compact, hour-by-hour readout similar to your reference screenshot.
Source: `results/pivot_retest_5min/hourly_stats_ALL.csv` + setup counts from `results/pivot_retest_5min/events_ALL.csv`.

| Hour (CT) | Total setups | TP1 same hour | TP2 same hour | TP1 within 2h | TP2 within 2h |
| --------: | -----------: | ------------: | ------------: | ------------: | ------------: |
|         0 |          395 |         72.9% |         50.1% |         87.1% |         70.1% |
|         1 |          477 |         78.6% |         63.9% |         85.7% |         71.9% |
|         2 |          436 |         64.4% |         35.8% |         72.7% |         44.7% |
|         3 |          320 |         55.3% |         37.2% |         64.7% |         52.2% |
|         4 |          270 |         58.5% |         45.2% |         72.2% |         55.9% |
|         5 |          336 |         67.3% |         49.1% |         77.7% |         65.5% |
|         6 |          354 |         67.2% |         58.2% |         86.2% |         80.2% |
|         7 |          398 |         82.2% |         69.6% |         87.7% |         77.4% |
|         8 |          500 |         77.4% |         46.2% |         82.4% |         53.2% |
|         9 |          306 |         54.2% |         42.5% |         63.1% |         53.3% |
|        10 |          209 |         52.6% |         45.9% |         65.1% |         55.0% |
|        11 |          244 |         60.7% |         43.9% |         73.0% |         58.2% |
|        12 |          293 |         68.6% |         43.3% |         79.2% |         53.9% |
|        13 |          329 |         68.4% |         53.8% |         73.3% |         56.8% |
|        14 |          331 |         39.6% |         36.6% |         40.5% |         37.5% |
|        15 |           51 |         23.5% |         11.8% |         35.3% |         45.1% |
|        16 |           48 |         70.8% |         54.2% |         79.2% |         58.3% |
|        17 |          316 |         63.6% |         39.6% |         73.1% |         53.8% |
|        18 |          286 |         70.3% |         50.7% |         80.4% |         60.5% |
|        19 |          361 |         64.3% |         41.6% |         72.9% |         51.2% |
|        20 |          307 |         51.1% |         45.6% |         63.2% |         53.4% |
|        21 |          284 |         59.2% |         40.1% |         72.5% |         54.2% |
|        22 |          293 |         60.4% |         43.3% |         74.4% |         57.3% |
|        23 |          331 |         62.5% |         50.2% |         78.9% |         64.7% |

## External validation (Databento)

Validation script: `src/verify_events_databento.py`

- Replays event outcomes on independent Databento data (`GLBX.MDP3`).
- Compares expected vs observed values for each event row.
- Writes:
  - `results/pivot_retest_5min/verification_databento.csv`
  - `results/pivot_retest_5min/verification_missing_prev_hourly_bar.csv`

Latest January verification result in this repo:

| Period  | Rows checked | `ok` rows | Outcome matches (`all_outcome_match`) | Missing prev-hour rows |
| ------- | -----------: | --------: | ------------------------------------: | ---------------------: |
| 2026-01 |           51 |        47 |                               47 / 47 |                      4 |

## Repository map

- `src/pivot_retest_5min.py`: core 5m strategy backtest.
- `src/pivot_retest_1m.py`: 1m variant.
- `src/pivot_retest_5min_same_direction.py`: 5m same-direction variant.
- `src/pivot_retest_1m_same_direction.py`: 1m same-direction variant.
- `src/verify_events_databento.py`: independent verification against Databento.
- `results/`: generated event logs, summaries, hourly statistics, verification reports.
