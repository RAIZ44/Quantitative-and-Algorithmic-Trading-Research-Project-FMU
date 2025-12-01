# Strong Candle Pivot-Retest Backtest (NQ Futures)

This project implements a research/backtesting script for a **pivot-retest strategy** on Nasdaq futures (NQ), using both **hourly** and **minute** candlestick data.

The script:

- Loads a specific **month** of NQ data from separate hourly and minute CSV files.
- Detects **strong closed candles** (hourly candles that close above the previous high or below the previous low).
- Defines **pivot**, **TP1**, and **TP2** levels based on those strong candles.
- Scans minute data to see:
  - If/when price **retests the pivot** within the first 20 minutes of the hour.
  - Whether **TP1 / TP2** are hit within:
    - The **same hour**.
    - A **2-hour window**.
- Produces summary stats (by direction and by hour-of-day) and a detailed event log as CSVs.

---

## Project Structure

A typical folder layout:

```text
Stock Market Predictor/
├─ code/
│  └─ analysis.py          # Main backtest script (this file)
├─ data/
│  ├─ nq-1h_bk.csv         # Hourly NQ candles
│  └─ nq-1m_bk.csv         # 1-minute NQ candles
└─ results/
   ├─ pivot_retest_events_YYYY-MM.csv
   ├─ pivot_retest_summary_YYYY-MM.csv
   └─ pivot_retest_hourly_stats_YYYY-MM.csv
