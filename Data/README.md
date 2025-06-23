# Data Folder

This folder is intentionally left empty.

The original hedge fund return data from the HFR database is **not included** due to licensing restrictions.  
To reproduce the results, you may download the required data directly from [Hedge Fund Research](https://www.hfr.com). Once obtained, place the files in this directory.

## Included Templates

To facilitate replication, the following `.xlsx` templates are provided. These files contain:
- The exact **column headers** used in the project
- The full list of **index dates** matching the time series
- An indication that each index starts at a base value of **1000** when the data becomes available

These templates mirror the structure expected by the code:

- `data_template.xlsx` → corresponds to `HFRI_full.xlsx` used in `main.py`
- `bench_template.xlsx` → corresponds to `Benchmark.xlsx` used in `main.py`

> Note: These templates contain no actual return or index data — they are purely structural.

## 🔗 Additional Data

Complementary macroeconomic and benchmark data (e.g., Treasury yields, equity indices) was sourced from the [FRED database](https://fred.stlouisfed.org).
