import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_timespan(file: pq.ParquetFile):
    first_rg = file.metadata.row_group(0)
    first_col_stats = first_rg.column(0)

    time_start = first_col_stats.statistics.min

    last_rg = file.metadata.row_group(file.num_row_groups - 1)
    last_col_stats = last_rg.column(0)

    time_end = last_col_stats.statistics.max

    return (int(time_start), int(time_end))


def get_batches(start_time: int, end_time: int):
    start_dt = pd.to_datetime(start_time, unit="ms")
    end_dt = pd.to_datetime(end_time, unit="ms")

    all_weeks = pd.date_range(
        start=start_dt.floor("D") - pd.Timedelta(days=start_dt.weekday()),
        end=end_dt,
        freq="W-MON",
    )

    week_starts = all_weeks
    week_ends = week_starts + pd.Timedelta(days=7) - pd.Timedelta(milliseconds=1)

    range_starts = np.maximum(start_dt, week_starts)
    range_ends = np.minimum(end_dt, week_ends)

    mask = range_starts <= range_ends

    start_index = range_starts[mask]
    end_index = range_ends[mask]

    start_ms = 1000 * (start_index.astype(np.int64) // 10**6)
    end_ms = 1000 * (end_index.astype(np.int64) // 10**6)

    return list(zip(start_ms, end_ms))


def is_stationary(series, alpha=0.01):
    result = adfuller(series, regression="c")
    return result[1] < alpha


def get_ar_matrix(Y, p=15):
    n = len(Y)
    Y_restricted = np.empty((n - p, p))
    for m in range(1, p + 1):
        Y_restricted[:, m - 1] = Y[p - m : n - m]

    return Y_restricted, Y[p:]


def get_ardl_matrix(Y, X, p=15):
    n = len(Y)
    design_matrix = np.empty((n - p, 2 * p))

    for i in range(1, p + 1):
        design_matrix[:, i - 1] = Y[p - i : n - i]

    for i in range(1, p + 1):
        design_matrix[:, p + i - 1] = X[p - i : n - i]

    return design_matrix, Y[p:]


def granger(time_series1, time_series2):
    Y_restricted, Y_restricted_target = get_ar_matrix(time_series2)
    X_full, Y_full_target = get_ardl_matrix(time_series2, time_series1)

    _, restricted_model_residuals, _, _ = np.linalg.lstsq(
        Y_restricted, Y_restricted_target, rcond=None
    )
    _, full_model_residuals, _, _ = np.linalg.lstsq(X_full, Y_full_target, rcond=None)

    if restricted_model_residuals.size == 0 or full_model_residuals.size == 0:
        return

    return np.log(
        max(restricted_model_residuals[0], 1e-15) / max(full_model_residuals[0], 1e-15)
    )


def main():
    print("Loading Parquet file into memory...")
    table = pq.read_table("vwap_returns_2020.parquet")
    column_names = [name for name in table.column_names if name != "index"]

    index_arr = table.column("index").to_numpy()

    symbol_data_dict = {}
    start_indices = {}

    print("Pre-processing symbols and finding valid time ranges...")
    for sym in column_names:
        sym_arr = table.column(sym).to_numpy()
        symbol_data_dict[sym] = sym_arr

        valid_mask = ~np.isnan(sym_arr)
        if not np.any(valid_mask):
            continue

        valid_positions = np.where(valid_mask)[0]
        start_time = index_arr[valid_positions[0]]
        end_time = index_arr[valid_positions[-1]]

        start_indices[sym] = (start_time, end_time)
    print("Starting stationary checks...")

    for symbol, (start, end) in start_indices.items():
        weeks = get_batches(start, end)

        sym_arr = symbol_data_dict[symbol]

        for week_start, week_end in weeks:
            mask = (index_arr >= week_start) & (index_arr <= week_end)

            weekly_data = sym_arr[mask]

            if len(weekly_data) == 0:
                continue

            stationary = is_stationary(weekly_data)

            if not stationary:
                print(f"{symbol} skipping {week_start}-{week_end}")
            else:
                print(f"[{symbol}] {week_start}-{week_end} pass")


if __name__ == "__main__":
    main()
