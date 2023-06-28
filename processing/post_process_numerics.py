#!/usr/bin/env python

"""
Script to read in the numerics CSV data produced by the convert_to_csv.py script and produce the same
files but averaged per minute. Specifically:

Produce 1 minute means (for the previous minute) for each numeric, except for btbRR, for
which we want the SD for the last 1min and 5min (i.e., rolling 1min and 5min HRV)

Usage:
```
python -u post_process_numerics.py --input-dir csv-data --input-file csv_summary.2020_08_23_2023_05_31.csv --output-folder csv-processed-data --max-patients 3
```

"""

import argparse
import warnings
from concurrent import futures
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

MEAN_NUMERIC_COLUMNS = ['HR', 'RR', 'SpO2', 'NBPs', 'NBPd', 'Perf']
VERBOSE = False


def std(x):
    return np.std(x)


def generate_grouped_df(df, col, output_col, period_min=1, stat="mean"):
    df_col = df[df["measure"] == col]
    df_col = df_col.resample(f"{period_min}min", origin='epoch').agg({"csn": "max", "val": stat})

    # Remove any NaN after grouping by the period
    df_col = df_col[~df_col["val"].isna()]

    # Add one minute so that the times represent the mean of the 1 min before this time
    df_col.index = df_col.index + pd.Timedelta(minutes=period_min)

    df_col["measure"] = output_col

    return df_col


def process_patient(input_args):
    i, tot, csn, input_folder, output_folder = input_args

    filename = f"{input_folder}/{str(csn)[-2:]}/{csn}.csv"
    print(f"[{i}/{tot}] Working on patient {csn} at {filename}")
    try:
        df = pd.read_csv(filename, dtype={'csn': str})
        df.index = pd.to_datetime(df["recorded_time"])

        dfs = []

        for c in MEAN_NUMERIC_COLUMNS:
            df_col = generate_grouped_df(df, c, c, period_min=1, stat="mean")
            dfs.append(df_col)

        df_col = generate_grouped_df(df, "btbRRInt_ms", "1min_HRV", period_min=1, stat=std)
        dfs.append(df_col)
        df_col = generate_grouped_df(df, "btbRRInt_ms", "5min_HRV", period_min=5, stat=std)
        dfs.append(df_col)

        df_out = pd.concat(dfs, axis=0)

        output_folder_path = f"{output_folder}/{str(csn)[-2:]}"
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        output_filename = f"{output_folder_path}/{csn}.csv"
        df_out.to_csv(output_filename)

    except Exception as e:
        print(f"[ERROR] for patient {csn} due to {e}")


def run(input_folder, input_file, output_folder, limit):
    df = pd.read_csv(input_file)
    patients = df["CSN"].tolist()

    fs = []
    with futures.ThreadPoolExecutor(32) as executor:
        i = 0
        for csn in tqdm(patients, disable=True):
            i += 1
            if limit is not None and i > limit:
                break

            input_args = [i, df.shape[0], csn, input_folder, output_folder]
            future = executor.submit(process_patient, input_args)
            fs.append(future)

    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        future.result(timeout=60 * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-dir',
                        required=True,
                        help='Where the data files are located.')
    parser.add_argument('-f', '--input-file',
                        required=True,
                        help='Where the summary is located.')
    parser.add_argument('-od', '--output-folder',
                        required=True,
                        help='Where the output data folder is located.')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')
    args = parser.parse_args()

    # Where the data files are located
    input_dir = args.input_dir

    # Where the summary is located
    input_file = args.input_file

    # Where the output data folder is located
    output_folder = args.output_folder

    limit = int(args.max_patients) if args.max_patients is not None else None

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_folder={output_folder}")
    print("-" * 30)

    run(input_dir, input_file, output_folder, limit)

    print("DONE")
