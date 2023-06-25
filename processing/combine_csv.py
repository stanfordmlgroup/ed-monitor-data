#!/usr/bin/env python

"""
Script to combine patient numerics CSV files

Usage:
```
python -u combine_csv.py --input-dir csv-data --input-file csv_summary.2020_08_23_2023_05_31.csv --output-file csv.2020_08_23_2023_05_31.csv --max-patients 3
```

"""

import argparse
import csv
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def run(input_folder, input_file, output_file, limit):
    df = pd.read_csv(input_file)
    patients = df["CSN"].tolist()

    with open(output_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["csn", "recorded_time", "measure", "val"])

        i = 0
        for csn in patients:
            i += 1
            if limit is not None and i >= limit:
                break

            filename = f"{input_folder}/{str(csn)[-2:]}/{csn}.csv"
            try:
                df_csv = pd.read_csv(filename)
                for i, row in df_csv.iterrows():
                    writer.writerow(row.tolist())
                print(f"[{i}/{len(patients)}] Completed")
            except Exception as e:
                print(f"[ERROR] for patient {csn} due to {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-dir',
                        required=True,
                        help='Where the data files are located.')
    parser.add_argument('-f', '--input-file',
                        required=True,
                        help='Where the summary is located.')
    parser.add_argument('-of', '--output-file',
                        required=True,
                        help='Where the output summary file is located.')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')
    args = parser.parse_args()

    # Where the data files are located
    input_dir = args.input_dir

    # Where the summary is located
    input_file = args.input_file

    # Where the output summary file is located
    output_file = args.output_file

    limit = int(args.max_patients) if args.max_patients is not None else None

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_file={output_file}, limit={limit}")
    print("-" * 30)

    run(input_dir, input_file, output_file, limit)

    print("DONE")
