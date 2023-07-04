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


def run(input_folder, input_file, output_file, column_order, limit):
    df = pd.read_csv(input_file)
    patients = df["CSN"].tolist()

    with open(output_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        headers = column_order.split(",")
        writer.writerow(headers)

        i = 0
        for csn in patients:
            i += 1
            if limit is not None and i >= limit:
                break

            filename = f"{input_folder}/{str(csn)[-2:]}/{csn}.csv"
            try:
                j = 0
                with open(filename, 'r') as csvfile_read:
                    reader = csv.reader(csvfile_read)
                    for row in reader:
                        # Skip writing header line
                        if j >= 1:
                            # In some cases, there could be extra columns at the end which we will remove
                            writer.writerow(row[:len(headers)])
                        j += 1

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
    parser.add_argument('-co', '--column-order',
                        default="csn,recorded_time,measure,val",
                        help='The order of the columns as a comma separated list.')
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

    column_order = args.column_order

    limit = int(args.max_patients) if args.max_patients is not None else None

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_file={output_file}, column_order={column_order}, limit={limit}")
    print("-" * 30)

    run(input_dir, input_file, output_file, column_order, limit)

    print("DONE")
