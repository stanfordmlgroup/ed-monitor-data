#!/usr/bin/env python

"""
Script to generate patient numerics

Usage:
```
python -u /deep/u/tomjin/ed-monitor-data/processing/generate_numerics_dataset.py --input-dir /deep2/group/ed-monitor/processed/2020_08_01-2022_09_27/patient-data --input-file /deep/group/ed-monitor/processed/2020_08_01-2022_09_27/consolidated.csv --output-folder /deep/group/ed-monitor-self-supervised/v5/ --max-patients 3
```

"""

import argparse
import csv
import warnings
from concurrent import futures
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

NUMERIC_COLUMNS = ['HR', 'RR', 'SpO2', 'btbRRInt_ms', 'NBPs', 'NBPd', 'Perf']
VERBOSE = False


def process_patient(input_args):
    i, tot, csn, input_folder, output_folder = input_args

    filename = f"{input_folder}/{csn}/{csn}.h5"
    print(f"[{i}/{tot}] Working on patient {csn} at {filename}")
    try:
        with h5py.File(filename, "r") as f:
            # Folders are kept sane by outputting objects into subfolders based on last two digits of CSN
            folder_hash = str(csn)[-2:]

            Path(f"{output_folder}/{folder_hash}/{csn}").mkdir(parents=True, exist_ok=True)
            for c in NUMERIC_COLUMNS:
                with open(f"{output_folder}/{folder_hash}/{csn}/{c}.csv", "w") as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(["recorded_time", c])
                    vals = np.array(f["numerics"][c])
                    times = np.array(f["numerics"][f"{c}-time"])
                    for i in range(len(vals)):
                        if not np.isnan(vals[i]):
                            writer.writerow([times[i], vals[i]])
        return True
    except Exception as e:
        print(f"[ERROR] for patient {csn} due to {e}")
        return False
        # raise e


def run(input_folder, input_file, output_folder, limit):
    df = pd.read_csv(input_file)
    patients = df["patient_id"].tolist()

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
        result = future.result(timeout=60 * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-dir',
                        required=True,
                        help='Where the data files are located')
    parser.add_argument('-f', '--input-file',
                        required=True,
                        help='Where the summary is located')
    parser.add_argument('-od', '--output-folder',
                        required=True,
                        help='Where the output data folder is located')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')
    args = parser.parse_args()

    # Where the data files are located
    input_dir = args.input_dir

    # Where the summary is located
    input_file = args.input_file

    # Where the output data file is located
    output_folder = args.output_folder

    limit = int(args.max_patients) if args.max_patients is not None else None

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_folder={output_folder}")
    print("-" * 30)

    run(input_dir, input_file, output_folder, limit)

    print("DONE")
