#!/usr/bin/env python

"""
Script to generate patient numerics

Usage:
```
python -u convert_to_csv.py --input-dir /deep2/group/ed-monitor/processed/2020_08_01-2022_09_27/patient-data --input-file /deep/group/ed-monitor/processed/2020_08_01-2022_09_27/consolidated.csv --output-folder /deep/group/ed-monitor-self-supervised/v5/ --max-patients 3
```

"""

import argparse
import csv
import warnings
from concurrent import futures
from pathlib import Path

import os
import shutil
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import boto3
import botocore

warnings.filterwarnings("ignore")

NUMERIC_COLUMNS = ['HR', 'RR', 'SpO2', 'btbRRInt_ms', 'NBPs', 'NBPd', 'Perf']
VERBOSE = False


def parse_s3_uri(s3_uri):
    # Remove the "s3://" prefix
    uri_without_prefix = s3_uri[5:]

    # Split the URI into bucket and key
    bucket_end_index = uri_without_prefix.find('/')
    bucket_name = uri_without_prefix[:bucket_end_index]
    key = uri_without_prefix[bucket_end_index + 1:]

    return bucket_name, key


def download_s3_file(s3_uri, local_path):
    s3_client = boto3.client('s3')

    # Parse the S3 URI
    try:
        bucket_name, key = parse_s3_uri(s3_uri)
    except ValueError as e:
        print(f"Invalid S3 URI: {e}")
        return

    # Download the file
    try:
        s3_client.download_file(bucket_name, key, local_path)
        print(f"File downloaded successfully to: {local_path}")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            print(f"An error occurred while downloading the file: {e}")


def process_patient(input_args):
    i, tot, csn, input_folder, output_folder = input_args

    filename = f"{input_folder}/{str(csn)[-2:]}/{csn}.h5"
    print(f"[{i}/{tot}] Working on patient {csn} at {filename}")
    try:
        availability = [csn]

        if filename.startswith("s3"):
            tmp_file = f"/tmp/{csn}.h5"
            download_s3_file(filename, tmp_file)
            filename = tmp_file

        with h5py.File(filename, "r") as f:
            # Folders are kept sane by outputting objects into subfolders based on last two digits of CSN
            folder_hash = str(csn)[-2:]
            full_output_folder = f"{output_folder}/{folder_hash}/{csn}"
            Path(full_output_folder).mkdir(parents=True, exist_ok=True)

            for c in NUMERIC_COLUMNS:
                with open(f"{full_output_folder}/{c}.csv", "w") as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(["recorded_time", c])
                    if c in f["numerics"]:
                        vals = np.array(f["numerics"][c])
                        times = np.array(f["numerics"][f"{c}-time"])
                        for i in range(len(vals)):
                            if not np.isnan(vals[i]):
                                writer.writerow([times[i], vals[i]])
                        availability.append(1)
                    else:
                        availability.append(0)

        if output_folder.startswith("s3"):
            os.remove(filename)

        return availability
    except Exception as e:
        print(f"[ERROR] for patient {csn} due to {e}")
        return None
        # raise e


def run(input_folder, input_file, output_folder, output_file, limit):
    if input_file.startswith("s3"):
        tmp_file = "/tmp/input.csv"
        download_s3_file(input_file, tmp_file)
        input_file = tmp_file

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

    with open(f"{output_file}", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        header = ["CSN"]
        for c in NUMERIC_COLUMNS:
            header.append(c)
        writer.writerow(header)
        for future in futures.as_completed(fs):
            # Blocking call - wait for 1 hour for a single future to complete
            # (highly unlikely, most likely something is wrong)
            result = future.result(timeout=60 * 60)
            if result is not None:
                writer.writerow(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-dir',
                        required=True,
                        help='Where the data files are located. If S3, then provide an S3 URI like s3://bucket/path')
    parser.add_argument('-f', '--input-file',
                        required=True,
                        help='Where the summary is located. If S3, then provide an S3 URI like s3://bucket/file.csv')
    parser.add_argument('-od', '--output-folder',
                        required=True,
                        help='Where the output data folder is located.')
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

    # Where the output data folder is located
    output_folder = args.output_folder

    # Where the output summary file is located
    output_file = args.output_file

    limit = int(args.max_patients) if args.max_patients is not None else None

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_folder={output_folder}, output_file={output_file}")
    print("-" * 30)

    run(input_dir, input_file, output_folder, output_file, limit)

    print("DONE")
