#!/usr/bin/env python

"""
Script to generate numerics data

Usage:
```
python -u /deep/u/tomjin/ed-monitor-self-supervised/preprocessing/generate_downstream_dataset_numerics_only.py --input-dir /deep/group/ed-monitor-self-supervised/v3/patient-data --input-file /deep/group/ed-monitor-self-supervised/v3/consolidated.csv --output-data-file /deep/group/ed-monitor-self-supervised/v4/downstream.15min.60min.head.h5 --output-summary-file /deep/group/ed-monitor-self-supervised/v4/downstream.15min.60min.head.csv --pre-minutes 15 --max-patients 3
```

"""

import argparse
import csv
from concurrent import futures
from datetime import datetime, timedelta

import pytz

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

NUMERIC_COLUMNS = ['HR', 'RR', 'SpO2', 'btbRRInt_ms', 'NBPs', 'NBPd', 'Perf']
VERBOSE = False


def get_numerics(numerics_obj, start_epoch, end_epoch, pre_numerics_len_sec):
    start = start_epoch
    end = end_epoch
    output = {}
    for col in NUMERIC_COLUMNS:
        output[col] = np.zeros(
            pre_numerics_len_sec)  # Roughly reserve space for 1 data point per sec (required to obtain a square matrix)
        output[col][:] = np.NaN
        output[f"{col}-time"] = np.zeros(pre_numerics_len_sec)
        output[f"{col}-length"] = 0
        if col in numerics_obj:
            vals = np.array(numerics_obj[col])
            times = np.array(numerics_obj[f"{col}-time"])
            indices_of_interest = np.argwhere(np.logical_and(times >= start, times <= end))
            # Take only the values closest to the alignment point
            indices_of_interest = indices_of_interest[-pre_numerics_len_sec:]
            offset = max(0, pre_numerics_len_sec - len(indices_of_interest))
            output[col][offset:] = np.squeeze(vals[indices_of_interest])
            output[f"{col}-time"][offset:] = np.squeeze(times[indices_of_interest]).astype(int)
            output[f"{col}-length"] = len(indices_of_interest)
    return output


def get_numerics_averaged_by_minute(numerics_obj, start_epoch, end_epoch, post_numerics_min):
    start = start_epoch
    end = end_epoch
    output = {}
    for col in NUMERIC_COLUMNS:
        output[col] = np.zeros(post_numerics_min)
        output[col][:] = np.NaN
        output[f"{col}-time"] = np.zeros(post_numerics_min)
        output[f"{col}-length"] = 0

        if col in numerics_obj:
            vals = np.array(numerics_obj[col])
            times = np.array(numerics_obj[f"{col}-time"])
            indices_of_interest = np.argwhere(np.logical_and(times >= start, times <= end))
            if VERBOSE:
                print(f"len(vals) = {len(vals)}")
                print(f"len(times) = {len(times)}")
                print(f"start = {datetime.fromtimestamp(start).isoformat('T')}")
                print(f"end = {datetime.fromtimestamp(end).isoformat('T')}")
                print(f"len(indices_of_interest) = {len(indices_of_interest)}")

            temp_min = 0
            temp_list = []
            valid_vals = 0
            for idx in indices_of_interest:
                #                 if VERBOSE:
                #                     print(f"idx = {idx}")
                if temp_min >= post_numerics_min:
                    break
                if (start + temp_min * 60) <= times[idx] < (start + (temp_min + 1) * 60):
                    temp_list.append(vals[idx])
                else:
                    output[col][temp_min] = np.mean(temp_list)
                    output[f"{col}-time"][temp_min] = start + temp_min * 60
                    if len(temp_list) > 0:
                        valid_vals += 1
                    temp_list = [vals[idx]]
                    temp_min += 1

            if len(temp_list) > 0 and temp_min < post_numerics_min:
                output[col][temp_min] = np.mean(temp_list)
                output[f"{col}-time"][temp_min] = start + temp_min * 60

            output[f"{col}-length"] = valid_vals
    return output


def process_patient(input_args):
    i, df, csn, input_folder, pre_minutes_min, align_col = input_args

    filename = f"{input_folder}/{csn}/{csn}.h5"
    print(f"[{i}/{df.shape[0]}] Working on patient {csn} at {filename}")
    try:

        output = {
            "csn": csn,
            "alignment_times": [],
            "numerics_before": {},
        }
        for col in NUMERIC_COLUMNS:
            output["numerics_before"][col] = {
                "vals": [],
                "times": [],
                "lengths": [],
            }

        with h5py.File(filename, "r") as f:
            row = df[df["patient_id"] == csn]

            roomed_start = row["Roomed_time"].item()
            roomed_start = datetime.strptime(roomed_start, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
            roomed_start = pytz.timezone('America/Vancouver').localize(roomed_start)
            if align_col is not None:
                max_alignment_time = row[align_col].item()
                max_alignment_time = datetime.strptime(max_alignment_time, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
                max_alignment_time = pytz.timezone('America/Vancouver').localize(max_alignment_time)
            else:
                max_alignment_time = None

            # start_time is when the waveform monitoring starts (note that anything before the
            # recommended trim was an empty array, even though it might have technically been
            # part of the patient's visit)
            start_time = roomed_start
            alignment_time = start_time + timedelta(seconds=(pre_minutes_min * 60))

            if max_alignment_time is not None:
                # The idea is that we take either the minimum of the alignment time we derived from
                # the specified pre minutes or the max alignment time. Example:
                # Say we had pre-minutes of 10 min
                #
                # -------------------------
                #    ^
                #    calculated alignment time
                #             ^ max alignment time
                #  We choose the actual alignment time here to be the calculated alignment time.
                #
                if max_alignment_time.timestamp() < alignment_time.timestamp():
                    alignment_time = max_alignment_time

            start_time_epoch = int(start_time.timestamp())
            alignment_time_epoch = int(alignment_time.timestamp())

            if VERBOSE:
                print(f"start_time = {start_time}")
                print(f"max_alignment_time = {max_alignment_time}")
                print(f"alignment_time = {alignment_time}")

            # numerics_map_before = get_numerics(f["numerics"], start_time_epoch, alignment_time_epoch,
            #                                    pre_numerics_len_sec=pre_minutes_min * 60)
            numerics_map_before = get_numerics_averaged_by_minute(f["numerics"], start_time_epoch, alignment_time_epoch,
                                                                  pre_minutes_min)
            for col in NUMERIC_COLUMNS:
                output["numerics_before"][col]["vals"].append(numerics_map_before[col])
                output["numerics_before"][col]["times"].append(numerics_map_before[f"{col}-time"])
                output["numerics_before"][col]["lengths"].append(numerics_map_before[f"{col}-length"])

            output["alignment_times"].append(alignment_time_epoch)

        output_str = f"[{i}/{df.shape[0]}]   [{csn}] "
        print(output_str)
        return output
    except Exception as e:
        print(f"[ERROR] for patient {csn} due to {e}")
        return None
        # raise e


def run(input_folder, input_file, output_data_file, output_summary_file,
        pre_minutes_min, align_col, limit):
    df = pd.read_csv(input_file)
    patients = df["patient_id"].tolist()

    csns = []
    alignment_times = []
    numerics_before = {}
    for col in NUMERIC_COLUMNS:
        numerics_before[col] = {
            "vals": [],
            "times": [],
            "lengths": [],
        }

    fs = []
    with futures.ThreadPoolExecutor(32) as executor:
        i = 0
        for csn in tqdm(patients, disable=True):
            i += 1
            if limit is not None and i > limit:
                break

            input_args = [i, df, csn, input_folder, pre_minutes_min, align_col]
            future = executor.submit(process_patient, input_args)
            fs.append(future)

    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result = future.result(timeout=60 * 60)
        if result is not None:
            csns.extend([result["csn"] for t in result["alignment_times"]])
            alignment_times.extend(result["alignment_times"])
            for col in NUMERIC_COLUMNS:
                numerics_before[col]["vals"].extend(result["numerics_before"][col]["vals"])
                numerics_before[col]["times"].extend(result["numerics_before"][col][f"times"])
                numerics_before[col]["lengths"].extend(result["numerics_before"][col][f"lengths"])

    with h5py.File(output_data_file, "w") as f:
        f.create_dataset("alignment_times", data=alignment_times)
        dset_before = f.create_group("numerics_before")
        for k in NUMERIC_COLUMNS:
            dset_k = dset_before.create_group(k)
            dset_k.create_dataset(f"vals", data=numerics_before[k]["vals"])
            dset_k.create_dataset(f"times", data=numerics_before[k][f"times"])

    with open(output_summary_file, "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        headers = ["patient_id", "alignment_time"]
        for k in NUMERIC_COLUMNS:
            headers.append(f"{k}_before_length")
        writer.writerow(headers)

        i = 0
        while i < len(alignment_times):
            row = [csns[i], alignment_times[i]]
            for k in NUMERIC_COLUMNS:
                row.append(numerics_before[k]["lengths"][i])
            writer.writerow(row)
            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-dir',
                        required=True,
                        help='Where the data files are located')
    parser.add_argument('-f', '--input-file',
                        required=True,
                        help='Where the summary is located')
    parser.add_argument('-od', '--output-data-file',
                        required=True,
                        help='Where the output data file is located')
    parser.add_argument('-os', '--output-summary-file',
                        required=True,
                        help='Where the output summary file is located')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')
    parser.add_argument('-pr', '--pre-minutes',
                        default=15,
                        help='Number of minutes since beginning of waveform recording to use as the initial window of high resolution input data')
    parser.add_argument('-a', '--align',
                        default=None,
                        help='Specify a column to use as the maximum alignment column. e.g. we might want to collect as much numerics before the first blood draw time')

    args = parser.parse_args()

    # Where the data files are located
    input_dir = args.input_dir

    # Where the summary is located
    input_file = args.input_file

    # Where the output data file is located
    output_data_file = args.output_data_file

    # Where the output summary file is located
    output_summary_file = args.output_summary_file

    pre_minutes_min = int(args.pre_minutes)

    limit = int(args.max_patients) if args.max_patients is not None else None

    align_col = args.align

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_data_file={output_data_file}, output_summary_file={output_summary_file}")
    print(f"pre_minutes_min={pre_minutes_min}")
    print("-" * 30)

    run(input_dir, input_file, output_data_file, output_summary_file, pre_minutes_min, align_col, limit)

    print("DONE")
