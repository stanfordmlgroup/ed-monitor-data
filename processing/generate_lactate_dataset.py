#!/usr/bin/env python

"""
Script to generate a dataset to summarize key features for the lactate project

Usage:
```
python -u /deep/u/tomjin/ed-monitor-lactate/preprocessing/generate_lactate_dataset.py --input-dir /deep/group/ed-monitor-self-supervised/v3/patient-data --input-file /deep/group/ed-monitor-self-supervised/v3/consolidated.csv --output-file /deep/group/ed-monitor-self-supervised/v4/downstream.15min.60min.head.csv
```

"""

import argparse
import csv
from concurrent import futures
from datetime import datetime, timedelta
from dateutil import parser as dt_parser
import traceback

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample, find_peaks

from edm.utils.ptt import get_ptt
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

TYPE_II = "II"
TYPE_PLETH = "Pleth"
WAVEFORM_COLUMNS = [TYPE_II, TYPE_PLETH]
NUMERIC_COLUMNS = ['HR', 'RR', 'SpO2', 'btbRRInt_ms', 'NBPs', 'NBPd', 'Perf']
VERBOSE = False


def get_numerics_averaged_by_minute(numerics_obj, start_epoch, end_epoch):
    start = start_epoch
    end = end_epoch
    output = {}
    output_var = {}
    for col in NUMERIC_COLUMNS:
        output[col] = []
        output_var[col] = []

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

            if len(indices_of_interest) == 0 and len(vals) > 0:
                # Let's carry forward the last value since there were no numerics that fell in our range
                output[col].append(vals[-1])
                output_var[col].append(0)

            temp_min = 0
            temp_list = []
            valid_vals = 0

            # Handle case where there are gaps in the recorded data at the start
            if len(indices_of_interest) > 0:
                while times[indices_of_interest[0]] >= (start + (temp_min + 1) * 60):
                    output[col].append(np.nan)
                    output_var[col].append(np.nan)
                    temp_min += 1

            for idx in indices_of_interest:
                if (start + temp_min * 60) <= times[idx] < (start + (temp_min + 1) * 60):
                    temp_list.append(vals[idx])
                else:
                    if len(temp_list) == 0:
                        output[col].append(np.nan)
                        output_var[col].append(np.nan)
                    else:
                        output[col].append(np.mean(temp_list))
                        output_var[col].append(np.var(temp_list))
                        valid_vals += 1
                    temp_list = [vals[idx]]
                    temp_min += 1

                    # Handle case where there are gaps in the recorded data
                    while times[idx] >= (start + (temp_min + 1) * 60):
                        output[col].append(np.nan)
                        output_var[col].append(np.nan)
                        temp_min += 1

            if len(temp_list) > 0:
                output[col].append(np.mean(temp_list))
                output_var[col].append(np.var(temp_list))
                valid_vals += 1
        output[col] = np.array(output[col])
        output_var[col] = np.array(output_var[col])
    return output, output_var


def get_trend(vals_of_interest):
    try:
        vals_of_interest_no_nan = vals_of_interest[~np.isnan(vals_of_interest)]
        if len(vals_of_interest_no_nan) < 2:
            return np.nan
        x = np.arange(0, len(vals_of_interest_no_nan))
        y = np.array(vals_of_interest_no_nan)
        z = np.polyfit(x, y, 1)
        return z[0]
    except Exception as e:
        print("[get_trend]", e)
        return np.nan


def get_numerics_stats(obj):
    output = {}
    for col in NUMERIC_COLUMNS:
        output[col] = {}
        if col in obj:
            vals_of_interest = obj[col]
            try:
                output[col]["mean"] = np.nanmean(vals_of_interest)
            except Exception as e:
                print("[get_numerics_stats]", e)
                output[col]["mean"] = np.nan
            output[col]["trend"] = get_trend(vals_of_interest)
    return output


def get_waveform_features(f, waveform_start_time, start_time, end_time, waveform_len_sec=60):
    ii_processed = resample(f['waveforms']['II'][:], int(len(f['waveforms']['II'][:]) / 4))
    ppg_processed = f['waveforms']['Pleth'][:]

    ptts = []
    peaks = []

    if waveform_start_time > start_time:
        current_time = waveform_start_time
        curr_offset = 0
    else:
        current_time = start_time
        curr_offset = int((start_time - waveform_start_time).total_seconds() * 125)

    for window in range(curr_offset, len(ii_processed), waveform_len_sec * 125):
        if current_time >= (end_time - timedelta(seconds=waveform_len_sec)):
            break
        try:
            ppg_window = ppg_processed[window:window + waveform_len_sec * 125]
            ptt = get_ptt(ppg_window, ii_processed[window:window + waveform_len_sec * 125])
            ptts.append(ptt)
            pleth_peaks, _ = find_peaks(ppg_window, distance=37)  # Assuming max 200 bpm @ 125 Hz => 125/(200/60) = 37.5
            peaks.append(np.nanmean(ppg_window[pleth_peaks]))
        except Exception as e:
            print("[get_waveform_features]", e)
            ptts.append(np.nan)
            peaks.append(np.nan)
            pass
        current_time += timedelta(seconds=waveform_len_sec)

    return np.array(ptts), np.array(peaks)


def process_patient(input_args):
    i, df, csn, input_folder = input_args

    filename = f"{input_folder}/{csn}/{csn}.h5"
    print(f"[{datetime.now().isoformat()}] [{i}/{df.shape[0]}] Working on patient {csn} at {filename}")
    try:
        output = [csn]

        with h5py.File(filename, "r") as f:
            row = df[df["patient_id"] == csn]

            waveform_start = row["waveform_start_time"].item()
            waveform_start = datetime.strptime(waveform_start, '%Y-%m-%d %H:%M:%S%z')
            start_time = row["Collection_time_1"].item()
            start_time = dt_parser.parse(start_time).replace(tzinfo=waveform_start.tzinfo)
            start_time_epoch = start_time.timestamp()
            end_time = row["Collection_time_2"].item()
            end_time = dt_parser.parse(end_time).replace(tzinfo=waveform_start.tzinfo)
            end_time_epoch = end_time.timestamp()

            try:
                ptts, peaks = get_waveform_features(f, waveform_start, start_time, end_time)
                if len(ptts) > 0:
                    ptt_mean = np.nanmean(ptts)
                    ptt_trend = get_trend(ptts)
                else:
                    ptt_mean = np.nan
                    ptt_trend = np.nan
                if len(peaks) > 0:
                    peaks_mean = np.nanmean(peaks)
                    peaks_trend = get_trend(peaks)
                else:
                    peaks_mean = np.nan
                    peaks_trend = np.nan
                output.extend([ptt_mean, ptt_trend, peaks_mean, peaks_trend])
            except Exception as ec:
                raise Exception(f"Patient did not have any useable waveforms. Technical: {ec}")

            numerics_mean_obj, numerics_var_obj = get_numerics_averaged_by_minute(f["numerics"], start_time_epoch, end_time_epoch)
            numerics_mean_stats = get_numerics_stats(numerics_mean_obj)
            numerics_var_stats = get_numerics_stats(numerics_var_obj)

            for col in NUMERIC_COLUMNS:
                output.append(numerics_mean_stats[col]["mean"])
                output.append(numerics_mean_stats[col]["trend"])
                output.append(numerics_var_stats[col]["mean"])
                output.append(numerics_var_stats[col]["trend"])

        return output
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] [ERROR] for patient {csn} due to {e}")
#         print(traceback.format_exc())
        return None
#         raise e


def run(input_folder, input_file, output_file, limit):
    df = pd.read_csv(input_file)
    patients = df["patient_id"].tolist()

    output_rows = []
    header = ["csn", "ptt_mean", "ptt_trend", "peaks_mean", "peaks_trend"]
    for col in NUMERIC_COLUMNS:
        header.append(f"{col}")
        header.append(f"{col}-trend")
        header.append(f"{col}-var")
        header.append(f"{col}-var-trend")
    output_rows.append(header)

    fs = []
    with futures.ThreadPoolExecutor(32) as executor:
        i = 0
        for csn in tqdm(patients, disable=True):
            i += 1
            if limit is not None and i > limit:
                break

            input_args = [i, df, csn, input_folder]
            future = executor.submit(process_patient, input_args)
            fs.append(future)

    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result = future.result(timeout=60 * 60)
        if result is not None:
            output_rows.append(result)

    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        for row in output_rows:
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-dir',
                        required=True,
                        help='Where the data files are located')
    parser.add_argument('-f', '--input-file',
                        required=True,
                        help='Where the summary is located')
    parser.add_argument('-o', '--output-file',
                        required=True,
                        help='Where the output file is located')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')

    args = parser.parse_args()

    # Where the data files are located
    input_dir = args.input_dir

    # Where the summary is located
    input_file = args.input_file

    # Where the output file is located
    output_file = args.output_file

    limit = int(args.max_patients) if args.max_patients is not None else None

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_file={output_file}")
    print("-" * 30)

    run(input_dir, input_file, output_file, limit)

    print("DONE")
