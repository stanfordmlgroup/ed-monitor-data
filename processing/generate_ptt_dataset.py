#!/usr/bin/env python

"""
Given the output of the consolidate_numerics_waveforms.py file, extract the continuous PTT measures.

Usage:
```
python -u /deep/u/tomjin/ed-monitor-data/processing/generate_ptt_dataset.py --input-dir /deep2/group/ed-monitor/processed/2020_08_01-2022_09_27/patient-data --input-file /deep/group/ed-monitor/processed/2020_08_01-2022_09_27/consolidated.csv --output-file /deep/group/ed-monitor/processed/2020_08_01-2022_09_27/ptt.csv --max-patients 3
```

"""

import argparse
import csv
import gc
import traceback
import warnings
from concurrent import futures
from datetime import datetime, timedelta
from decimal import Decimal

import os
import boto3
import botocore
import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

from edm.utils.ptt import get_ptt

warnings.filterwarnings("ignore")


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

def get_waveform_offsets(start_time, current_time, freq, time_jumps):
    """
    Returns the offset in the waveform for the query time, taking into
    account the time_jumps that have occurred
    :param start_time: The datetime representing when the waveform started
    :param current_time: The datetime representing the current timepoint in the waveform we want to get offset for
    :param freq: The frequency of the waveform
    :param time_jumps: An array of time jumps that occurred in the waveform
    """

    # Find the latest time jump that is before the current_time.
    # Compute the number of seconds since the latest time jump
    current_time_to_time_jump_interval = None
    current_time_to_time_jump_time = None
    current_time_to_time_jump_pos = None
    for tj in time_jumps:
        # ----|    |---
        #     ^    ^
        # pos_1    pos_2
        #
        # tj = (
        #   (pos_1, time_1),
        #   (pos_2, time_2)
        # )
        #
        # Case #1: current_time is after pos_2
        # (the waveform offset would be based on the difference between current_time and pos_2)
        # ---| |---
        #        | <- current_time
        #
        # Case #2: current_time is before pos_1
        # (the waveform offset would be based on the difference between current_time and start_time
        # since no time jumps were relevant in this case)
        # ---| |---
        #  |
        # Case #3: current_time is in between pos_1 and pos_2
        # (not possible, since the code would have jumped ahead to the pos_2)
        # ---| |---
        #     |
        interval_to_time_jump = Decimal(str(round(current_time.timestamp(), 3))) - Decimal(str(round(tj[1][1], 3)))
        if interval_to_time_jump >= 0 and (current_time_to_time_jump_time is None or interval_to_time_jump < current_time_to_time_jump_interval):
            current_time_to_time_jump_time = Decimal(str(round(tj[1][1], 3)))
            current_time_to_time_jump_pos = int(tj[1][0])
            current_time_to_time_jump_interval = interval_to_time_jump

    if current_time_to_time_jump_pos is None:
        start = ((current_time - start_time).total_seconds()) * freq
    else:
        start = (Decimal(str(round(current_time.timestamp(), 3))) - current_time_to_time_jump_time) * freq + current_time_to_time_jump_pos
    return int(start)


def assert_waveform_validity(waveform_base, waveform_type):
    """
    Assert that waveform_base is non-empty and is not flat-lined
    :param waveform_base: The waveform
    :param waveform_type: The type of waveform
    """
    if len(waveform_base) == 0 or (waveform_base == waveform_base[0]).all():
        # The waveform is just empty so skip this patient
        raise Exception(f"Waveform {waveform_type} is empty or flat-line")


def get_time_jump_window(waveform_time_jumps, waveform_type, start, seg_len):
    """
    Returns a tuple of the next position and next time stamp of any time jumps that the current window overlaps with.
    Returns None, None if no time jump is applicable

    :param waveform_time_jumps: Any time jumps that occurred in the waveform
    :param waveform_type: The type of waveform (e.g. II)
    :param start: The start offset of the current window
    :param seg_len: The length of the window
    """
    for tj in waveform_time_jumps[waveform_type]:
        # tj = ((prev pos, prev time), (next pos, next time))
        next_pos = tj[1][0]
        next_time = tj[1][1]
        if start < next_pos < (start + seg_len):
            return next_pos, next_time
    return None, None


def get_ptt_for_patient(ii, pleth, start_time, start_trim_sec, end_time, waveform_len_sec,
                        waveforms_time_jumps, stride_length_sec=10, csn=None):
    """
    Returns a tuple of the pulse-transit-times (PTT) and the times corresponding to the PTTs.
    PTTs are calculated using a sliding window along the entire waveform.

    :param ii: The lead-II ECG array at 500 Hz for the entire patient stay
    :param pleth: The PPG array at 125 Hz for the entire patient stay
    :param start_time: The datetime object representing the start time of the waveform
    :param start_trim_sec: The recommended trim in seconds (anything before this is invalid values)
    :param end_time: The datetime object representing the end time of the waveform
    :param waveform_len_sec: The length of the waveform in seconds
    :param waveforms_time_jumps: Any time jumps in the waveform itself
    :param stride_length_sec: The number of seconds to slide when sliding the window
    """

    # We start looking for waveforms from the left, trying to find the first good waveform
    #
    current_time = start_time + timedelta(seconds=(start_trim_sec))

    # Sanity check that the waveforms are all actually present and not just empty array (saves time)
    assert_waveform_validity(ii, "II")
    assert_waveform_validity(pleth, "Pleth")

    ptt_times = []
    ptts = []

    # For each sliding window...
    while current_time <= (end_time - timedelta(seconds=waveform_len_sec)):
        start_ii = get_waveform_offsets(start_time, current_time, 500, waveforms_time_jumps["II"])
        start_pleth = get_waveform_offsets(start_time, current_time, 125, waveforms_time_jumps["Pleth"])
        orig_ii_seg_len = int(waveform_len_sec * 500)
        seg_len = int(waveform_len_sec * 125)

        # This shouldn't matter if we use II or PPG but we explicitly make an assertion
        ii_time_jump_next_pos, ii_time_jump_next_time = get_time_jump_window(waveforms_time_jumps, "II", start_ii, orig_ii_seg_len)
        ppg_time_jump_next_pos, ppg_time_jump_next_time = get_time_jump_window(waveforms_time_jumps, "Pleth", start_pleth, seg_len)
        assert ii_time_jump_next_time == ppg_time_jump_next_time

        if ii_time_jump_next_pos is None:
            # There was no time jump so we can use this window
            ii_processed = resample(ii[start_ii:(start_ii + orig_ii_seg_len)], int(orig_ii_seg_len / 4))
            ppg_processed = pleth[start_pleth:(start_pleth + seg_len)]

            assert len(ii_processed) == seg_len
            assert len(ppg_processed) == seg_len

            ptt = get_ptt(ppg_processed, ii_processed)
            ptts.append(ptt)
            ptt_times.append(current_time)

        if ii_time_jump_next_pos is not None:
            # We must forcibly move the sliding window to the next time jump point
            print(f"[{datetime.now().isoformat()}] [{csn}] Time jump occurred")
            current_time = datetime.fromtimestamp(ii_time_jump_next_time, tz=current_time.tzinfo)
        else:
            current_time += timedelta(seconds=stride_length_sec)

    return ptts, ptt_times


def process_patient(input_args):
    """
    Processes the PTTs for a single patient visit.
    """
    i, df, csn, waveform_start, input_folder, output_file = input_args

    filename = f"{input_folder}/{str(csn)[-2:]}/{csn}.h5"
    original_filename = filename
    print(f"[{datetime.now().isoformat()}] [{i}/{df.shape[0]}] Working on patient {csn} at {filename}")
    try:
        if original_filename.startswith("s3"):
            tmp_file = f"/tmp/{csn}.h5"
            download_s3_file(filename, tmp_file)
            filename = tmp_file

        with h5py.File(filename, "r") as f:
            row = df[df["patient_id"] == csn]

            waveform_start = row["waveform_start_time"].item()
            waveform_start = datetime.strptime(waveform_start, '%Y-%m-%d %H:%M:%S.%f%z')

            waveform_end = row["waveform_end_time"].item()
            waveform_end = datetime.strptime(waveform_end, '%Y-%m-%d %H:%M:%S.%f%z')
            recommended_trim_start_sec = int(row["recommended_trim_start_sec"].item())

            if "II" not in f['waveforms'] or "Pleth" not in f['waveforms']:
                return None

            ii = f['waveforms']['II'][:]
            ppg = f['waveforms']['Pleth'][:]
            waveforms_time_jumps = f["waveforms_time_jumps"]

            ptts, ptt_times = get_ptt_for_patient(ii, ppg, waveform_start, recommended_trim_start_sec, waveform_end, waveform_len_sec=60,
                                                  waveforms_time_jumps=waveforms_time_jumps, stride_length_sec=60)

        if original_filename.startswith("s3"):
            os.remove(filename)

        return (csn, ptt_times, ptts)
    except Exception as e:
        print(f"[ERROR] for patient {csn} due to {e}")
        print(traceback.format_exc())
        return None


def run(input_folder, input_file, output_file, limit):
    """
    Runs the script given the parameters
    """
    if input_file.startswith("s3"):
        tmp_file = "/tmp/input.csv"
        download_s3_file(input_file, tmp_file)
        input_file = tmp_file

    df = pd.read_csv(input_file)

    fs = {}
    with futures.ThreadPoolExecutor(32) as executor:
        i = 0
        for idx, row in tqdm(df.iterrows(), disable=True):
            i += 1
            if limit is not None and i > limit:
                break

            csn = row["patient_id"]
            waveform_start = row["waveform_start_time"]
            input_args = [i, df, csn, waveform_start, input_folder, output_file]
            future = executor.submit(process_patient, input_args)
            fs[future] = i

    with open(f"{output_file}", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["CSN", "time", "ptt"])
        for future in futures.as_completed(fs):
            # Blocking call - wait for 1 hour for a single future to complete
            # (highly unlikely, most likely something is wrong)
            output = future.result(timeout=60 * 60)
            fs.pop(future)
            if output is not None:
                csn, ptt_times, ptts = output
                assert len(ptt_times) == len(ptts)
                for i, ptt in enumerate(ptts):
                    if not np.isnan(ptt):
                        writer.writerow([csn, ptt_times[i].isoformat(), ptt])
            gc.collect() # just to be sure gc is called


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

    # Where the output data file is located
    output_file = args.output_file

    limit = int(args.max_patients) if args.max_patients is not None else None

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_file={output_file}")
    print("-" * 30)

    run(input_dir, input_file, output_file, limit)

    print("DONE")
