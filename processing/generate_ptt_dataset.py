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

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

from edm.utils.ptt import get_ptt

warnings.filterwarnings("ignore")


def get_waveform_offsets(start_time, current_time, freq, time_jumps):
    """
    Returns the offset in the waveform for the query time, taking into
    account the time_jumps that have occurred
    """

    # Find the latest time jump that is before the current_time.
    # Compute the number of seconds since the latest time jump
    current_time_to_time_jump_interval = None
    current_time_to_time_jump_time = None
    current_time_to_time_jump_pos = None
    for tj in time_jumps:
        # tj = ((pos_1, time_1), (pos_2, time_2))
        # Case #1: After gap
        # ---| |---
        #        | <- current_time
        # Case #2: Before gap
        # ---| |---
        #  |
        # Case #3: In between gap
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
    if len(waveform_base) == 0 or (waveform_base == waveform_base[0]).all():
        # The waveform is just empty so skip this patient
        raise Exception(f"Waveform {waveform_type} is empty or flat-line")


def get_time_jump_window(waveform_time_jumps, waveform_type, start, seg_len):
    """
    Returns the time jump window, including the next position and next time stamp.
    Returns None, None if no timejump is applicable
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
    i, df, csn, waveform_start, input_folder, output_file = input_args

    filename = f"{input_folder}/{str(csn)[-2:]}/{csn}.h5"
    print(f"[{datetime.now().isoformat()}] [{i}/{df.shape[0]}] Working on patient {csn} at {filename}")
    try:
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

            return (csn, ptt_times, ptts)
    except Exception as e:
        print(f"[ERROR] for patient {csn} due to {e}")
        print(traceback.format_exc())
        return None


def run(input_folder, input_file, output_file, limit):
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
