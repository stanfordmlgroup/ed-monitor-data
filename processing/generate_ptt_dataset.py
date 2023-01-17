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
import warnings
from concurrent import futures
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks, resample

warnings.filterwarnings("ignore")


def get_ptt(pleth, ii):
    """
    Retrieves the pulse transit time
    Based on: https://github.com/rajpurkarlab/ed-monitor-decompensation/blob/2aa5d97cd4a56a4bb957e50aa26376ad5584f454/hrv_ptt/ptt.py
    """
    assert len(pleth) == len(ii)
    #     assert len(pleth) == 15 * 125

    if np.any(pleth < 0) or np.any(ii < -10):
        # Simple quality check
        return np.nan

    pleth_search = pleth
    ii_search = ii
    ii_peaks, _ = find_peaks(ii_search, distance=37) # Assuming max 200 bpm @ 125 Hz => 125/(200/60) = 37.5
    pleth_peaks, _ = find_peaks(pleth_search, distance=37) # Assuming max 200 bpm @ 125 Hz => 125/(200/60) = 37.5
    ii_peaks_norm = ii_peaks * 8 # Assumes 125 Hz waveforms, so this normalizes to 1000ms
    pleth_peaks_norm = pleth_peaks * 8 # Assumes 125 Hz waveforms, so this normalizes to 1000ms

    ptts = []

    # get RR_ints
    ii_int = np.array([ii_peaks_norm[i + 1] - ii_peaks_norm[i] for i in range(0, len(ii_peaks_norm) - 1)])

    ii_peaks_used = []
    pleth_peaks_used = []
    # Now let's filter for just the peaks we know are good between ECG and PPG:
    while len(ii_peaks_norm) > 0 and len(pleth_peaks_norm) > 0:

        # move pleth_peaks up one if we have an offset of peaks
        while len(ii_peaks_norm) > 0 and len(pleth_peaks_norm) > 0 and ii_peaks_norm[0] > pleth_peaks_norm[0]:
            pleth_peaks_norm = pleth_peaks_norm[1:]
            break

        # if we have nothing left, break out
        if len(ii_peaks_norm) == 0 or len(pleth_peaks_norm) == 0:
            break

        # if we are too behind
        while len(ii_peaks_norm) > 0 and len(pleth_peaks_norm) > 0 and pleth_peaks_norm[0] - ii_peaks_norm[0] > ii_int[
            0]:
            ii_peaks_norm = ii_peaks_norm[1:]
            if len(ii_int) > 1:
                ii_int = ii_int[1:]
            break

        # now we can get our PTTs
        ptt = pleth_peaks_norm[0] - ii_peaks_norm[0]
        if ptt < ii_int[0] and ptt > 0:
            ii_peaks_used.append(int(ii_peaks_norm[0] / 8))
            pleth_peaks_used.append(int(pleth_peaks_norm[0] / 8))
            ptts.append(ptt)

        # Update for next PTT
        ii_peaks_norm = ii_peaks_norm[1:]
        pleth_peaks_norm = pleth_peaks_norm[1:]
        if len(ii_int) > 1:
            ii_int = ii_int[1:]
        else:
            break

    if len(ptts) > 0:
        mean_ptt = np.mean(ptts)
        return float(mean_ptt)
    else:
        return np.nan


def process_patient(input_args):
    i, tot, csn, waveform_start, input_folder, output_file = input_args

    filename = f"{input_folder}/{csn}/{csn}.h5"
    print(f"[{i}/{tot}] Working on patient {csn} at {filename}")
    try:
        with h5py.File(filename, "r") as f:
            ii_processed = resample(f['waveforms']['II'][:], int(len(f['waveforms']['II'][:]) / 4))
            ppg_processed = f['waveforms']['Pleth'][:]

            ptts = []
            for window in range(0, len(ii_processed), 60 * 125):
                ptt = get_ptt(ppg_processed[window:window + 60 * 125], ii_processed[window:window + 60 * 125])
                ptts.append(ptt)
            return (csn, waveform_start, ptts)
    except Exception as e:
        print(f"[ERROR] for patient {csn} due to {e}")
        return None
#         raise e


def run(input_folder, input_file, output_file, limit):
    df = pd.read_csv(input_file)

    fs = []
    with futures.ThreadPoolExecutor(32) as executor:
        i = 0
        for idx, row in tqdm(df.iterrows(), disable=True):
            i += 1
            if limit is not None and i > limit:
                break

            csn = row["patient_id"]
            waveform_start = row["waveform_start_time"]
            input_args = [i, df.shape[0], csn, waveform_start, input_folder, output_file]
            future = executor.submit(process_patient, input_args)
            fs.append(future)

    with open(f"{output_file}", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["CSN", "time", "ptt"])
        for future in futures.as_completed(fs):
            # Blocking call - wait for 1 hour for a single future to complete
            # (highly unlikely, most likely something is wrong)
            output = future.result(timeout=60 * 60)
            if output is not None:
                csn, waveform_start, ptts = output
                waveform_start_time = datetime.strptime(waveform_start, "%Y-%m-%d %H:%M:%S%z")
                t = waveform_start_time
                for ptt in ptts:
                    if not np.isnan(ptt):
                        writer.writerow([csn, t.isoformat(), ptt])
                    t += timedelta(minutes=1)

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
