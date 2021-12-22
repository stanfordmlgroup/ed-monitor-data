#!/usr/bin/env python

"""
Script to prepare the ED-acquired Philips waveforms, producing a single file containing all the files that can be loaded as a tensor. 

Extracts N waveforms BEFORE and AFTER the specified time boundary when a medication is administered.

Example: python prepare_ed_waveforms_time_boundary.py -i /deep/group/lactate/v3/consolidated.filtered.administered.csv -d /deep/group/lactate/v3/patient-data -o /deep/group/lactate/v3/patient-data/waveforms/admin-time-1-boundary -l 15 -f 500 -n -w II -m 10 -t Admin_time_1 -b 600 -p 1
"""

import argparse
import csv
import datetime
import pickle
from concurrent import futures
from pathlib import Path
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import pytz

from edm.utils.waveforms import WAVEFORMS_OF_INTERST, get_waveform

pd.set_option('display.max_columns', 500)

PATIENCE = 30 # number of rounds before we give up trying to find a non-empty waveform
INNER_RIGHT_BUFFER = 30 * 60 * 500 # bolus can take 20-30 min to complete, so ensure we pick right waveforms after medication is fully administered

def load_pkl_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def process_record(input_args):
    i, total_rows, waveform_df, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, max_samples_per_patient, time_boundary, buffer = input_args

    # Ensure consistent range selected for each patient file
    np.random.seed(i)
    
    try:
        patient_id = waveform_df.iloc[[i]]["patient_id"].item()
        trim_length = waveform_df.iloc[[i]]["trim_length"].item()

        recommended_trim_start_sec = waveform_df.iloc[[i]]["recommended_trim_start_sec"].item()
        recommended_trim_end_sec = waveform_df.iloc[[i]]["recommended_trim_end_sec"].item()
        time_boundary_val = waveform_df.iloc[[i]][time_boundary].item()
        if not isinstance(time_boundary_val, str):
            print(f"[{i}/{total_rows}] {patient_id} waveform did not have a {time_boundary} value")
            return None, None
        else:
            time_boundary_time = datetime.datetime.strptime(time_boundary_val, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
            time_boundary_time = pytz.timezone('America/Vancouver').localize(time_boundary_time)

            waveform_start_time = waveform_df.iloc[[i]]["waveform_start_time"].item()
            waveform_start_time = datetime.datetime.strptime(waveform_start_time, "%Y-%m-%d %H:%M:%S%z").replace(tzinfo=None)
            # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
            waveform_start_time = pytz.timezone('America/Vancouver').localize(waveform_start_time)
            time_boundary_offset_sec = (time_boundary_time - waveform_start_time).total_seconds()


        info = load_pkl_file(f"{patient_dir}/{patient_id}/info.pkl")

        num_waveforms_processed_left = {}
        num_waveforms_processed_right = {}
        for waveform_type in waveform_types:
            if waveform_type not in num_waveforms_processed_left:
                num_waveforms_processed_left[waveform_type] = 0
                num_waveforms_processed_right[waveform_type] = 0

        waveforms_left = {}
        waveforms_right = {}
        for waveform_type in waveform_types:

            waveform_base = np.load(f"{patient_dir}/{patient_id}/{waveform_type}.dat.npy")

            waveform_config = WAVEFORMS_OF_INTERST[waveform_type]
            fs = waveform_config["orig_frequency"]

            if waveform_type in info["supported_types"]:
                window_size = int(waveform_length * fs)
                attempt_left = 1
                attempt_right = 1

                while num_waveforms_processed_left[waveform_type] < max_samples_per_patient:

                    start_offset = int(max(0, recommended_trim_start_sec * fs))
                    end_offset = int(min(len(waveform_base), recommended_trim_end_sec * fs))
                    
                    if ((time_boundary_offset_sec + buffer) * fs) > end_offset:
                        print(f"[{i}/{total_rows}] {patient_id} waveform did not have enough values on the left")
                        return None, None

                    middle_offset = time_boundary_offset_sec * fs
                    # print(f"start_offset={start_offset}, end_offset={end_offset}, middle_offset={middle_offset}")
                    range_l = max(start_offset, middle_offset - buffer * fs)
                    range_r = max(range_l + 1, middle_offset - window_size)
                    pointer_left = np.random.randint(range_l, range_r)
                    waveform_left, quality_left = get_waveform(waveform_base, pointer_left, window_size, fs, should_normalize=should_normalize, bandpass_type=waveform_config["bandpass_type"], bandwidth=waveform_config["bandpass_freq"], target_fs=waveform_target_freq, ecg_quality_check=True)
                    
                    if attempt_left >= PATIENCE:
                        # We've run out of patience so we'll use whatever waveform we get
                        print(f"[{i}/{total_rows}] {patient_id} waveform ran out of patience at LEFT {pointer}.")
                    elif quality_left == 0:
                        print(f"[{i}/{total_rows}] {patient_id} waveform was empty at {pointer_left}. Trying again...")
                        attempt_left += 1
                        continue

                    print(f"[{i}/{total_rows}] {patient_id} waveform {waveform_type} of shape {waveform_left.shape} at {pointer_left}, LEFT picked between {range_l} and {range_r}")

                    if waveform_type not in waveforms_left:
                        waveforms_left[waveform_type] = []

                    waveforms_left[waveform_type].append({
                        "record_name": patient_id,
                        "pointer": pointer_left,
                        "waveform": waveform_left
                    })

                    num_waveforms_processed_left[waveform_type] += 1


                while num_waveforms_processed_right[waveform_type] < max_samples_per_patient:

                    start_offset = int(max(0, recommended_trim_start_sec * fs))
                    end_offset = int(min(len(waveform_base), recommended_trim_end_sec * fs))

                    if (time_boundary_offset_sec * fs + INNER_RIGHT_BUFFER) < start_offset:
                        print(f"[{i}/{total_rows}] {patient_id} waveform did not have enough values on the right")
                        return None, None

                    middle_offset = time_boundary_offset_sec * fs + INNER_RIGHT_BUFFER
                    range_l = middle_offset
                    range_r = min(max(range_l + 1, middle_offset + buffer * fs), end_offset - window_size)
                    pointer_right = np.random.randint(range_l, range_r)
                    waveform_right, quality_right = get_waveform(waveform_base, pointer_right, window_size, fs, should_normalize=should_normalize, bandpass_type=waveform_config["bandpass_type"], bandwidth=waveform_config["bandpass_freq"], target_fs=waveform_target_freq, ecg_quality_check=True)

                    if attempt_right >= PATIENCE:
                        # We've run out of patience so we'll use whatever waveform we get
                        print(f"[{i}/{total_rows}] {patient_id} waveform ran out of patience at RIGHT {pointer_right}.")
                    elif quality_right == 0:
                        print(f"[{i}/{total_rows}] {patient_id} waveform was empty at {pointer_right}. Trying again...")
                        attempt_right += 1
                        continue

                    print(f"[{i}/{total_rows}] {patient_id} waveform {waveform_type} of shape {waveform_right.shape} at {pointer_right}, RIGHT picked between {range_l} and {range_r}")

                    if waveform_type not in waveforms_right:
                        waveforms_right[waveform_type] = []

                    waveforms_right[waveform_type].append({
                        "record_name": patient_id,
                        "pointer": pointer_right,
                        "waveform": waveform_right
                    })

                    num_waveforms_processed_right[waveform_type] += 1

                assert num_waveforms_processed_left[waveform_type] == max_samples_per_patient
                assert num_waveforms_processed_right[waveform_type] == max_samples_per_patient
                print(f"[{i}/{total_rows}] {patient_id} waveform {waveform_type} completed")

        return waveforms_left, waveforms_right
    except Exception as e:
        print("Unexpected error:", e)
        return None, None
    

def run(args):
    print(f"START TIME: {datetime.datetime.now()}")
    waveform_types = args.waveform_types.split(",")
    input_file = args.waveform_file
    patient_dir = args.patient_dir
    should_normalize = args.normalize
    output_folder = args.output_folder
    max_samples_per_patient = int(args.max_samples_per_patient)
    max_patients = int(args.max_patients) if args.max_patients is not None else None
    waveform_length = int(args.length)
    waveform_target_freq = float(args.frequency)
    waveform_types = waveform_types
    time_boundary = args.time_boundary
    buffer = int(args.buffer)
    
    output_folder = f"{output_folder}/{waveform_length}sec-{int(waveform_target_freq)}hz-{int(should_normalize)}norm-{max_samples_per_patient}wpp"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for w in waveform_types:
        Path(f"{output_folder}/{w}/left").mkdir(parents=True, exist_ok=True)
        Path(f"{output_folder}/{w}/right").mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_file)
    print(f"Found {input_file} with shape {df.shape}")
    total_rows = len(df)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, row in tqdm(df.iterrows(), disable=True):
            input_args = [i, total_rows, df, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, max_samples_per_patient, time_boundary, buffer]
            future = executor.submit(process_record, input_args)
            fs.append(future)
            if max_patients is not None and i >= (max_patients - 1):
                break

    waveforms_left = {}
    waveforms_right = {}
    for w in waveform_types:
        waveforms_left[w] = []
        waveforms_right[w] = []
    
    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result_left, result_right = future.result(timeout=60*60)
        if result_left is not None:
            for w in result_left.keys():
                waveforms_left[w].extend(result_left[w])
        if result_right is not None:
            for w in result_right.keys():
                waveforms_right[w].extend(result_right[w])

    for w in waveform_types:
        output_waveforms_left = []
        output_waveforms_right = []

        with open(f"{output_folder}/{w}/left/summary.csv", "w") as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            headers = ["record_name", "pointer"]
            writer.writerow(headers)
            for row in waveforms_left[w]:
                writer.writerow([row["record_name"], row["pointer"]])
                output_waveforms_left.append(row["waveform"])

        with open(f"{output_folder}/{w}/right/summary.csv", "w") as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            headers = ["record_name", "pointer"]
            writer.writerow(headers)
            for row in waveforms_right[w]:
                writer.writerow([row["record_name"], row["pointer"]])
                output_waveforms_right.append(row["waveform"])

        output_tensor_left = np.array(output_waveforms_left)
        output_tensor_right = np.array(output_waveforms_right)
        np.save(f"{output_folder}/{w}/left/waveforms.dat", output_tensor_left)
        np.save(f"{output_folder}/{w}/right/waveforms.dat", output_tensor_right)
    print(f"Output is written to: {output_folder}/{w}/left/summary.csv")
    print(f"Output is written to: {output_folder}/{w}/right/summary.csv")
    print(f"Output is written to: {output_folder}/{w}/left/waveforms.dat.npy")
    print(f"Output is written to: {output_folder}/{w}/right/waveforms.dat.npy")
    print(f"END TIME: {datetime.datetime.now()}")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts waveforms from the patient dir to create a single file')
    parser.add_argument('-i', '--waveform-file',
                        required=True,
                        help='The path to the waveforms file')
    parser.add_argument('-d', '--patient-dir',
                        required=True,
                        help='The path to the patient dir')
    parser.add_argument('-o', '--output-folder',
                        required=True,
                        help='Folder where the output files will be written')
    parser.add_argument('-l', '--length',
                        default=15,
                        help='Length of the output waveform (sec)')
    parser.add_argument('-n', '--normalize', action='store_true', help='Normalize the waveform')
    parser.add_argument('-f', '--frequency',
                        default=500,
                        help='Length of the output frequency (Hz)')
    parser.add_argument('-m', '--max-samples-per-patient',
                        default=1,
                        help='Maximum number of samples to pull from each unique patient visit')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')
    parser.add_argument('-w', '--waveform-types',
                        required=True,
                        help='Comma separated list of waveform types to process. Supported values: II, PLETH, RESP')
    parser.add_argument('-t', '--time-boundary',
                        default=None,
                        help='Provide a field name of a column which represents when a medication was administered')
    parser.add_argument('-b', '--buffer',
                        default=None,
                        help='Time in seconds before/after the boundary to sample from')

    args = parser.parse_args()

    run(args)

    print("DONE")
