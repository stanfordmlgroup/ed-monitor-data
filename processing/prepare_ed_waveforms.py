#!/usr/bin/env python

"""
Script to prepare the ED-acquired Philips waveforms, producing a single file containing all the files that can be loaded as a tensor. 

Example: python prepare_ed_waveforms.py -i /deep/group/pulmonary-embolism/v2/consolidated.filtered.csv -d /deep/group/pulmonary-embolism/v2/patient-data -o /deep/group/pulmonary-embolism/v2/waveforms -l 15 -f 500 -w II -p 1 -n
"""

import datetime
import pandas as pd
import numpy as np
import os
import sys
import pytz
import re
import csv
import matplotlib.pyplot as plt
import math
import pickle
from biosppy.signals.tools import filter_signal
from concurrent import futures
import argparse
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import wfdb
import torch
from scipy import signal
from scipy.signal import decimate, resample
from pathlib import Path
import datetime

pd.set_option('display.max_columns', 500)

WAVEFORMS_OF_INTERST = {
    "II": {
        "orig_frequency": 500,
        "bandpass_type": 'filter',
        "bandpass_freq": [3, 45]
    }, 
    "Pleth": {
        "orig_frequency": 125,
        "bandpass_type": None,
        "bandpass_freq": [3, 45]
    },
    "Resp": {
        "orig_frequency": 62.5,
        "bandpass_type": 'cheby2',
        "bandpass_freq": [0.5, 10]
    }
}
PATIENCE = 10 # number of rounds before we give up trying to find a non-empty waveform

def normalize(seq, smooth=1e-8):
    """
    Normalize each sequence between -1 and 1
    """
    return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq) + smooth) - 1

def apply_filter(signal, filter_bandwidth=[3, 45], fs=500):
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    try:
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth,
                                     sampling_rate=fs)
    except:
        pass

    return signal

def get_waveform(waveform, start, window_sz, orig_fs, should_normalize=False, bandpass_type=None, bandwidth=[3, 45], target_fs=None):
    waveform = waveform[(start):(start + window_sz)]
    if bandpass_type == "cheby2":
        # Recommended by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6316358/
        b, a = signal.cheby2(4, 40, bandwidth, 'bandpass', fs=orig_fs, output='ba')
        waveform = signal.lfilter(b, a, waveform)
    elif bandpass_type == "filter":
        waveform = apply_filter(waveform, filter_bandwidth=bandwidth, fs=orig_fs)
    if should_normalize:
        bottom, top = np.percentile(waveform, [1, 99])
        waveform = np.clip(waveform, bottom, top)
        waveform = normalize(waveform)
        
    if target_fs is not None:
        # Standardize sampling rate
        if orig_fs > target_fs:
            waveform = decimate(waveform, int(orig_fs / target_fs))
        elif orig_fs < target_fs:
            waveform = resample(waveform, int(waveform.shape[-1] * (target_fs / orig_fs)))
        waveform = np.squeeze(waveform)

    if abs(sum(np.diff(waveform))) > 0:
        # Did we sample an empty waveform?
        return waveform
    else:
        return None

def load_pkl_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def process_record(input_args):
    i, total_rows, waveform_df, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, max_samples_per_patient, sample_before = input_args

    # Ensure consistent range selected for each patient file
    np.random.seed(i)
    
    try:
        patient_id = waveform_df.iloc[[i]]["patient_id"].item()
        trim_length = waveform_df.iloc[[i]]["trim_length"].item()

        recommended_trim_start_sec = waveform_df.iloc[[i]]["recommended_trim_start_sec"].item()
        recommended_trim_end_sec = waveform_df.iloc[[i]]["recommended_trim_end_sec"].item()
        sample_before_val = int(waveform_df.iloc[[i]][sample_before].item()) if sample_before is not None else None

        cls = waveform_df.iloc[[i]]["outcome"].item()

        info = load_pkl_file(f"{patient_dir}/{patient_id}/info.pkl")


        num_waveforms_processed = {}
        for waveform_type in waveform_types:
            if waveform_type not in num_waveforms_processed:
                num_waveforms_processed[waveform_type] = 0

        waveforms = {}
        for waveform_type in waveform_types:

            waveform_base = np.load(f"{patient_dir}/{patient_id}/{waveform_type}.dat.npy")

            waveform_config = WAVEFORMS_OF_INTERST[waveform_type]
            fs = waveform_config["orig_frequency"]
            if num_waveforms_processed[waveform_type] >= max_samples_per_patient:
                continue

            if waveform_type in info["supported_types"]:
                window_size = int(waveform_length * fs)
                pointer = 0
                attempt = 1
                while num_waveforms_processed[waveform_type] < max_samples_per_patient and attempt < PATIENCE:

                    start_offset = int(max(0, recommended_trim_start_sec * fs))
                    end_offset = int(min(len(waveform_base), recommended_trim_end_sec * fs))
                    if sample_before_val is not None:
                        end_offset = min(end_offset, sample_before_val * fs)

                    pointer = np.random.randint(start_offset, max(1, end_offset - window_size))

                    waveform = get_waveform(waveform_base, pointer, window_size, fs, should_normalize=should_normalize, bandpass_type=waveform_config["bandpass_type"], bandwidth=waveform_config["bandpass_freq"], target_fs=waveform_target_freq)
                    if waveform is None:
                        print(f"[{i}/{total_rows}] {patient_id} waveform was empty at {pointer}. Trying again...")
                        attempt += 1
                        continue

                    print(f"[{i}/{total_rows}] {patient_id} waveform {waveform_type} of shape {waveform.shape} at {pointer}")

                    if waveform_type not in waveforms:
                        waveforms[waveform_type] = []

                    waveforms[waveform_type].append({
                        "record_name": patient_id,
                        "pointer": pointer,
                        "waveform": waveform
                    })

                    pointer += window_size
                    num_waveforms_processed[waveform_type] += 1

        return waveforms
    except Exception as e:
        print("Unexpected error:", e)
        return {}
    

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
    sample_before = args.sample_before
    
    output_folder = f"{output_folder}/{waveform_length}sec-{int(waveform_target_freq)}hz-{int(should_normalize)}norm-{max_samples_per_patient}wpp"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for w in waveform_types:
        Path(f"{output_folder}/{w}").mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_file)
    print(f"Found {input_file} with shape {df.shape}")
    total_rows = len(df)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, row in tqdm(df.iterrows(), disable=True):
            input_args = [i, total_rows, df, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, max_samples_per_patient, sample_before]
            future = executor.submit(process_record, input_args)
            fs.append(future)
            if max_patients is not None and i >= (max_patients - 1):
                break

    waveforms = {}
    for w in waveform_types:
        waveforms[w] = []
    
    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result = future.result(timeout=60*60)
        if result is not None:
            for w in result.keys():
                waveforms[w].extend(result[w])

    for w in waveform_types:
        output_waveforms = []
        with open(f"{output_folder}/{w}/summary.csv", "w") as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            headers = ["record_name", "pointer"]
            writer.writerow(headers)
            for row in waveforms[w]:
                writer.writerow([row["record_name"], row["pointer"]])
                output_waveforms.append(row["waveform"])

        output_tensor = np.array(output_waveforms)
        np.save(f"{output_folder}/{w}/waveforms.dat", output_tensor)
    print(f"Output is written to: {output_folder}/{w}/summary.csv")
    print(f"Output is written to: {output_folder}/{w}/waveforms.dat.npy")
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
    parser.add_argument('-b', '--sample-before',
                        default=None,
                        help='Provide a field name of a column that contains the maximum offset from the waveform start time that waveforms will be sampled from. If none, samples from any part of the waveform in the recommended range')

    args = parser.parse_args()

    run(args)

    print("DONE")
