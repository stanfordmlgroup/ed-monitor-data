#!/usr/bin/env python

"""
Script to prepare the MIMIC waveforms, producing a single file containing all the files that can be loaded as a tensor. 

Example: python prepare_mimic_waveforms.py -i /deep/group/mimic3wdb-matched/physionet.org/files/mimic3wdb-matched/1.0/RECORDS-waveforms -o /deep/group/mimic3wdb-matched/files -l 5 -f 500 -w II
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

pd.set_option('display.max_columns', 500)

ROOT_FOLDER = "/deep/group/mimic3wdb-matched/physionet.org/files/mimic3wdb-matched/1.0"
WAVEFORMS_OF_INTERST = {
    "II": {
        "normalize": False,
        "bandpass_type": 'filter',
        "bandpass_freq": [3, 45]
    }, 
    "PLETH": {
        "normalize": False,
        "bandpass_type": None,
        "bandpass_freq": [3, 45]
    },
    "RESP": {
        "normalize": False,
        "bandpass_type": 'cheby2',
        "bandpass_freq": [0.5, 10]
    }
}
LEFT_OFFSET = 1000 # a couple of seconds offset to avoid any irregularities in the first segment
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

def get_waveform(record, start, index, window_sz, should_normalize=False, bandpass_type=None, bandwidth=[3, 45], target_fs=None):
    waveform = record.p_signal[:, index][(start):(start + window_sz)]
    if bandpass_type == "cheby2":
        # Recommended by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6316358/
        b, a = signal.cheby2(4, 40, bandwidth, 'bandpass', fs=record.fs, output='ba')
        waveform = signal.lfilter(b, a, waveform)
    elif bandpass_type == "filter":
        waveform = apply_filter(waveform, filter_bandwidth=bandwidth, fs=record.fs)
    if should_normalize:
        bottom, top = np.percentile(waveform, [1, 99])
        waveform = np.clip(waveform, bottom, top)
        waveform = normalize(waveform)
        
    if target_fs is not None:
        # Standardize sampling rate
        if record.fs > target_fs:
            waveform = decimate(waveform, int(record.fs / target_fs))
        elif record.fs < target_fs:
            waveform = resample(waveform, int(waveform.shape[-1] * (target_fs / record.fs)))
        waveform = np.squeeze(waveform)
        
    if sum(np.diff(waveform)) > 0:
        # Did we sample an empty waveform?
        return waveform
    else:
        return None

def process_record(input_args):
    """
    waveform_id: An id of format "p00/p000020/p000020-2183-04-28-17-47"
    output_dir: Folder to store output files
    """
    i, total_rows, waveform_id, waveform_length, waveform_target_freq, waveform_types, max_samples_per_patient = input_args

    # Ensure consistent range selected for each patient file
    np.random.seed(i)
    
    try:
        record_name = f"{ROOT_FOLDER}/{waveform_id}"
        patient_folder = record_name[:record_name.rindex("/")]
        header = wfdb.rdheader(record_name, rd_segments=True)

        fs = header.fs
        num_waveforms_processed = {}
        for waveform_type in waveform_types:
            if waveform_type not in num_waveforms_processed:
                num_waveforms_processed[waveform_type] = 0

        waveforms = {}
        for segment in header.segments:
            if segment is None:
                continue
            if '_layout' in segment.record_name:
                continue

            for waveform_type in waveform_types:
                if num_waveforms_processed[waveform_type] >= max_samples_per_patient:
                    continue

                waveform_config = WAVEFORMS_OF_INTERST[waveform_type]
                if waveform_type in segment.sig_name and segment.sig_len > (waveform_length * fs):
                    record = wfdb.rdrecord(f"{patient_folder}/{segment.record_name}")
                    if waveform_type in record.sig_name:
                        window_size = int(waveform_length * fs)
                        pointer = 0
                        attempt = 1
                        while num_waveforms_processed[waveform_type] < max_samples_per_patient and attempt < PATIENCE:
                            pointer = np.random.randint(0, max(1, segment.sig_len - window_size))
                            w_index = record.sig_name.index(waveform_type)
                            waveform = get_waveform(record, pointer, w_index, window_size, should_normalize=waveform_config["normalize"], bandpass_type=waveform_config["bandpass_type"], bandwidth=waveform_config["bandpass_freq"], target_fs=waveform_target_freq)
                            if waveform is None:
                                print(f"[{i}/{total_rows}] {segment.record_name} waveform was empty at {pointer}. Trying again...")
                                attempt += 1
                                continue

                            print(f"[{i}/{total_rows}] {segment.record_name} waveform {waveform_type} of shape {waveform.shape}")

                            if waveform_type not in waveforms:
                                waveforms[waveform_type] = []

                            subject_id = waveform_id.split("/")[1].replace("p", "")
                            waveforms[waveform_type].append({
                                "record_name": segment.record_name,
                                "subject_id": subject_id,
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
    waveform_types = args.waveform_types.split(",")
    input_file = args.records_waveform_file
    output_folder = args.output_folder
    max_samples_per_patient = int(args.max_samples_per_patient)
    max_patients = int(args.max_patients) if args.max_patients is not None else None
    waveform_length = int(args.length)
    waveform_target_freq = float(args.frequency)
    waveform_types = waveform_types
    
    output_folder = f"{output_folder}/{waveform_length}sec-{int(waveform_target_freq)}hz-{max_samples_per_patient}wpp"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for w in waveform_types:
        Path(f"{output_folder}/{w}").mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_file, header=None)
    print(f"Found {input_file} with shape {df.shape}")
    total_rows = len(df)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, row in tqdm(df.iterrows(), disable=True):
            input_args = [i, total_rows, row[0], waveform_length, waveform_target_freq, waveform_types, max_samples_per_patient]
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
            headers = ["subject_id", "record_name", "pointer"]
            writer.writerow(headers)
            for row in waveforms[w]:
                writer.writerow([row["subject_id"], row["record_name"], row["pointer"]])
                output_waveforms.append(row["waveform"])

        output_tensor = torch.FloatTensor(output_waveforms)
        torch.save(output_tensor, f"{output_folder}/{w}/waveforms.pt")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Matches a cohort file with a Philips bed summary file')
    parser.add_argument('-i', '--records-waveform-file',
                        required=True,
                        help='The path to the RECORDS-waveforms file')
    parser.add_argument('-o', '--output-folder',
                        required=True,
                        help='Folder where the output files will be written')
    parser.add_argument('-l', '--length',
                        default=5,
                        help='Length of the output waveform (sec)')
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

    args = parser.parse_args()

    run(args)

    print("DONE")
