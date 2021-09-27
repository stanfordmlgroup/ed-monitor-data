#!/usr/bin/env python

"""
Script to prepare the MIMIC waveforms, specifically designed to produce spectrogram files. 

Example: python prepare_mimic_waveforms_spec.py -i /deep/group/mimic3wdb-matched/physionet.org/files/mimic3wdb-matched/1.0/RECORDS-waveforms -o /deep/group/mimic3wdb-matched/processed -l 15 -w II
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
from pathlib import Path
from concurrent import futures
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import wfdb
import cv2
from scipy import signal
from pathlib import Path
from biosppy.signals.tools import filter_signal
import numpy as np
from ssqueezepy import cwt, ssq_cwt, ssq_stft

pd.set_option('display.max_columns', 500)

ROOT_FOLDER = "/deep/group/mimic3wdb-matched/physionet.org/files/mimic3wdb-matched/1.0"
WAVEFORMS_OF_INTERST = {
    "II": {
        "target_fs": 500,
        "normalize": True,
        "bandpass_type": 'filter',
        "bandpass_freq": [3, 45]
    }, 
    "PLETH": {
        "target_fs": 125,
        "normalize": True,
        "bandpass_type": None,
        "bandpass_freq": [3, 45]
    },
    "RESP": {
        "target_fs": 62.5,
        "normalize": True,
        "bandpass_type": 'cheby2',
        "bandpass_freq": [0.5, 10]
    }
}
LEFT_OFFSET = 1000 # a couple of seconds offset to avoid any irregularities in the first segment
MAX_SAMPLES_PER_PATIENT = 1


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

def get_waveform(record, start, index, window_sz, should_normalize=True, bandpass_type=None, bandwidth=[3, 45], target_fs=None):
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
        target_width = int(target_fs * (len(waveform) / record.fs))
        waveform = np.squeeze(cv2.resize(waveform, dsize=(1, target_width)))
    return waveform

def process_record(input_args):
    """
    waveform_id: An id of format "p00/p000020/p000020-2183-04-28-17-47"
    output_dir: Folder to store output files
    """
    i, total_rows, waveform_id, output_folder, waveform_length, waveform_types, output_type = input_args
    
    try:

        # Ensure consistent range selected for each patient file
        np.random.seed(i)

        record_name = f"{ROOT_FOLDER}/{waveform_id}"
        patient_folder = record_name[:record_name.rindex("/")]
        header = wfdb.rdheader(record_name, rd_segments=True)
    #     display(header.__dict__)

        fs = header.fs
    #     print(header.segments[2].__dict__)
        num_waveforms_processed = {}
        for waveform_type in waveform_types:
            if waveform_type not in num_waveforms_processed:
                num_waveforms_processed[waveform_type] = 0

        pointer_offset = None
        waveforms = {}
        for segment in header.segments:
            if segment is None:
                continue
            if '_layout' in segment.record_name:
                continue
    #         print(segment.__dict__)

            for waveform_type in waveform_types:
                if MAX_SAMPLES_PER_PATIENT is not None and num_waveforms_processed[waveform_type] >= MAX_SAMPLES_PER_PATIENT:
                    continue
            
                waveform_config = WAVEFORMS_OF_INTERST[waveform_type]
                if waveform_type in segment.sig_name and segment.sig_len > (waveform_length * fs):
    #                 print(f"{patient_folder}/{segment.record_name}")

                    record = wfdb.rdrecord(f"{patient_folder}/{segment.record_name}")
                    if waveform_type in record.sig_name:
                        # Found a good waveform to process

                        window_size = int(waveform_length * fs)
                        if MAX_SAMPLES_PER_PATIENT == 1:
                            if pointer_offset is not None:
                                pointer = pointer_offset
                            else:
                                # We will actually choose a random waveform
                                pointer = np.random.randint(0, max(1, segment.sig_len - window_size))
                                pointer_offset = pointer
                        else:
                            # Otherwise, we build consecutive windows starting from the left
                            pointer = LEFT_OFFSET
                        while (pointer + window_size) < segment.sig_len:
                            w_index = record.sig_name.index(waveform_type)
                            waveform = get_waveform(record, pointer, w_index, window_size, should_normalize=waveform_config["normalize"], bandpass_type=waveform_config["bandpass_type"], bandwidth=waveform_config["bandpass_freq"], target_fs=waveform_config["target_fs"])
                            print(f"[{i}/{total_rows}] {segment.record_name} waveform {waveform_type} of shape {waveform.shape}")
                            
                            if output_type == "NPY":
                                np.save(f"{output_folder}/{waveform_type}/files/{segment.record_name}-{pointer}.dat", waveform)
                                if waveform_type not in waveforms:
                                    waveforms[waveform_type] = []

                                waveforms[waveform_type].append({
                                    "name": f"{segment.record_name}-{pointer}.dat",
                                    "size": len(waveform)
                                })
                            elif output_type == "SPEC":
                                Twxo, Wxo, *_ = ssq_cwt(waveform, fs=waveform_config["target_fs"])
                                Wxo = cv2.resize(np.abs(Wxo), dsize=(Wxo.shape[0], Wxo.shape[0]))
                                np.save(f"{output_folder}/{waveform_type}/files/{segment.record_name}-{pointer}.dat", Wxo)
                                if waveform_type not in waveforms:
                                    waveforms[waveform_type] = []

                                waveforms[waveform_type].append({
                                    "name": f"{segment.record_name}-{pointer}.dat",
                                    "size": len(waveform)
                                })
                            else:
                                raise Exception("Not supported output_type")

                            pointer += window_size
                            num_waveforms_processed[waveform_type] += 1
                            if MAX_SAMPLES_PER_PATIENT is not None and num_waveforms_processed[waveform_type] >= MAX_SAMPLES_PER_PATIENT:
                                break

        return waveforms
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return {}

def run(args):
    waveform_types = args.waveform_types.split(",")
    input_file = args.records_waveform_file
    output_folder = args.output_folder
    waveform_length = int(args.length)
    waveform_types = waveform_types
    output_type = args.output_type
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for w in waveform_types:
        Path(f"{output_folder}/{w}/files").mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_file, header=None)
    print(f"Found {input_file} with shape {df.shape}")
    total_rows = len(df)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, row in tqdm(df.iterrows(), disable=True):
            input_args = [i, total_rows, row[0], output_folder, waveform_length, waveform_types, output_type]
            future = executor.submit(process_record, input_args)
            fs.append(future)
#             if i > 10:
#                 break

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
        output_file = f"{output_folder}/{w}/summary.csv"
        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            headers = ["waveform", "size"]
            writer.writerow(headers)
            for row in waveforms[w]:
                writer.writerow([row["name"], row["size"]])

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
                        required=True,
                        help='Length of the output waveform (sec)')
    parser.add_argument('-w', '--waveform-types',
                        required=True,
                        help='Comma separated list of waveform types to process. Supported values: II, PLETH, RESP')
    parser.add_argument('-t', '--output-type',
                        required=False,
                        default="SPEC",
                        help='The type of output. Supported values: NPY, SPEC')

    args = parser.parse_args()

    run(args)

    print("DONE")
