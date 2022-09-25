#!/usr/bin/env python

"""
Script to extract the numerics file given the consolidated file. Outputs:
- A summary.csv file containing the CSNs and length of data that was extracted
- Folders containing the actual pickle object with the extracted data for each patient. Note that each patient pickle object is stored under subfolders based on the last two digits of the CSN.

Example: python prepare_ed_numerics_from_consolidated.py -i /deep/group/pulmonary-embolism/v2/consolidated.filtered.csv -d /deep/group/ed-monitor/2020_08_23_2020_09_23,/deep/group/ed-monitor/2020_09_23_2020_11_30,/deep/group/ed-monitor/2020_11_30_2020_12_31,/deep/group/ed-monitor/2021_01_01_2021_01_31,/deep/group/ed-monitor/2021_02_01_2021_02_28,/deep/group/ed-monitor/2021_03_01_2021_03_31,/deep/group/ed-monitor/2021_04_01_2021_05_12,/deep/group/ed-monitor/2021_05_13_2021_05_31,/deep/group/ed-monitor/2021_06_01_2021_06_30,/deep/group/ed-monitor/2021_07_01_2021_07_31 -o /deep/group/pulmonary-embolism/v2 -p 3

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
import os.path
from os import listdir
from os.path import isfile, join

pd.set_option('display.max_columns', 500)

COLUMNS = [
    "HR",
    "btbHR",
    "btbRRInt_ms",
    "SpO2",
    "RR"
]

def load_file(patient_dirs, study):
    output_files = []
    for folder_path in patient_dirs:
        # We need to look into multiple patient directories to find out where the study is actually located
        
        # Note:
        # - `folder_path` is expected to be in the following format: /deep/group/ed-monitor/2020_08_23_2020_09_23
        # - However, studies are actually located at: /deep/group/ed-monitor/2020_08_23_2020_09_23/data/2020_08_23_2020_09_23/STUDY-XXXXXXX
        #
        actual_date_range = folder_path.split("/")[-1]
        folder_path = os.path.join(folder_path, f"data/{actual_date_range}")
        study_folder = os.path.join(folder_path, study)
        
        if os.path.isdir(study_folder):
            for f in sorted(listdir(study_folder)):
                if f.endswith("numerics.csv"):
                    output_files.append(pd.read_csv(f"{study_folder}/{f}").rename(columns=lambda x: x.strip()).replace(r'^\s*$', np.nan, regex=True))
    return output_files


def process_numerics_file(patient_id, patient_dirs, studies, start, end):
    output_vals = {}

    output_vals["time"] = []
    for col in COLUMNS:
        output_vals[col] = []
    
    for study in sorted(studies):
        for df in load_file(patient_dirs, study):
            if df is None:
                # There are no numerics for some reason
                print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file with study {study} does not exist!")
                continue

            for i, row in df.iterrows():
                # Data is only available spuriously, so collect all measures as we can between start/end

                date_str = row["Date"].strip() + " " + row["Time"].strip()
                row_time = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
                if row_time < start or row_time > end:
                    # Row is out of our study range
                    continue

                output_vals["time"].append(row_time)
                for col in COLUMNS:
                    if col in row:
                        if isinstance(row[col], str):
                            output_vals[col].append(float(row[col].strip()))
                        elif isinstance(row[col], float) and not math.isnan(row[col]):
                            output_vals[col].append(row[col])
    
    if len(output_vals[COLUMNS[0]]) == 0:
        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file with studies {studies} is empty!")
    
    return output_vals, patient_id


def process_record(input_args):
    i, total_rows, waveform_df, patient_dirs = input_args

    # Ensure consistent range selected for each patient file
    np.random.seed(i)
    
    try:
        patient_id = waveform_df.iloc[[i]]["patient_id"].item()
        print(f"[{i}/{total_rows}] Working on patient={patient_id}")
    
        waveform_start_time = waveform_df.iloc[[i]]["roomed_time"].item()
        waveform_end_time = waveform_df.iloc[[i]]["dispo_time"].item()
        studies = waveform_df.iloc[[i]]["studies"].item().split(",")
        
        waveform_start_time = datetime.datetime.strptime(waveform_start_time, '%Y-%m-%d %H:%M:%S%z')
        waveform_end_time = datetime.datetime.strptime(waveform_end_time, '%Y-%m-%d %H:%M:%S%z')

        numerics, pt = process_numerics_file(patient_id, patient_dirs, studies, waveform_start_time, waveform_end_time)

        return numerics, pt
    except Exception as e:
        print("Unexpected error:", e)
        return None, None
    

def run(args):
    print(f"START TIME: {datetime.datetime.now()}")
    consolidated_file = args.consolidated_file
    patient_dirs = args.patient_dirs.split(",")
    output_folder = args.output_folder
    max_patients = int(args.max_patients) if args.max_patients is not None else None

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(consolidated_file)
    print(f"Found {consolidated_file} with shape {df.shape}")
    total_rows = len(df)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, row in tqdm(df.iterrows(), disable=True):
            input_args = [i, total_rows, df, patient_dirs]
            future = executor.submit(process_record, input_args)
            fs.append(future)
            if max_patients is not None and i >= (max_patients - 1):
                break

    output_obj = {
        "patient_ids": [],
        "time": []
    }
    for col in COLUMNS:
        # Will be uneven in length
        output_obj[col] = []
    
    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result, pt = future.result(timeout=60*60)
        if result is not None:
            for w in result.keys():
                output_obj[w].append(result[w])
            output_obj["patient_ids"].append(pt)

    with open(f"{output_folder}/numerics.pkl", 'wb') as handle:
        pickle.dump(output_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        
    print(f"Output is written to: {output_folder}/numerics.pkl")
    print(f"END TIME: {datetime.datetime.now()}")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a consolidated file, extracts the numerics file from the patient dir to create a single file')
    parser.add_argument('-i', '--consolidated-file',
                        required=True,
                        help='The path to the consolidated file')
    parser.add_argument('-d', '--patient-dirs',
                        required=True,
                        help='The path to the patient dirs')
    parser.add_argument('-o', '--output-folder',
                        required=True,
                        help='Folder where the output files will be written')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')

    args = parser.parse_args()

    run(args)

    print("DONE")
