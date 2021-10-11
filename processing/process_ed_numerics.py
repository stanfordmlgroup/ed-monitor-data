#!/usr/bin/env python

"""
Script to consolidate numeric values from the numeric hash folders. 
Filter to those specific to the provided study.
Runs after prepare_ed_numerics_*.py file. 

Example: python process_ed_numerics.py -i /deep/group/lactate/v1/consolidated.filtered.csv -f /deep/group/physiologic-states/v1/processed -o /deep/group/lactate/v1 -b Collection_time_1 -p 10
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
from concurrent import futures
import argparse
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
from pathlib import Path
import datetime
import os.path
from os import listdir
from os.path import isfile, join

pd.set_option('display.max_columns', 500)

def process_record(input_args):
    curr_index, total_rows, waveform_df, input_folder, before_col = input_args

    # Ensure consistent range selected for each patient file
    np.random.seed(curr_index)
    
    try:
        patient_id = waveform_df.iloc[[curr_index]]["patient_id"].item()
        print(f"[{curr_index}/{total_rows}] Working on patient={patient_id}")

        # Folders are kept sane by outputting objects into subfolders based on last two digits of CSN
        folder_hash = str(patient_id)[-2:]
        output_hash_folder = f"{input_folder}/{folder_hash}"

        index_to_sample_before_hr = 0
        index_to_sample_before_rr = 0
        index_to_sample_before_spo2 = 0
        
        with open(f"{output_hash_folder}/{patient_id}.pkl", 'rb') as handle:
            b = pickle.load(handle)

            if before_col is not None:
                cutoff_time = waveform_df.iloc[[curr_index]][before_col].item()
                cutoff_time = datetime.datetime.strptime(cutoff_time, '%Y-%m-%dT%H:%M:%S%z').replace(tzinfo=None)
                # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
                cutoff_time = pytz.timezone('America/Vancouver').localize(cutoff_time)

                for i in range(len(b["HR-time"])):
                    if b["HR-time"][i] <= cutoff_time:
                        index_to_sample_before_hr = i
                    else:
                        break
                for i in range(len(b["RR-time"])):
                    if b["RR-time"][i] <= cutoff_time:
                        index_to_sample_before_rr = i
                    else:
                        break
                for i in range(len(b["SpO2-time"])):
                    if b["SpO2-time"][i] <= cutoff_time:
                        index_to_sample_before_spo2 = i
                    else:
                        break

                print(f"[{curr_index}/{total_rows}] Truncating patient={patient_id} at hr={index_to_sample_before_hr}/{len(b['HR'])} rr={index_to_sample_before_rr}/{len(b['RR'])} spo2={index_to_sample_before_spo2}/{len(b['SpO2'])}")

            else:
                index_to_sample_before_hr = None
                index_to_sample_before_rr = None
                index_to_sample_before_spo2 = None

            info = {
                "index_to_sample_before_hr": index_to_sample_before_hr,
                "index_to_sample_before_rr": index_to_sample_before_rr,
                "index_to_sample_before_spo2": index_to_sample_before_spo2
            }

            return patient_id, b["HR"][:index_to_sample_before_hr], b["RR"][:index_to_sample_before_rr], b["SpO2"][:index_to_sample_before_spo2], info

    except Exception as e:
        print("Unexpected error:", e)
        return None, None, None, None, None
    

def run(args):
    print(f"START TIME: {datetime.datetime.now()}")
    consolidated_file = args.consolidated_file
    input_folder = args.input_folder
    output_folder = args.output_folder
    before_col = args.before_col
    max_patients = int(args.max_patients) if args.max_patients is not None else None
    
    df = pd.read_csv(consolidated_file)
    print(f"Found {consolidated_file} with shape {df.shape}")
    total_rows = len(df)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, row in tqdm(df.iterrows(), disable=True):
            input_args = [i, total_rows, df, input_folder, before_col]
            future = executor.submit(process_record, input_args)
            fs.append(future)
            if max_patients is not None and i >= (max_patients - 1):
                break

    output_obj = {
        "patient_ids": [],
        "hr": [],
        "rr": [],
        "spo2": []
    }
    
    with open(f"{output_folder}/numerics-summary.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["csn", "index_to_sample_before_hr", "index_to_sample_before_rr", "index_to_sample_before_spo2"])
        for future in futures.as_completed(fs):
            # Blocking call - wait for 1 hour for a single future to complete
            # (highly unlikely, most likely something is wrong)
            pt, hr, rr, spo2, info = future.result(timeout=60*60)
            if pt is not None:
                output_obj["patient_ids"].append(pt)
                output_obj["hr"].append(hr)
                output_obj["rr"].append(rr)
                output_obj["spo2"].append(spo2)

                writer.writerow([pt, info["index_to_sample_before_hr"], info["index_to_sample_before_rr"], info["index_to_sample_before_spo2"]])

    with open(f"{output_folder}/numerics-processed.pkl", 'wb') as handle:
        pickle.dump(output_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        
    print(f"Output is written to: {output_folder}/numerics-processed.pkl")
    print(f"Output is written to: {output_folder}/numerics-summary.csv")
    print(f"END TIME: {datetime.datetime.now()}")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a consolidated file, extracts the numerics file from the patient dir to create a single file')
    parser.add_argument('-i', '--consolidated-file',
                        required=True,
                        help='The path to the consolidated file')
    parser.add_argument('-f', '--input-folder',
                        required=True,
                        help='The input folder')
    parser.add_argument('-o', '--output-folder',
                        required=True,
                        help='The output folder')
    parser.add_argument('-b', '--before-col',
                        default=None,
                        required=False,
                        help='The column in the consolidated file before which the numerics should be selected from')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')

    args = parser.parse_args()

    run(args)

    print("DONE")
