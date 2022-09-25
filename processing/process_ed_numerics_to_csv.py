#!/usr/bin/env python

"""
Script to convert each pickle file into a CSV file, keeping the numeric hash folder structure.
Filter to those specific to the provided study.
Runs after prepare_ed_numerics_*.py file. 

Example: python process_ed_numerics_to_csv.py -i /deep/group/physiologic-states/v1/processed -o /deep/group/physiologic-states/v1/csv -p 10
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

COLS = [
    "HR",
    "RR",
    "SpO2",
    "NBPs",
    "NBPd",
    "btbRRInt_ms"
]

def process_record(input_args):
    curr_index, input_folder, file_name, output_folder = input_args
    
    try:
        patient_id = file_name.replace(".pkl", "")
        len_info = {}
        for c in COLS:
            len_info[c] = 0
        
        print(f"[{curr_index}] Working on patient={patient_id}")

        # Folders are kept sane by outputting objects into subfolders based on last two digits of CSN
        folder_hash = str(patient_id)[-2:]
        
        with open(f"{input_folder}/{folder_hash}/{patient_id}.pkl", 'rb') as handle:
            b = pickle.load(handle)

            Path(f"{output_folder}/{folder_hash}/{patient_id}").mkdir(parents=True, exist_ok=True)
            for c in COLS:
                with open(f"{output_folder}/{folder_hash}/{patient_id}/{c}.csv", "w") as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(["recorded_time", c])
                    for i in range(len(b[f"{c}-time"])):
                        writer.writerow([b[f"{c}-time"][i], b[c][i]])
                    len_info[c] = len(b[f"{c}-time"])
        return patient_id, len_info
    except Exception as e:
        print("Unexpected error:", e)
        return None, None
    

def run(args):
    print(f"START TIME: {datetime.datetime.now()}")

    input_folder = args.input_folder
    output_folder = args.output_folder
    max_patients = int(args.max_patients) if args.max_patients is not None else None

    break_all = False
    i = 0
    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        rows = []
        for root, dirs, files in os.walk(input_folder):
            if break_all:
                break
            for name in files:
                if name.endswith(".pkl"):
                    input_args = [i, input_folder, name, output_folder]
                    future = executor.submit(process_record, input_args)
                    fs.append(future)
                    if max_patients is not None and i >= (max_patients - 1):
                        break_all= True
                        break
                    i += 1

    with open(f"{output_folder}/summary.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        headers = ["CSN"]
        for c in COLS:
            headers.append(f"{c}-length")
        writer.writerow(headers)
        
        for future in futures.as_completed(fs):
            # Blocking call - wait for 1 hour for a single future to complete
            # (highly unlikely, most likely something is wrong)
            pt, info = future.result(timeout=60*60)
            if pt is not None:
                output_row = [pt]
                for c in COLS:
                    output_row.append(info[c])
                writer.writerow(output_row)  
        
    print(f"Output is written to: {output_folder}/summary.csv")
    print(f"END TIME: {datetime.datetime.now()}")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts PKL to CSV')
    parser.add_argument('-i', '--input-folder',
                        required=True,
                        help='The input folder')
    parser.add_argument('-o', '--output-folder',
                        required=True,
                        help='The output folder')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')

    args = parser.parse_args()

    run(args)

    print("DONE")
