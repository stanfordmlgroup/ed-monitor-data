#!/usr/bin/env python

"""
Script to extract the numerics file given the original visits. Used if you don't need to match against the actual waveform files (e.g. if RESP waveforms are not actually available, you can still retrieve the numerics file).

Files are written out to:
- /deep/group/physiologic-states/v1/processed/<hash>/<CSN>.pkl
where <hash> is the last two characters of the CSN

Example: python prepare_ed_numerics_from_matched_cohort.py -i /deep/group/physiologic-states/v1/matched-cohort.csv -d /deep/group/ed-monitor/2020_08_23_2020_09_23,/deep/group/ed-monitor/2020_09_23_2020_11_30,/deep/group/ed-monitor/2020_11_30_2020_12_31,/deep/group/ed-monitor/2021_01_01_2021_01_31,/deep/group/ed-monitor/2021_02_01_2021_02_28,/deep/group/ed-monitor/2021_03_01_2021_03_31,/deep/group/ed-monitor/2021_04_01_2021_05_12,/deep/group/ed-monitor/2021_05_13_2021_05_31,/deep/group/ed-monitor/2021_06_01_2021_06_30,/deep/group/ed-monitor/2021_07_01_2021_07_31,/deep/group/ed-monitor/2021_06_01_2021_06_30,/deep/group/ed-monitor/2021_08_01_2021_09_16 -o /deep/group/physiologic-states/v1/test -p 100
"""

import argparse
import datetime
import math
import os
import os.path
import pickle
import shutil
from concurrent import futures
from os import listdir
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from tqdm import tqdm

pd.set_option('display.max_columns', 500)

COLUMNS = [
    "HR",
    "SpO2",
    "RR",
    "NBPs",
    "NBPd",
    "btbRRInt_ms",
    "Perf"
]

def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def load_file(study_to_patient_dir, study):
    output_files = []

    if study in study_to_patient_dir:
        folder_path = study_to_patient_dir[study]

        # Note:
        # - `folder_path` is expected to be in the following format: /deep/group/ed-monitor/2020_08_23_2020_09_23
        # - However, studies are actually located at: /deep/group/ed-monitor/2020_08_23_2020_09_23/data/2020_08_23_2020_09_23/STUDY-XXXXXXX
        #
        actual_date_range = folder_path.split("/")[-1]
        folder_path = os.path.join(folder_path, f"data/{actual_date_range}")
        study_folder = os.path.join(folder_path, study)

        if os.path.isdir(study_folder):
            for f in sorted(listdir(study_folder)):
                # print(f"Considering file {f} in {study_folder}")
                if f.endswith("numerics.csv") and is_non_zero_file(f"{study_folder}/{f}"):
                    print(f"FOUND FILE {f}")
                    output_files.append(
                        pd.read_csv(f"{study_folder}/{f}").rename(columns=lambda x: x.strip()).replace(r'^\s*$', np.nan,
                                                                                                       regex=True))
    else:
        print(f"Could not determine where study {study} is located")
    return output_files


def process_numerics_file(curr_index, total_rows, patient_id, patient_dirs, study_to_patient_dir, studies, start, end):
    output_vals = {}
    
    print(f"process_numerics_file start = {start} end = {end}")

    for col in COLUMNS:
        output_vals[col] = []
        output_vals[f"{col}-time"] = []

    for study in sorted(studies):
        for df in load_file(study_to_patient_dir, study):
            if df is None:
                # There are no numerics for some reason
                print(
                    f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file with study {study} does not exist!")
                continue

            for i, row in df.iterrows():
                # Data is only available spuriously, so collect all measures as we can between start/end

                date_str = row["Date"].strip() + " " + row["Time"].strip()
                row_time = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
                if row_time < start or row_time > end:
                    # Row is out of our study range
                    continue

                for col in COLUMNS:
                    if col in row:
                        if isinstance(row[col], str):
                            output_vals[col].append(float(row[col].strip()))
                            output_vals[f"{col}-time"].append(row_time)
                        elif isinstance(row[col], float) and not math.isnan(row[col]):
                            output_vals[col].append(row[col])
                            output_vals[f"{col}-time"].append(row_time)

    is_non_empty = False
    non_empty_len = 0
    for col in COLUMNS:
        if len(output_vals[col]) > 0:
            is_non_empty = True
            non_empty_len = len(output_vals[col])
    if is_non_empty:
        print(
            f"[{curr_index}/{total_rows}] [{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file with studies {studies} has len {non_empty_len}")
        return output_vals, patient_id
    else:
        print(
            f"[{curr_index}/{total_rows}] [{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file with studies {studies} is empty!")
        return None, None


def process_record(input_args):
    i, total_rows, row, patient_dirs, study_to_patient_dir, output_folder, existing_output_folder = input_args

    # Ensure consistent range selected for each patient file
    np.random.seed(i)

    try:
        patient_id = row["CSN"]
        print(f"[{i}/{total_rows}] Working on patient={patient_id}")

        # Keep folders sane by outputting objects into subfolders based on last two digits of CSN
        folder_hash = str(patient_id)[-2:]

        # Does the file already exist? If so, we can just copy it
        #

        # Check current folder
        output_hash_folder = f"{output_folder}/{folder_hash}"
        Path(output_hash_folder).mkdir(parents=True, exist_ok=True)
        output_file = f"{output_hash_folder}/{patient_id}.pkl"
        print(f"Checking for output_file {output_file}")
        if os.path.isfile(output_file):
            # If folder already exists, just skip
            print(
                f"[{i}/{total_rows}] [{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file already exists")
            return None, None

        # Check existing folder
        if existing_output_folder is not None:
            existing_output_hash_folder = f"{existing_output_folder}/{folder_hash}"
            existing_file = f"{existing_output_hash_folder}/{patient_id}.pkl"
            if os.path.isfile(existing_file):
                # If file already exists, just copy it over
                shutil.copyfile(existing_file, output_file)
                print(
                    f"[{i}/{total_rows}] [{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Copied numerics file from {existing_file} to {output_file}")
                return None, None

        roomed_time = row["Roomed_time"]
        dispo_time = row["Dispo_time"]
        studies = row["final_studies"].split(",")

        roomed_time = datetime.datetime.strptime(roomed_time, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = datetime.datetime.strptime(dispo_time, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)

        # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)

        numerics, pt = process_numerics_file(i, total_rows, patient_id, patient_dirs, study_to_patient_dir, studies,
                                             roomed_time, dispo_time)

        print(
            f"[{i}/{total_rows}] [{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Trying to Pickle")
        if numerics is not None:
            with open(f"{output_hash_folder}/{patient_id}.pkl", 'wb') as handle:
                pickle.dump(numerics, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(
                f"[{i}/{total_rows}] [{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Pickle dumped!")
        return numerics, pt
    except Exception as e:
        print("Unexpected error:", e)
        # raise e
        return None, None


def run(args):
    print(f"START TIME: {datetime.datetime.now()}")
    consolidated_file = args.consolidated_file
    patient_dirs = args.patient_dirs.split(",")
    output_folder = args.output_folder
    existing_output_folder = args.existing_output_folder
    existing_log_file = args.existing_log_file
    start_date = args.start_date
        
    max_patients = int(args.max_patients) if args.max_patients is not None else None

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    csns_already_processed = set()
    if existing_log_file is not None:
        with open(existing_log_file, "r") as f:
            for row in f:
                if "Numerics file already exists" in row or "Pickle dumped!" in row or "Copied numerics file" in row:
                    csn = row.replace("[", "").replace("]", "").split(" ")[1].strip()
                    csns_already_processed.add(csn)
                    print(f"Found already processed CSN: {csn}")
                elif "Found already processed CSN: " in row:
                    csn = row.replace("Found already processed CSN: ", "").strip()
                    csns_already_processed.add(csn)
                    print(f"Found already processed CSN: {csn}")
    print(f"Found {len(csns_already_processed)} csns_already_processed")

    study_to_patient_dir = {}
    for folder_path in patient_dirs:
        files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        for f in files:
            df = pd.read_csv(f"{folder_path}/{f}")
            for i, row in df.iterrows():
                study_to_patient_dir[row["StudyId"]] = folder_path
    print(f"Found {len(study_to_patient_dir)} study_to_patient_dir")

    df = pd.read_csv(consolidated_file)
    print(f"Found {consolidated_file} with shape {df.shape}")
    
    if start_date is not None:
        print(f"Start date specified - filtering to only those lines after {start_date}")
        df = df[df["Arrival_time"] >= start_date]
        print(f"Dataframe now filtered to have shape {df.shape}")

    total_rows = len(df)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, row in tqdm(df.iterrows(), disable=True):
            input_args = [len(fs), total_rows, row, patient_dirs, study_to_patient_dir, output_folder, existing_output_folder]
            csn = str(row["CSN"])
            if csn in csns_already_processed:
                print(f"Skipping {csn} as it was already processed")
                continue
            future = executor.submit(process_record, input_args)
            # process_record(input_args)
            fs.append(future)
            if max_patients is not None and len(fs) >= (max_patients - 1):
                break

    patients_with_data = 0
    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result, pt = future.result(timeout=60 * 60)
        if result is not None:
            patients_with_data += 1

    print(f"Found patients_with_data={patients_with_data}")
    print(f"END TIME: {datetime.datetime.now()}")


#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare the numerics files from the patient dir')
    parser.add_argument('-i', '--consolidated-file',
                        required=True,
                        help='The path to the consolidated file')
    parser.add_argument('-d', '--patient-dirs',
                        required=True,
                        help='The path to the patient dirs')
    parser.add_argument('-e', '--existing-output-folder',
                        default=None,
                        help='Folder where the output files might already exist (speeds up processing if processing new version)')
    parser.add_argument('-l', '--existing-log-file',
                        default=None,
                        help='If the script is terminated, you can pass in the log file of a previous run to have this script continue off from where it left off.')
    parser.add_argument('-o', '--output-folder',
                        required=True,
                        help='Folder where the output files will be written')
    parser.add_argument('-s', '--start-date',
                        default=None,
                        help='ISO8601-formatted datetime to start processing from. This is useful if we have an updated consolidated with new data. e.g. 2020-08-01T00:06:00Z')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')

    args = parser.parse_args()

    run(args)

    print("DONE")
