#!/usr/bin/env python

"""
Script to find waht leads are available for a given patient population

Usage:
- python -u /deep/u/tomjin/ed-monitor-data/processing/find_available_leads.py "/deep/group/ed-monitor/patient_data_v9/matched-cohort.csv" "/deep/group/ed-monitor/patient_data_v9/matched-export.csv" "/deep/group/ed-monitor/patient_data_v9/lead-availability.csv"

"""

import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import pytz
import re
import csv
import matplotlib.pyplot as plt
import math
from pathlib import Path
from concurrent import futures

pd.set_option('display.max_columns', 500)

LIMIT = None

def process_study(input_args):
    curr_patient_index, total_patients, patient_id, studies, patient_to_actual_times, patient_to_row, study_to_info = input_args
    min_start = None
    max_end = None

    print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Starting patient processing {curr_patient_index}/{total_patients}...")

    if patient_id not in patient_to_actual_times:
        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Patient {patient_id} skipped because there was no end dispo")
        return {}

    waveform_types = set()
    for study in studies:
        info = study_to_info[study]
        study_path = info["study_folder"]

        if not os.path.isdir(study_path):
            print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Patient {patient_id} skipped because study path not found {study_path}")
            break

        for f in os.listdir(study_path):
            if f.endswith("dat"):
                print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Patient {patient_id} found file {f}")
                prefix = f.split(".")[0]  # e.g. STUDY-028806_2020-09-19_00-00-00
                actual_type = f.split(".")[-2]  # e.g. II
                waveform_types.add(actual_type)

    return {
        "patient_id": patient_id,
        "waveform_types": waveform_types
    }


def process_studies(patient_to_actual_times, patient_to_studies, patient_to_row, study_to_info):
    patient_id_to_results = {}

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        i = 0
        for patient_id, studies in tqdm(patient_to_studies.items(), disable=True):
            i += 1
            if LIMIT is not None and i > LIMIT:
                break

            input_args = [i, len(patient_to_studies), patient_id, studies, patient_to_actual_times,
                          patient_to_row, study_to_info]
            future = executor.submit(process_study, input_args)
            fs.append(future)

    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result = future.result(timeout=60*60)
        if result is not None:
            if "patient_id" in result:
                patient_id_to_results[result["patient_id"]] = result
            else:
                # The patient was skipped so we can ignore it
                pass
    return patient_id_to_results


def load_mapping_file(mapping_file):
    df = pd.read_csv(mapping_file)

    patient_to_row = {}
    patient_to_actual_times = {}
    for index, row in df.iterrows():
        roomed_time = datetime.datetime.strptime(row["Roomed_time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        if str(row["Dispo_time"]) == "nan":
            continue
        dispo_time = datetime.datetime.strptime(row["Dispo_time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)

        arrival_time = datetime.datetime.strptime(row["Arrival_time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
        arrival_time = pytz.timezone('America/Vancouver').localize(arrival_time)

        case_id = row["CSN"]
        patient_to_actual_times[case_id] = {
            "roomed_time": roomed_time,
            "dispo_time": dispo_time,
            "arrival_time": arrival_time
        }
        patient_to_row[case_id] = row

    print(f"Total patients in cohort file = {len(patient_to_actual_times)}")
    return patient_to_actual_times, patient_to_row


def load_exports_file(exports_file):
    df = pd.read_csv(exports_file)

    patient_to_studies = {}
    patient_to_row = {}
    study_to_info = {}
    study_to_patient = {}
    for index, row in df.iterrows():
        study = row["StudyId"]
        case = row["CSN"]
        folder_path = row["path"]
        start_time = row["StartTime"]
        end_time = row["EndTime"]
        patient_to_row[case] = row

        start_time = datetime.datetime.strptime(start_time, '%m/%d/%y %H:%M:%S')
        start_time = start_time.astimezone(pytz.timezone('America/Vancouver'))
        end_time = datetime.datetime.strptime(end_time, '%m/%d/%y %H:%M:%S')
        end_time = end_time.astimezone(pytz.timezone('America/Vancouver'))
        
        # Note:
        # - `folder_path` is expected to be in the following format: /deep/group/ed-monitor/2020_08_23_2020_09_23
        # - However, studies are actually located at: /deep/group/ed-monitor/2020_08_23_2020_09_23/data/2020_08_23_2020_09_23/STUDY-XXXXXXX
        #
        actual_date_range = folder_path.split("/")[-1]
        folder_path = os.path.join(folder_path, f"data/{actual_date_range}")
        study_folder = os.path.join(folder_path, study)

        if case not in patient_to_studies:
            patient_to_studies[case] = []
        patient_to_studies[case].append(study)
        study_to_info[study] = {
            "export_start_time": start_time,
            "export_end_time": end_time,
            "study_folder": study_folder
        }
        if study not in study_to_patient:
            study_to_patient[study] = case
        elif study_to_patient[study] != case:
            print(f"Found same study {study} used for multiple patients: {case}, {study_to_patient[study]}")

    print("---")
    print(
        f"Total patients in mapped file = {len(patient_to_studies)}")

    return patient_to_studies, study_to_info


if __name__ == '__main__':
    # Mapping file contains the original cohort information. It is primarily used here to retrieve basic information on the patient.
    mapping_file = sys.argv[1]

    # Exports file contains the patient and STUDY ID relationship.
    exports_file = sys.argv[2]

    # Where the output summary should be written to
    output_file = sys.argv[3]

    if len(sys.argv) > 4:
        LIMIT = int(sys.argv[4])

    print("=" * 30)
    print(f"Starting file availability check with mapping_file={mapping_file}, exports_file={exports_file}, limit={LIMIT}")
    print("-" * 30)

    patient_to_actual_times, patient_to_row = load_mapping_file(mapping_file)
    patient_to_studies, study_to_info = load_exports_file(exports_file)
    patient_id_to_results = process_studies(patient_to_actual_times, patient_to_studies, patient_to_row, study_to_info)

    with open(output_file, "w") as f:
        f.write("csn,waveforms\n")
        for k, v in patient_id_to_results.items():
            output_waveforms = "\t".join(v["waveform_types"])
            f.write(f"{k},{output_waveforms}\n")

    print("DONE")
