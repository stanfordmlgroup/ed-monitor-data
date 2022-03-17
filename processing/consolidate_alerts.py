#!/usr/bin/env python

"""
Script to consolidate all the alerts into a single file.

Usage:
- python -u /deep/u/tomjin/ed-monitor-data/processing/consolidate_alerts.py -m /deep/group/ed-monitor-self-supervised/v3/matched-cohort.csv -e /deep/group/ed-monitor-self-supervised/v3/matched-export.csv -f /deep/group/ed-monitor-self-supervised/v3/alerts.csv -l 3

"""

import argparse
import datetime
import pandas as pd
from tqdm import tqdm
import os
import pytz
import time
import csv
from concurrent import futures

DEBUG = False
ALERT_COLUMNS = ["ClinicalUnit", "Bed", "StudyId", "AnnounceDate", "AnnounceTime", "OnsetDate", "OnsetTime", "Label", "Source", "Code", "Severity", "Kind", "SubTypeId", "EndDate", "EndTime", "AlarmBurden", "IsSilenced", "SilenceCount", "SilenceTimes"]
ALERT_COL_TO_INDEX = {}
for i, v in enumerate(ALERT_COLUMNS):
    ALERT_COL_TO_INDEX[v] = i

def strip_row(row):
    return [r.strip() for r in row]

def load_alerts_file(study_to_study_folder, study, start, end):
    output_files = []
    
    start_date = start.date()
    end_date = end.date()

    if study in study_to_study_folder:
        folder_path = study_to_study_folder[study]

        if os.path.isdir(folder_path):
            for f in sorted(os.listdir(folder_path)):
                if f.endswith("alerts.csv"):
                    # print(f"Reading file {folder_path}/{f}")
                    
                    file_date_str = f.split("_")[1] # 2020-08-01
                    file_date = datetime.datetime.strptime(file_date_str, '%Y-%m-%d').date()
                    if file_date < start_date or file_date > end_date:
                        # File is not relevant
                        continue
                    
                    rows = []
                    r = 0
                    with open(f"{folder_path}/{f}", "r") as fr:
                        csv_reader = csv.reader(fr)
                        for row in csv_reader:
                            if r == 0:
                                # Header so skip
                                pass
                            else:
                                content = strip_row(row[:len(ALERT_COLUMNS) - 1])
                                silence_times = "\t".join(strip_row(row[len(ALERT_COLUMNS) - 1:]))
                                output_row = []
                                output_row.extend(content)
                                output_row.append(silence_times)
                                if len(output_row) == len(ALERT_COLUMNS):
                                    rows.append(output_row)
                                else:
                                    print(f"Row in file {folder_path}/{f} has len {len(output_row)} but expected {len(ALERT_COLUMNS)}")
                            r += 1

                    # print(f"Read file {folder_path}/{f} with {len(rows)} rows")
                    output_files.append(rows)
    else:
        print(f"Could not determine where study {study} is located")
    return output_files


def process_alerts_file(patient_id, study_to_study_folder, studies, start, end):
    output_rows = []
    for study in sorted(studies):
        for df in load_alerts_file(study_to_study_folder, study, start, end):
            if df is None:
                # There are no numerics for some reason
                print(
                    f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file with study {study} does not exist!")
                continue

            for row in df:
                # Data is only available spuriously, so collect all measures as we can between start/end

                date_str = row[ALERT_COL_TO_INDEX["AnnounceDate"]].strip() + " " + row[ALERT_COL_TO_INDEX["AnnounceTime"]].strip()
                row_time = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
                if row_time < start or row_time > end:
                    # Row is out of our study range
                    continue

                output_row = [patient_id]
                for i, col in enumerate(ALERT_COLUMNS):
                    if i < len(row):
                        output_row.append(row[ALERT_COL_TO_INDEX[col]])
                    else:
                        output_row.append("")
                output_rows.append(output_row)

        print(
            f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Alerts file with studies {studies} has len {len(output_rows)}")
    return output_rows, patient_id


def process_study(input_args):
    curr_patient_index, total_patients, patient_id, studies, patient_to_actual_times, patient_to_row, study_to_info, study_to_study_folder = input_args

    try:
        print(
            f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Starting patient processing {curr_patient_index}/{total_patients}...")

        if patient_id not in patient_to_actual_times:
            print(
                f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Patient {patient_id} skipped because there was no end dispo")
            return {}

        roomed_time = patient_to_actual_times[patient_id]["roomed_time"]
        dispo_time = patient_to_actual_times[patient_id]["dispo_time"]

        # Extract alerts data
        #
        alerts, pt = process_alerts_file(patient_id, study_to_study_folder, studies, roomed_time, dispo_time)
        return {
            "alerts": alerts,
            "patient_id": pt
        }
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None


def process_studies(patient_to_actual_times, patient_to_studies, patient_to_row, study_to_info, study_to_study_folder,
                    limit):
    output_rows = []

    input_args_list = []
    i = 0
    for patient_id, studies in tqdm(patient_to_studies.items(), disable=True):
        i += 1
        if limit is not None and i > limit:
            break

        input_args = [i, len(patient_to_studies), patient_id, studies, patient_to_actual_times,
                      patient_to_row, study_to_info, study_to_study_folder]
        input_args_list.append(input_args)
    
    fs = []
    with futures.ThreadPoolExecutor(16) as executor:
        results = executor.map(process_study, input_args_list)

    for result in results:
        if result is not None:
            output_rows.extend(result["alerts"])

    return output_rows


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
    study_to_study_folder = {}

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
        study_to_study_folder[study] = study_folder

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

    return patient_to_studies, study_to_info, study_to_study_folder


def write_output_file(output_rows, output_file):
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        headers = ["patient_id"]
        headers.extend(ALERT_COLUMNS)
        writer.writerow(headers)

        print(f"Received {len(output_rows)} to write")

        for row in output_rows:
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves and consolidates numeric and waveform data from raw study folder')
    parser.add_argument('-m', '--mapping-file',
                        required=True,
                        help='The path to a matched cohort file. e.g. " /deep/group/ed-monitor-self-supervised/v3/matched-cohort.csv"')
    parser.add_argument('-e', '--exports-file',
                        required=True,
                        help='Exports file e.g. "/deep/group/ed-monitor-self-supervised/v3/matched-export.csv"')
    parser.add_argument('-f', '--output-file',
                        required=True,
                        help='The path to the output consolidated summary file. e.g. "/deep/group/ed-monitor-self-supervised/v3/alerts.csv"')
    parser.add_argument('-l', '--limit',
                        required=False,
                        default=None,
                        help='Maximum number of patients to produce')

    args = parser.parse_args()

    # Mapping file contains the original cohort information. It is primarily used here to retrieve basic information on the patient.
    mapping_file = args.mapping_file

    # Exports file contains the patient and STUDY ID relationship.
    exports_file = args.exports_file

    # Where the output summary should be written to
    output_file = args.output_file

    if args.limit is not None:
        limit = int(args.limit)
    else:
        limit = None

    print("=" * 30)
    print(
        f"Starting data consolidation with mapping_file={mapping_file}, exports_file={exports_file}, limit={limit}")
    print("-" * 30)

    patient_to_actual_times, patient_to_row = load_mapping_file(mapping_file)
    patient_to_studies, study_to_info, study_to_study_folder = load_exports_file(exports_file)
    output_rows = process_studies(patient_to_actual_times, patient_to_studies, patient_to_row, study_to_info,
                                            study_to_study_folder, limit)

    print("==" * 30)
    write_output_file(output_rows, output_file)

    print("DONE")
