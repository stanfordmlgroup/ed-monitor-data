#!/usr/bin/env python

"""
Script to consolidate all the raw waveform files into a single folder. This script has the following features:
- Trims waveforms to the start/end times specified in the input file.
- Combines multiple studies if necessary.
- Concatenates multiple waveform files into one numpy object for easier access (one per each waveform type).
- Resamples all waveforms to 500Hz (to match the frequency of ECG).
- Optionally trims out empty waveforms based on a heuristic.

Each patient visit will be written out in the following format to the output folder specified:
- 1001/
    - info.pkl
    - II.dat
    - Pleth.dat
    - Resp.dat

Usage:
- python -u /deep/u/tomjin/ed-monitor-data/processing/consolidate.py /deep/group/lactate/v1/matched-cohort.csv /deep/group/lactate/v1/matched-export.csv /deep/group/lactate/v1/patient-data /deep/group/lactate/v1/consolidated.csv 3

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
import pickle
from biosppy.signals.tools import filter_signal
from pathlib import Path
from concurrent import futures

pd.set_option('display.max_columns', 500)

START = "start"
END = "end"

PATIENT = "patient"
DATE_DIR = "date_dir"
STUDIES = "studies"
START_TIME = "start_time"
END_TIME = "end_time"

# We always round to the nearest 500 when performing calculations to ensure that all ECG, Pleth, and Resp
# data are perfectly aligned (since the least common multiple of their sample rates is 500)
BASE_UNIT = 500

# Number of seconds of continuous data after which the file will be considered to have data.
# This is importance since the initial readings may not be very accurate (application/removal of probes)
# so this range is considered "blocked" from processing.
# e.g. XXXX---------------------XXXX
#      where 'X' are blocked margin points.
MARGIN_SECONDS = 5

# Empirically determined range of valid values (after gain is applied)
# Note that they are generously padded to account for outliers
VALID_RANGES = {
    "Resp": (-10, 10),
    "Pleth": (0, 5000),
    "II": (-10, 10),
}

WAVEFORM_TYPES = {"II", "Pleth", "Resp"}
WAVEFORM_SAMPLE_RATES = {
    "I": 500,
    "II": 500,
    "III": 500,
    "V": 500,
    "aVR": 500,
    "Pleth": 125,
    "Resp": 62.5
}

DEBUG = True
LIMIT = None
TRIM_WAVEFORMS = False

def get_time_from_clock_file(clock_path, return_type=START):
    """
    Note that we drop the milliseconds since we don't care about the accuracy at the millisecond level.
    This makes future calculations easier since we don't have to consider the milliseconds when calculating
    offsets with the patient admit/discharge times (which don't have millisecond granularity).
    """
    with open(clock_path, "r") as f:
        lines = f.read().splitlines()
        if return_type == START:
            row = lines[0]
            row_arr = re.split(" +", row)
            date_str = " ".join(row_arr[1:4])
            # 09/12/2020 20:51:38.088 -07:00
            date_obj = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
            date_obj = date_obj.replace(microsecond=0)
            return date_obj
        elif return_type == END:
            row = lines[-1]
            row_arr = re.split(" +", row)
            date_str = " ".join(row_arr[1:4])
            # 09/12/2020 20:51:38.088 -07:00
            date_obj = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
            date_obj = date_obj.replace(microsecond=0)
            return date_obj
        else:
            raise NotImplementedError("Unknown return_type specified")


def read_info(info_file_name):
    def get_gain(row_arr):
        for element in row_arr:
            if "gain=" in element:
                return float(element.split("=")[1])
        return 1.0

    def get_ylim(row_arr):
        for i, element in enumerate(row_arr):
            if "y-limits=" in element:
                return [float(row_arr[i + 1]), float(row_arr[i + 2])]
        return []

    info_map = {}
    with open(info_file_name, "r") as f:
        for row in f:
            if row.startswith("#"):
                continue
            row_arr = row.split(" ")
            wave_name = row_arr[0]  # e.g. STUDY-014959_2020-09-12_00-00-00.II.dat
            if wave_name != "dummy":
                # Waveform types:
                #
                # Respiration waveform (e.g. rise and fall of chest)
                #
                # CO2 waveform (e.g. mmHg of CO2)
                #
                # Plethysmograph waveform (e.g. light through finger)
                # Note: SpO2 is derived from this data, but we need information on IR and red spectrums
                #       which this dataset does not provide.
                #
                # ECG waveforms (most commonly lead II, but also III, V are seen)
                #

                wave_type = wave_name.split(".")[1]  # e.g. II, Pleth, Resp, etc.
                gain = get_gain(row_arr)
                y_lim = get_ylim(row_arr)
                info_obj = {
                    "gain": gain,
                    "y_lim": y_lim,
                    "wave_type": wave_type,
                    "sample_rate": float(row_arr[1]),
                    "units": row_arr[3],
                    "filepath": info_file_name.replace(".info", "." + wave_type + ".dat"),
                    "is_ecg": wave_type != "Resp" and wave_type != "Pleth" and wave_type != "CO2"
                }
                info_map[wave_type] = info_obj
    return info_map


def get_skip_waveform_seconds(patient_id, waveform, wave_type, sample_rate):
    """
    Trims the waveforms to the point where there is actual data.
    This is done by moving a sliding window until we get a full
    window of values within the specified range.
    """
    start = 0
    end = 0

    window = []
    for i in range(0, len(waveform), int(sample_rate)):
        if len(window) < MARGIN_SECONDS:
            window.append(waveform[i])
        else:
            window.pop(0)
            window.append(waveform[i])
            min_value = np.min(window)
            max_value = np.max(window)
            if min_value >= VALID_RANGES[wave_type][0] and max_value <= VALID_RANGES[wave_type][1]:
                # We found a valid window so we can use this as the start index
                start = i
                break
            assert len(window) == MARGIN_SECONDS

    window = []
    for i in range(len(waveform) - 1, -1, -int(sample_rate)):
        if len(window) < MARGIN_SECONDS:
            window.append(waveform[i])
        else:
            window.pop(0)
            window.append(waveform[i])
            min_value = np.min(window)
            max_value = np.max(window)
            if min_value >= VALID_RANGES[wave_type][0] and max_value <= VALID_RANGES[wave_type][1]:
                # We found a valid window so we can use this as the start index
                end = i
                break
            assert len(window) == MARGIN_SECONDS

    if start > end:
        if DEBUG:
            print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > [WARN] start {start} > end {end} so truncating the end to start")
        end = start

    return math.ceil(start / sample_rate), math.floor(end / sample_rate)


def join_waveforms(patient_id, file_metadata_list, w):
    final_waveform = np.array([])
    waveform_start_time = None
    waveform_end_time = None
    for i, metadata in enumerate(file_metadata_list):
        # Note we can iterate in order of the metadata because files were scanned in chronological
        # order based on their file prefixes.
        gain = metadata["info"][w]["gain"]
        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Reading from file {metadata['filename']} trimming {int(metadata['start_offset'])} to {int(metadata['end_offset'])}")
        waveform = np.fromfile(metadata["filename"], dtype=np.int16)
        waveform = waveform[int(metadata["start_offset"]):int(metadata["end_offset"])]
        waveform = waveform * gain
        final_waveform = np.concatenate((final_waveform, waveform))

        if i == 0:
            waveform_start_time = metadata["start_offset_time"]
        if i == len(file_metadata_list) - 1:
            waveform_end_time = metadata["end_offset_time"]

    return final_waveform, waveform_start_time, waveform_end_time


def get_waveform_offset(study_start_time, patient_time, sample_rate=500):
    return (patient_time - study_start_time).total_seconds() * sample_rate


def parse_vital(vital, index_to_return=0):
    if vital == "":
        return float('nan')
    try:
        return float(vital)
    except Exception:
        return float('nan')

def process_study(input_args):
    curr_patient_index, total_patients, patient_id, studies, patient_to_actual_times, patient_to_row, study_to_info, output_dir = input_args
    min_start = None
    max_end = None

    print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Starting patient processing {curr_patient_index}/{total_patients}...")

    if patient_id not in patient_to_actual_times:
        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Patient {patient_id} skipped because there was no end dispo")
        return {}
    roomed_time = patient_to_actual_times[patient_id]["roomed_time"]
    dispo_time = patient_to_actual_times[patient_id]["dispo_time"]

    waveform_to_metadata = {}
    for study in studies:
        info = study_to_info[study]
        study_path = info["study_folder"]

        if DEBUG:
            print(
                f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Processing patient_id {patient_id} study {study} in dir {study_path} roomed_time = {roomed_time} dispo_time = {dispo_time}")

        if not os.path.isdir(study_path):
            print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Patient {patient_id} skipped because study path not found {study_path}")
            break

        start_times = {}
        end_times = {}
        for f in os.listdir(study_path):
            prefix = f.split(".")[0]  # e.g. STUDY-028806_2020-09-19_00-00-00
            actual_type = f.split(".")[-2]  # e.g. II
            if actual_type == "clock":
                try:
                    start_times[prefix] = get_time_from_clock_file(os.path.join(study_path, f), return_type=START)
                    end_times[prefix] = get_time_from_clock_file(os.path.join(study_path, f), return_type=END)

                    if min_start is None or min_start > start_times[prefix]:
                        min_start = start_times[prefix]
                    if max_end is None or max_end < end_times[prefix]:
                        max_end = end_times[prefix]

                    if DEBUG:
                        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > clock file {prefix}: {start_times[prefix]} to {end_times[prefix]}")
                except IndexError:
                    if DEBUG:
                        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     >>> Clock file was empty")
        if len(start_times) == 0:
            if DEBUG:
                print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     >>> No clock file found for study " + study)
            continue

        for prefix in sorted(start_times.keys()):
            # It is assumed that the same info object can be reused for all files in this STUDY
            info_filename = os.path.join(study_path, prefix + ".info")
            info_obj = read_info(info_filename)

            for wave_type in WAVEFORM_TYPES:

                if wave_type not in info_obj:
                    if DEBUG:
                        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Wave type {wave_type} not found in study {study}")
                    continue

                if DEBUG:
                    print(
                        f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > For {wave_type} prefix {prefix} start={start_times[prefix]} end={end_times[prefix]}")

                # Patient start/end always start/end on the second but the waveforms start/end on
                # the millisecond so we must first trim the waveform to be within the patient time
                # frame and to ensure that the waveform will be trimmed starting from the start
                # of a second.
                waveform_start = start_times[prefix]
                waveform_end = end_times[prefix]
                if waveform_start <= dispo_time and roomed_time <= waveform_end:
                    start_offset_time = max(roomed_time, waveform_start)
                    end_offset_time = min(dispo_time, waveform_end)
                    start_offset = get_waveform_offset(waveform_start, start_offset_time,
                                                       sample_rate=info_obj[wave_type]["sample_rate"])
                    end_offset = get_waveform_offset(waveform_start, end_offset_time,
                                                     sample_rate=info_obj[wave_type]["sample_rate"])
                    filename = os.path.join(study_path, f"{prefix}.{wave_type}.dat")
                    if wave_type not in waveform_to_metadata:
                        waveform_to_metadata[wave_type] = []
                    waveform_to_metadata[wave_type].append({
                        "start_offset": start_offset,  # this is where we want to trim from
                        "start_offset_time": start_offset_time,
                        "end_offset": end_offset,  # this is where we want to trim to
                        "end_offset_time": end_offset_time,
                        "filename": filename,
                        "info": info_obj
                    })
                    if DEBUG:
                        print(
                            f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Waveform {filename} start={start_offset} end={end_offset} start_offset_time={str(start_offset_time)} end_offset_time={str(end_offset_time)}")

    if len(waveform_to_metadata) == 0:
        if DEBUG:
            print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > No waveforms found")

        return {
            "patient_id": patient_id,
            "data_length_sec": 0,
            "min_start": min_start,
            "max_end": max_end,
            "trim_start_sec": 0,
            "trim_end_sec": 0,
            "roomed_time": roomed_time,
            "dispo_time": dispo_time,
            "notes": "no waveforms found",
            "II": 1 if "II" in waveform_to_metadata else 0,
            "Pleth": 1 if "Pleth" in waveform_to_metadata else 0,
            "Resp": 1 if "Resp" in waveform_to_metadata else 0,
            "studies": ",".join(studies),
            "row": patient_to_row[patient_id]
        }

    # Our model is based on the assumption that there are lead II waveforms
    #
    if "II" not in waveform_to_metadata:
        if DEBUG:
            print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Lead II not found in waveforms")

        return {
            "patient_id": patient_id,
            "data_length_sec": 0,
            "min_start": min_start,
            "max_end": max_end,
            "trim_start_sec": 0,
            "trim_end_sec": 0,
            "roomed_time": roomed_time,
            "dispo_time": dispo_time,
            "notes": "lead II not found",
            "II": 1 if "II" in waveform_to_metadata else 0,
            "Pleth": 1 if "Pleth" in waveform_to_metadata else 0,
            "Resp": 1 if "Resp" in waveform_to_metadata else 0,
            "studies": ",".join(studies),
            "row": patient_to_row[patient_id]
        }

    waveform_type_to_waveform = {}
    waveform_type_to_times = {}
    for w, metadata_list in waveform_to_metadata.items():
        final_waveform, waveform_start_time, waveform_end_time = join_waveforms(patient_id, metadata_list, w)
        waveform_type_to_waveform[w] = final_waveform
        waveform_type_to_times[w] = {
            "start": waveform_start_time,
            "end": waveform_end_time
        }

    trim_start_sec, trim_end_sec = get_skip_waveform_seconds(patient_id, waveform_type_to_waveform["II"], "II",
                                                             WAVEFORM_SAMPLE_RATES["II"])
    if DEBUG:
        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Calculated recommended trim to be ({trim_start_sec}, {trim_end_sec})")

    if TRIM_WAVEFORMS:
        if trim_end_sec - trim_start_sec == 0:
            if DEBUG:
                print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > trim start = trim end")
            return {
                "patient_id": patient_id,
                "data_length_sec": 0,
                "min_start": min_start,
                "max_end": max_end,
                "trim_start_sec": trim_start_sec,
                "trim_end_sec": trim_end_sec,
                "roomed_time": roomed_time,
                "dispo_time": dispo_time,
                "notes": "trim start is the same as trim end",
                "II": 1 if "II" in waveform_to_metadata else 0,
                "Pleth": 1 if "Pleth" in waveform_to_metadata else 0,
                "Resp": 1 if "Resp" in waveform_to_metadata else 0,
                "studies": ",".join(studies),
                "row": patient_to_row[patient_id]
            }

        for w, waveform in waveform_type_to_waveform.items():
            sample_rate = WAVEFORM_SAMPLE_RATES[w]
            waveform_type_to_waveform[w] = waveform_type_to_waveform[w][
                                           int(trim_start_sec * sample_rate):int(trim_end_sec * sample_rate)]

    data_length_sec = round(len(waveform_type_to_waveform["II"]) / WAVEFORM_SAMPLE_RATES["II"], 1)
    for w, waveform in waveform_type_to_waveform.items():
        if DEBUG:
            print(
                f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Waveform {w} has length {len(waveform)} corresponding to {len(waveform) / WAVEFORM_SAMPLE_RATES[w]} secs")

        wave_length = len(waveform) / WAVEFORM_SAMPLE_RATES[w]
        # Rounding to 1 decimal place to account for how Resp is sampled at 62.5
        if data_length_sec != round(wave_length, 1):
            # Sometimes we have observed that the waveforms available change after rollover to the next day
            # so it is possible for inconsistent lengths to occur.
            #
            print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > [WARN] Inconsistent lengths detected")

    patient_output_path = os.path.join(output_dir, str(patient_id))
    Path(patient_output_path).mkdir(parents=True, exist_ok=True)
    for w, waveform in waveform_type_to_waveform.items():
        save_path = os.path.join(patient_output_path, f"{w}.dat")
        np.save(save_path, waveform)

    output_obj = {
        "data_length_sec": data_length_sec,
        "min_start": min_start,
        "max_end": max_end,
        "roomed_time": roomed_time,
        "dispo_time": dispo_time,
        "trim_start_sec": trim_start_sec,
        "trim_end_sec": trim_end_sec,
        "waveform_type_to_times": waveform_type_to_times,
        "supported_types": list(waveform_type_to_waveform.keys())
    }
    with open(os.path.join(patient_output_path, f"info.pkl"), 'wb') as handle:
        pickle.dump(output_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Return patient summary
    #
    return {
        "patient_id": patient_id,
        "data_length_sec": data_length_sec,
        "min_start": min_start,
        "max_end": max_end,
        "trim_start_sec": trim_start_sec,
        "trim_end_sec": trim_end_sec,
        "roomed_time": roomed_time,
        "dispo_time": dispo_time,
        "waveform_start_time": waveform_type_to_times["II"]["start"],
        "waveform_end_time": waveform_type_to_times["II"]["end"],
        "notes": "",
        "II": 1 if "II" in waveform_to_metadata else 0,
        "Pleth": 1 if "Pleth" in waveform_to_metadata else 0,
        "Resp": 1 if "Resp" in waveform_to_metadata else 0,
        "studies": ",".join(studies),
        "row": patient_to_row[patient_id]
    }


def process_studies(patient_to_actual_times, patient_to_studies, patient_to_row, study_to_info, output_dir):
    patient_id_to_results = {}

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        i = 0
        for patient_id, studies in tqdm(patient_to_studies.items(), disable=True):
            i += 1
            if LIMIT is not None and i > LIMIT:
                break

            input_args = [i, len(patient_to_studies), patient_id, studies, patient_to_actual_times,
                          patient_to_row, study_to_info, output_dir]
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

        row["SpO2"] = parse_vital(row["SpO2"])
        row["RR"] = parse_vital(row["RR"])
        row["HR"] = parse_vital(row["HR"])
        row["SBP"] = parse_vital(row["SBP"])
        row["DBP"] = parse_vital(row["DBP"])
        row["Temp"] = parse_vital(row["Temp"])

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


def write_output_file(patient_id_to_results, output_file):
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        headers = ["patient_id", "roomed_time", "dispo_time", "waveform_start_time", "waveform_end_time", "visit_length_sec", "data_length_sec",
                   "data_available_offset_sec", "data_start_offset_sec", "recommended_trim_start_sec",
                   "recommended_trim_end_sec", "min_study_start", "max_study_end", "II_available", "Pleth_available",
                   "Resp_available", "studies", "notes"]
        
        # Add additional headers
        random_pt = next(iter(patient_id_to_results))
        additional_headers = []
        for k in list(patient_id_to_results[random_pt]["row"].keys()):
            if k not in headers:
                headers.append(k)
                additional_headers.append(k)

        writer.writerow(headers)

        if DEBUG:
            print(headers)

        for k, v in patient_id_to_results.items():
            visit_length_sec = (v["dispo_time"] - v["roomed_time"]).total_seconds()

            if v["min_start"] is not None:
                data_start_offset_sec = (v["roomed_time"] - v["min_start"]).total_seconds()
                data_available_offset_sec = max(0, data_start_offset_sec)
            else:
                data_start_offset_sec = "N/A"
                data_available_offset_sec = "N/A"

            waveform_start_time = str(v["waveform_start_time"]) if "waveform_start_time" in v else ""
            waveform_end_time = str(v["waveform_end_time"]) if "waveform_end_time" in v else ""

            row = [k, str(v["roomed_time"]), str(v["dispo_time"]), waveform_start_time, waveform_end_time, visit_length_sec, v["data_length_sec"],
                   data_available_offset_sec, data_start_offset_sec, v["trim_start_sec"], v["trim_end_sec"],
                   str(v["min_start"]), str(v["max_end"]), v["II"], v["Pleth"], v["Resp"], v["studies"], v["notes"]]
            
            # Append the data from the original file
            for k in additional_headers:
                row.append(v["row"][k])

            if DEBUG:
                print(row)
            writer.writerow(row)


if __name__ == '__main__':
    # Mapping file contains the original cohort information. It is primarily used here to retrieve basic information on the patient.
    mapping_file = sys.argv[1]

    # Exports file contains the patient and STUDY ID relationship.
    exports_file = sys.argv[2]

    # Where the data files should be written to
    output_dir = sys.argv[3]

    # Where the output summary should be written to
    output_file = sys.argv[4]

    if len(sys.argv) > 5:
        LIMIT = int(sys.argv[5])

    print("=" * 30)
    print(f"Starting data consolidation with mapping_file={mapping_file}, exports_file={exports_file}, waveform_types={WAVEFORM_TYPES}, limit={LIMIT}")
    print("-" * 30)

    patient_to_actual_times, patient_to_row = load_mapping_file(mapping_file)
    patient_to_studies, study_to_info = load_exports_file(exports_file)
    patient_id_to_results = process_studies(patient_to_actual_times, patient_to_studies, patient_to_row, study_to_info, output_dir)

    print("==" * 30)
    write_output_file(patient_id_to_results, output_file)

    print("DONE")
