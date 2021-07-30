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
- python consolidate_data.py cohort_matched021021.csv export_matched021021.csv patient_data consolidated.txt
- python consolidate_data.py cohort_matched021021.csv export_matched021021.csv patient_data consolidated.txt 100
- nohup python -u consolidate_data.py ss/cohort_matched021021.csv ss/export_matched021021.csv patient_data consolidated.txt > consolidate_data.log &
- nohup python -u consolidate_data.py ss/cohort_matched021021.csv ss/export_matched021021.csv patient_data_v3 consolidated_v3.txt > consolidate_data_v3.log &

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

WAVEFORM_TYPES = {"I", "II", "III", "V", "aVR", "Pleth", "Resp"}
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

ECG_FEATURES = ["Atrial Paced Beats", "Pause Events", "Dual Paced Beats", "Total Paced Beats",
                "Maximum HR in Paced Runs", "Minimum HR in Paced Runs", "Number of Pacer Not Capture Events",
                "Number of Paced Runs", "Number of Pacer Not Pacing Events", "Supra-ventricular Beats",
                "Supra-ventricular Premature Beats", "Maximum HR in SVPB Runs", "Minimum HR in SVPB Runs",
                "Ventricular Paced Beats", "PVC Beats", "Maximum HR in PVC Runs", "Minimum HR in PVC Runs",
                "Number of Multiform PVCs", "Number of PVC Pairs", "Number of SVPB Runs", "Number of PVC Runs",
                "Number of R-on-T PVC Beats", "Number of V? Runs", "Percent Paced Beats that were Atrially Paced",
                "Percent Ventricular Bigeminy", "Percent Irregular Heart Rate",
                "Percent Paced Beats that were Dual Paced", "Percent of Beats that were Paced",
                "Percent Ventricular Trigeminy", "Percent Paced Beats that were Ventricularly Paced",
                "Normal Beats", "Square Root of NN Variance", "Percent Poor Signal", "Longest PVC Run", "pNN50",
                "Total Beats"]
VITAL_SIGN_FEATURES = ["HR", "btbHR", "btbRRInt_ms", "Pulse (SpO2)", "NBPs", "NBPd", "NBPm", "Perf", "SpO2", "RR"]


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


def plot_wave(filename, start, end, skip=100):
    a = np.fromfile(filename, dtype=np.int16)
    data_sb = a[start:end:skip]
    plt.plot(data_sb)


def apply_filter(signal, filter_bandwidth, fs=500):
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                 order=order, frequency=filter_bandwidth,
                                 sampling_rate=fs)
    return signal


def get_waveform_offset(study_start_time, patient_time, sample_rate=500):
    return (patient_time - study_start_time).total_seconds() * sample_rate


def parse_vital(vital, index_to_return=0):
    if vital == "":
        return 0
    try:
        return float(vital)
    except Exception:
        vital = vital.replace("/", " ")
        vital_arr = []
        for elem in vital.split(" "):
            try:
                a = float(elem)
                vital_arr.append(a)
            except Exception:
                continue
        if index_to_return < len(vital_arr):
            return vital_arr[index_to_return]
        else:
            return 0


def process_numerics_file(patient_id, metadata_list, patient_output_path):
    actual_start_time = None

    output_rows = []
    for metadata in metadata_list:
        # Note we multiple the offset in the metadata by the following ratio because the start/end offsets
        # are actually for the lead II ECG
        start_offset = int(int(metadata["start_offset"]) * (1000 / WAVEFORM_SAMPLE_RATES["II"]))

        filename = metadata["filename"].replace("II.dat", "numerics.csv")
        if not Path(filename).is_file():
            # There are no numerics for some reason
            print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file {filename} does not exist!")
            continue

        df = pd.read_csv(filename).rename(
            columns=lambda x: x.strip()
        ).replace(r'^\s*$', np.nan, regex=True)
        first_row = df.head(1)
        date_str = first_row["Date"].item().strip() + " " + first_row["Time"].item().strip()
        file_start_time = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
        if actual_start_time is None:
            # The ECG itself will be trimmed by the start_offset value. Therefore,
            # we should effectively begin the ECG at the start_offset value as well.
            actual_start_time = file_start_time + datetime.timedelta(milliseconds=start_offset)
            print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Processing numerics file {filename} at actual start time {actual_start_time} including offset of {start_offset}")

        # Accumulate the rows so we can later take the median of each vital sign data
        vital_sign_rows_so_far = []

        for i, row in df.iterrows():
            if "Atrial Paced Beats" not in row:
                # If there are no ECG specific features, then we input it with zeroes
                if i == 0:
                    print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > No ECG specific features found!")
                date_str = row["Date"].strip() + " " + row["Time"].strip()
                curr_time = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
                file_offset = (curr_time - actual_start_time).total_seconds()
                median_row = np.array(row.reindex(VITAL_SIGN_FEATURES).to_numpy(), dtype=np.float)
                # The following will return NA for all ECG features
                ecg_row = np.array(row.reindex(ECG_FEATURES), dtype=np.float)
                assert len(ecg_row) == len(ECG_FEATURES)
                new_row = np.concatenate([[int(file_offset * 1000)], ecg_row, median_row])
                output_rows.append(new_row)
            else:
                if math.isnan(float(row["Atrial Paced Beats"])):
                    vital_sign_rows_so_far.append(np.array(row.reindex(VITAL_SIGN_FEATURES).to_numpy(), dtype=np.float))
                else:
                    date_str = row["Date"].strip() + " " + row["Time"].strip()
                    curr_time = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
                    file_offset = (curr_time - actual_start_time).total_seconds()

                    if len(vital_sign_rows_so_far) == 0:
                        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Median row is empty so skipping!")
                        continue

                    median_row = np.nanmedian(vital_sign_rows_so_far, axis=0)

                    ecg_row = np.array(row.reindex(ECG_FEATURES), dtype=np.float)
                    assert len(ecg_row) == len(ECG_FEATURES)
                    new_row = np.concatenate([[int(file_offset * 1000)], ecg_row, median_row])
                    output_rows.append(new_row)

    # Write a consolidated numerics file containing features pulled from Philips
    #
    numerics_headers = ["offset_milliseconds"]
    numerics_headers.extend(ECG_FEATURES)
    numerics_headers.extend(VITAL_SIGN_FEATURES)
    with open(os.path.join(patient_output_path, f"numerics.csv"), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')
        writer.writerow(numerics_headers)
        for r in output_rows:
            writer.writerow(r)


def process_study(input_args):
    curr_patient_index, total_patients, patient_id, studies, patient_to_actual_times, patient_to_acs, study_to_info, output_dir = input_args
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
            "outcome": 1 if patient_id in patient_to_acs else 0
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
            "outcome": 1 if patient_id in patient_to_acs else 0
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
                "outcome": 1 if patient_id in patient_to_acs else 0
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

    # We pass in the waveform_to_metadata for Lead II beacuse this
    # object contains information on where we have cut the waveforms
    process_numerics_file(patient_id, waveform_to_metadata["II"], patient_output_path)

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
        "outcome": 1 if patient_id in patient_to_acs else 0
    }


def process_studies(patient_to_actual_times, patient_to_studies, patient_to_acs, study_to_info, output_dir):
    patient_id_to_results = {}

    fs = []
    with futures.ProcessPoolExecutor(8) as executor:
        i = 0
        for patient_id, studies in tqdm(patient_to_studies.items(), disable=True):
            i += 1
            if LIMIT is not None and i > LIMIT:
                break

            input_args = [i, len(patient_to_studies), patient_id, studies, patient_to_actual_times,
                          patient_to_acs, study_to_info, output_dir]
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

    patient_to_actual_times = {}
    acs_pos = 0
    acs_neg = 0
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

        try:
            first_trop_time = datetime.datetime.strptime(row["First_trop_result_time"], "%Y-%m-%dT%H:%M:%S%z").replace(
                tzinfo=None)
            # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
            first_trop_time = pytz.timezone('America/Vancouver').localize(first_trop_time)
        except:
            first_trop_time = ""

        try:
            max_trop_time = datetime.datetime.strptime(row["Max_trop_result_time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
            # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
            max_trop_time = pytz.timezone('America/Vancouver').localize(max_trop_time)
        except:
            max_trop_time = ""

        spo2 = parse_vital(row["SpO2"])
        rr = parse_vital(row["RR"])
        hr = parse_vital(row["HR"])
        bp_sys = parse_vital(row["SBP"])
        bp_dia = parse_vital(row["DBP"])
        temp = parse_vital(row["Temp"])

        case_id = row["CaseID"]
        case = row["Case_for_train"]
        patient_to_actual_times[case_id] = {
            "roomed_time": roomed_time,
            "dispo_time": dispo_time,
            "arrival_time": arrival_time,
            "spo2": spo2,
            "rr": rr,
            "hr": hr,
            "bp_sys": bp_sys,
            "bp_dia": bp_dia,
            "temp": temp,
            "first_trop": float(row["First_trop"]) if not math.isnan(row["First_trop"]) else 0,
            "first_trop_time": first_trop_time,
            "max_trop": float(row["Max_trop"]) if not math.isnan(row["Max_trop"]) else 0,
            "max_trop_time": max_trop_time,
            "case": int(case)
        }
        if int(case) == 1:
            acs_pos += 1
        else:
            acs_neg += 1

    print(
        f"Total patients in cohort file = {len(patient_to_actual_times)}; acs={acs_pos} ({100 * acs_pos / (acs_pos + acs_neg)})%")
    return patient_to_actual_times


def load_exports_file(exports_file, patient_to_actual_times):
    df = pd.read_csv(exports_file)

    patient_to_studies = {}
    patient_to_acs = {}
    study_to_info = {}
    study_to_patient = {}
    for index, row in df.iterrows():
        study = row["StudyId"]
        case = row["CaseID"]
        folder_path = row["path"]
        start_time = row["StartTime"]
        end_time = row["EndTime"]
        if case in patient_to_actual_times:
            if patient_to_actual_times[case]["case"] == 1:
                patient_to_acs[case] = True
        start_time = datetime.datetime.strptime(start_time, '%m/%d/%y %H:%M:%S')
        start_time = start_time.astimezone(pytz.timezone('America/Vancouver'))
        end_time = datetime.datetime.strptime(end_time, '%m/%d/%y %H:%M:%S')
        end_time = end_time.astimezone(pytz.timezone('America/Vancouver'))
        
        # 4/18: Temporary fix because the Feb/Mar files weren't moved to the expected location
        #       and I don't have permissions to move them myself.
        #
        if "2021_02_01_2021_02_28" in folder_path:
            folder_path = os.path.join(folder_path, "data/2021_02_01_2021_02_28")
            study_folder = os.path.join(folder_path, study)
        elif "2021_03_01_2021_03_31" in folder_path:
            folder_path = os.path.join(folder_path, "data/2021_03_01_2021_03_31")
            study_folder = os.path.join(folder_path, study)
        else:
            study_folder = os.path.join(os.path.join(folder_path, "data"), study)
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
        f"Total patients in mapped file = {len(patient_to_studies)}; acs={len(patient_to_acs)} ({100 * len(patient_to_acs) / (len(patient_to_studies))})%")

    return patient_to_studies, patient_to_acs, study_to_info


def write_output_file(patient_id_to_results, output_file, patient_to_actual_times):
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter='\t', quotechar='"')
        headers = ["patient_id", "arrival_time", "roomed_time", "dispo_time", "waveform_start_time", "waveform_end_time", "visit_length_sec", "data_length_sec",
                   "data_available_offset_sec", "data_start_offset_sec", "recommended_trim_start_sec",
                   "recommended_trim_end_sec", "min_study_start", "max_study_end", "II_available", "Pleth_available",
                   "Resp_available", "studies", "trimmed", "first_trop_time", "max_trop_time", "spo2", "rr", "hr", "bp_sys", "bp_dia",
                   "temp", "first_trop", "max_trop", "outcome", "notes"]
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

            first_trop_time = str(patient_to_actual_times[k]["first_trop_time"])
            max_trop_time = str(patient_to_actual_times[k]["max_trop_time"])
            arrival_time = str(patient_to_actual_times[k]["arrival_time"])
            spo2 = patient_to_actual_times[k]["spo2"]
            rr = patient_to_actual_times[k]["rr"]
            hr = patient_to_actual_times[k]["hr"]
            bp_sys = patient_to_actual_times[k]["bp_sys"]
            bp_dia = patient_to_actual_times[k]["bp_dia"]
            temp = patient_to_actual_times[k]["temp"]
            first_trop = patient_to_actual_times[k]["first_trop"]
            max_trop = patient_to_actual_times[k]["max_trop"]

            row = [k, arrival_time, str(v["roomed_time"]), str(v["dispo_time"]), waveform_start_time, waveform_end_time, visit_length_sec, v["data_length_sec"],
                   data_available_offset_sec, data_start_offset_sec, v["trim_start_sec"], v["trim_end_sec"],
                   str(v["min_start"]), str(v["max_end"]), v["II"], v["Pleth"], v["Resp"], v["studies"], TRIM_WAVEFORMS,
                   first_trop_time, max_trop_time, spo2, rr, hr, bp_sys, bp_dia, temp, first_trop, max_trop, v["outcome"],
                   v["notes"]]

            if DEBUG:
                print(row)
            writer.writerow(row)


if __name__ == '__main__':
    # Mapping file contains the original cohort information. It is primarily used here to retrieve basic information
    # on the patient such as whether they are ACS positive or not.
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

    patient_to_actual_times = load_mapping_file(mapping_file)
    patient_to_studies, patient_to_acs, study_to_info = load_exports_file(exports_file, patient_to_actual_times)
    patient_id_to_results = process_studies(patient_to_actual_times, patient_to_studies, patient_to_acs, study_to_info, output_dir)

    print("==" * 30)
    write_output_file(patient_id_to_results, output_file, patient_to_actual_times)

    print("DONE")
