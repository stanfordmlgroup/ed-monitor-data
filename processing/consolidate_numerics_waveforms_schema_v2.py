#!/usr/bin/env python

"""
Script to consolidate all the numerics and raw waveform files into a single folder. This script has the following features:
- Trims waveforms to the start/end times specified in the input file.
- Combines multiple studies if necessary.
- Concatenates multiple waveform files into one numpy object for easier access (one per each waveform type).
- Resamples all waveforms to 500Hz (to match the frequency of ECG).
- Optionally trims out empty waveforms based on a heuristic.

Each patient visit will be written out following the version 2 of the schema

Usage:
- python -u /deep/u/tomjin/ed-monitor-data/processing/consolidate_numerics_waveforms_schema_v2.py -m /deep/group/ed-monitor-self-supervised/v3/matched-cohort.csv -e /deep/group/ed-monitor-self-supervised/v3/matched-export.csv -o /deep/group/ed-monitor-self-supervised/v3/patient-data -f /deep/group/ed-monitor-self-supervised/v3/consolidated.csv -l 3

"""

import argparse
import csv
import datetime
import json
import math
import os
import re
import traceback
from concurrent import futures
from pathlib import Path
from decimal import *

import boto3
import h5py
import numpy as np
import pandas as pd
import pytz
from scipy.signal import resample
from tqdm import tqdm

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
WAVEFORM_PRIORITIES = ["II", "Pleth", "Resp"]
TARGET_WAVEFORM_SAMPLE_RATES = {
    "I": 500,
    "II": 500,
    "III": 500,
    "V": 500,
    "aVR": 500,
    "Pleth": 125,
    "ABP": 125,
    "Resp": 62.5
}

# Placeholder if the waveform is null at a particular segment
NULL_WAVEFORM_VALUE = -100

DEBUG = True


COLUMNS = [
    "HR",
    "SpO2",
    "RR",
    "NBPs",
    "NBPd",
    "NBPm",
    "btbRRInt_ms",
    "Perf"
]


def load_numerics_file(study_to_study_folder, study):
    output_files = []
    
    # print(f"[{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > starting to load numerics")
    if study in study_to_study_folder:
        folder_path = study_to_study_folder[study]

        if os.path.isdir(folder_path):
            for f in sorted(os.listdir(folder_path)):
                if f.endswith("numerics.csv"):
                    # TODO: Code bottleneck
                    # print(f"[{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > starting load individual file")
                    output_files.append(pd.read_csv(f"{folder_path}/{f}").rename(columns=lambda x: x.strip()).replace(r'^\s*$', np.nan, regex=True))
                    # print(f"[{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > done load individual file")
    else:
        print(f"Could not determine where study {study} is located")
    # print(f"[{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > finished loading numerics")
    return output_files


def process_numerics_file(patient_id, study_to_study_folder, studies, start, end):
    output_vals = {}

    for col in COLUMNS:
        output_vals[col] = {
            "vals": [],
            "times": []
        }
    
    # print(f"[{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > starting process_numerics_file")
    prev_time = None
    for study in studies:
        for df in load_numerics_file(study_to_study_folder, study):
            if df is None:
                # There are no numerics for some reason
                print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file with study {study} does not exist!")
                continue

            # TODO: Code bottleneck
            # print(f"[{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > parsing individual file")
            for i, row in df.iterrows():
                # Data is only available spuriously, so collect all measures as we can between start/end

                date_str = row["Date"].strip() + " " + row["Time"].strip()
                row_time = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
                if row_time < start or row_time > end:
                    # Row is out of our study range
                    continue
                if prev_time is not None and row_time < prev_time:
                    # Row is before the previously processed numeric value
                    print(f"[{os.getpid()}] [{datetime.datetime.now().isoformat()}] [WARN]     > overlapping numerics detected")
                    continue

                for col in COLUMNS:
                    if col in row:
                        if isinstance(row[col], str):
                            output_vals[col]["vals"].append(float(row[col].strip()))
                            output_vals[col]["times"].append(row_time.timestamp())
                        elif isinstance(row[col], float) and not math.isnan(row[col]):
                            output_vals[col]["vals"].append(row[col])
                            output_vals[col]["times"].append(row_time.timestamp())
                        prev_time = row_time
            # print(f"[{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > done parsing individual file")
    
    is_non_empty = False
    non_empty_len = 0
    for col in COLUMNS:
        if len(output_vals[col]["vals"]) > 0:
            is_non_empty = True
            non_empty_len = len(output_vals[col]["vals"])
    if is_non_empty:
        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file with studies {studies} has len {non_empty_len}")
        return output_vals, patient_id
    else:
        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Numerics file with studies {studies} is empty!")
        return None, None


def get_time_from_clock_file(clock_path, return_type=START):
    """
    Note that we drop the milliseconds since we don't care about the accuracy at the millisecond level.
    This makes future calculations easier since we don't have to consider the milliseconds when calculating
    offsets with the patient admit/discharge times (which don't have millisecond granularity).
    """
    with open(clock_path, "r") as f:
        lines = f.read().splitlines()
        if len(lines) < 1:
            return None
        if return_type == START:
            row = lines[0]
            row_arr = re.split(" +", row)
            date_str = " ".join(row_arr[1:4])
            # 09/12/2020 20:51:38.088 -07:00
            date_obj = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
            # date_obj = date_obj.replace(microsecond=0)
            return date_obj
        elif return_type == END:
            row = lines[-1]
            row_arr = re.split(" +", row)
            date_str = " ".join(row_arr[1:4])
            # 09/12/2020 20:51:38.088 -07:00
            date_obj = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f %z')
            # date_obj = date_obj.replace(microsecond=0)
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


def get_waveform_offset(study_start_time, patient_time, sample_rate=500):
    # Note that we can round the offset because the input study_start_time, patient_time were picked
    # to guarantee that we end up with a whole number
    return round((patient_time - study_start_time).total_seconds() * sample_rate)


def get_overlap_interval(available_waveforms):
    """
    The overlap interval is dominated by the waveform which has the longest interval:
    If RESP is available, the overlap interval is (1/62.5) = 16 ms
    If PPG is available, the overlap interval is (1/125) = 8 ms
    If ECG is available, the overlap interval is (1/500) = 2 ms
    """
    if "Resp" in available_waveforms:
        return Decimal('0.016')
    elif "Pleth" in available_waveforms:
        return Decimal('0.008')
    elif "II" in available_waveforms:
        return Decimal('0.002')
    else:
        raise Exception("No expected waveforms were found")


def handle_waveform_overlap_and_gap(patient_id, prev_time, metadata, final_waveform, sample_rate, available_waveforms):
    """
    There is an edge case where consecutive Philips monitoring files do not completely align.
    This means that the second file could start before or after the first file which could
    lead to alignment issues. Here, we ensure that the files are always joined at the alignment
    boundaries, meaning, the ECG, PPG, and RESP waveforms will never go out of sync.

    FILE #1

    ECG (500 Hz): X X X X X X X X X X X X X X X X
    PPG (125 Hz): Y       Y       Y       Y
    RESP (62.5 ): Z               Z
                                                ^
                                                End Time = 2021-09-02 00:00:00.003000-07:00

    -----

    FILE #2

    ECG (500 Hz):                              A A A A A A A A
    PPG (125 Hz):                              B       B
    RESP (62.5 ):                              C
                                               ^
                                               Start Time = 2021-09-02 00:00:00.002000-07:00

    -----

    CONSOLIDATED WAVEFORM:


    ECG (500 Hz): X X X X X X X X A A A A A A A A
    PPG (125 Hz): Y       Y       B       B
    RESP (62.5 ): Z               C
                                ^ ^
                Position = 19932   Position = 19933

    time_points (where position is 0-indexed):
    - Position 19932 = 2021-09-01 23:59:59.987000-07:00
    - Position 19933 = 2021-09-02 00:00:00.002000-07:00
    """
    if prev_time is not None:
        sample_rate = Decimal(str(round(sample_rate, 1)))
        overlap_interval = get_overlap_interval(available_waveforms)
        diff = Decimal(str(round((metadata["start_offset_time"] - prev_time).total_seconds(), 3)))
        if diff > 0:
            # The next waveform is after the previous waveform
            #
            print(
                f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] start offset was after previous time!")

            # e.g. diff = 1.314, overlap_interval = 0.016
            # number_of_cycles_of_null_to_add = 82
            number_of_cycles_of_null_to_add = math.floor(diff / overlap_interval)
            # number_of_sec_of_null_to_add = 1.312
            number_of_sec_of_null_to_add = number_of_cycles_of_null_to_add * overlap_interval
            # remaining_gap = 0.002
            remaining_gap = diff % overlap_interval

            final_waveform = np.concatenate((final_waveform, np.full(int(number_of_sec_of_null_to_add * sample_rate), NULL_WAVEFORM_VALUE)))

            if remaining_gap > 0:
                # This is "Position 19932" in the example above
                timepoint_1 = (len(final_waveform) - 1, metadata["start_offset_time"] - datetime.timedelta(seconds=float(remaining_gap)))

                # This is "Position 19933" in the example above
                timepoint_2 = (len(final_waveform), metadata["start_offset_time"])
                metadata["time_jumps"].append((timepoint_1, timepoint_2))

            return final_waveform
        elif diff < 0:
            # Rarely the waveforms will overlap so we can trim the previous waveform
            print(
                f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] start offset was before previous time!")
            if abs(diff * sample_rate) > 0:

                # e.g. diff = -1.314, overlap_interval = 0.016
                # number_of_cycles_of_null_to_add = -83
                number_of_cycles_to_remove = math.floor(diff / overlap_interval)
                # number_of_sec_to_remove = -1.328
                number_of_sec_to_remove = number_of_cycles_to_remove * overlap_interval
                # remaining_gap = 0.014
                remaining_gap = abs(number_of_sec_to_remove - diff)

                final_waveform = final_waveform[:int(number_of_sec_to_remove * sample_rate)]

                if remaining_gap > 0:
                    # This is "Position 19932" in the example above
                    timepoint_1 = (len(final_waveform) - 1, metadata["start_offset_time"] - datetime.timedelta(seconds=float(remaining_gap)))

                    # This is "Position 19933" in the example above
                    timepoint_2 = (len(final_waveform), metadata["start_offset_time"])
                    metadata["time_jumps"].append((timepoint_1, timepoint_2))

            return final_waveform
        else:
            # The alignment is perfect, no special handling required
            return final_waveform
    else:
        # This is the first file, so no need to worry about this edge case
        return final_waveform


def join_waveforms(patient_id, file_metadata_list, w, available_waveforms, waveform_start_end_times):
    final_waveform = np.array([])
    prev_time = None
    time_jumps = []

    # These are all the known waveform segments. Not all segments will have all waveform types
    # so we want to NULL out the waveforms for those segments which do not have data.
    available_start_end_times = []
    for waveform_start in sorted(waveform_start_end_times, key=lambda key: key):
        available_start_end_times.append((waveform_start, waveform_start_end_times[waveform_start]))

    start_time_to_metadata = {}
    for metadata in file_metadata_list:
        start_time_to_metadata[metadata["start_offset_time"]] = metadata

    for start_end_times in available_start_end_times:
        start_time = start_end_times[0]
        if start_time in start_time_to_metadata:
            metadata = start_time_to_metadata[start_time]
            gain = metadata["info"][w]["gain"]
            sample_rate = metadata["info"][w]["sample_rate"]
            waveform = np.fromfile(metadata["filename"], dtype=np.int16)
            waveform = waveform[int(metadata["start_offset"]):int(metadata["end_offset"])]
            waveform = waveform * gain
            assert len(waveform) == int(metadata["end_offset"]) - int(metadata["start_offset"]), "Trimmed waveform was not of expected length!"
        else:
            # This waveform type did not have any files for this time segment at all.
            # Pretend we read a waveform that was basically all null
            end_time = start_end_times[1]
            sample_rate = TARGET_WAVEFORM_SAMPLE_RATES[w]
            seg_len = get_waveform_offset(start_time, end_time, sample_rate=TARGET_WAVEFORM_SAMPLE_RATES[w])
            waveform = np.full(seg_len, NULL_WAVEFORM_VALUE)
            metadata = {
                "start_offset_time": start_time,
                "end_offset_time": end_time,
                "time_jumps": []
            }

        # Resample to the target waveform frequency
        if sample_rate != TARGET_WAVEFORM_SAMPLE_RATES[w]:
            print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] sample rate is not expected, resampling...")
            waveform = resample(waveform, int(len(waveform) * (TARGET_WAVEFORM_SAMPLE_RATES[w] / sample_rate)))

        final_waveform = handle_waveform_overlap_and_gap(patient_id, prev_time, metadata, final_waveform, TARGET_WAVEFORM_SAMPLE_RATES[w], available_waveforms)
        prev_time = metadata["end_offset_time"]
        final_waveform = np.concatenate((final_waveform, waveform))
        time_jumps.extend(metadata["time_jumps"])

    waveform_start_time = available_start_end_times[0][0] # Start time of first segment
    waveform_end_time = available_start_end_times[-1][1] # End time of last segment

    target_len_sec = (waveform_end_time - waveform_start_time).total_seconds()
    total_time_jump_sec = 0
    for tj in time_jumps:
        # tj = ((pos_1, time_1), (pos_2, time_2))
        total_time_jump_sec += (tj[1][1] - tj[0][1]).total_seconds()
    actual_time = len(final_waveform) / TARGET_WAVEFORM_SAMPLE_RATES[w] + total_time_jump_sec

    assert round(target_len_sec, 3) == round(actual_time, 3), "Actual waveform length does not match the expected time"

    return final_waveform, waveform_start_time, waveform_end_time, time_jumps


def parse_vital(vital, index_to_return=0):
    if vital == "":
        return float('nan')
    try:
        return float(vital)
    except Exception:
        return float('nan')


def get_start_offset_time(roomed_time, waveform_start):
    if roomed_time < waveform_start:
        return waveform_start
    elif roomed_time > waveform_start:
        # Ensure we return the roomed time that is a multiple of the cycle count of 16 ms
        # 16 ms because this would be the lowest common denominator of
        # ECG 500 hz (1/500 = 2 ms), PPG 125 Hz (1/125 = 8 ms), and Resp 62.5 Hz (1/62.5 = 16 ms)
        # e.g. roomed_time = 2023-01-01T00:00:00.861Z
        # e.g. waveform_start = 2023-01-01T00:00:00.160Z
        # diff = 0.861 - 0.160 = 0.701
        diff = Decimal(str(round((roomed_time - waveform_start).total_seconds(), 3)))
        # diff = 0.701 % 0.016 = 0.013
        diff = diff % Decimal('0.016')
        # Plus an additional cycle so that the time becomes after the roomed time
        return roomed_time - datetime.timedelta(seconds=float(diff)) + datetime.timedelta(seconds=0.016)
    else:
        return roomed_time


def get_end_offset_time(departure_time, waveform_end):
    if departure_time > waveform_end:
        return waveform_end
    elif departure_time < waveform_end:
        # Ensure we return the end time is a multiple of the cycle count of 16 ms
        # 16 ms because this would be the lowest common denominator of
        # ECG 500 hz (1/500 = 2 ms), PPG 125 Hz (1/125 = 8 ms), and Resp 62.5 Hz (1/62.5 = 16 ms)
        # e.g. waveform_end = 2023-05-31T00:00:00.002Z
        # e.g. departure_time = 2023-05-30T23:52:00.000Z
        # diff = 480.002 sec
        diff = Decimal(str(round((waveform_end - departure_time).total_seconds(), 3)))
        # diff = 0.701 % 0.016 = 0.013
        diff = diff % Decimal('0.016')
        # Subtract an additional cycle so that the time becomes before the dispo time
        return departure_time + datetime.timedelta(seconds=float(diff)) - datetime.timedelta(seconds=0.016)
    else:
        return departure_time


def format_time_jumps(time_jumps):
    output = []
    for tj in time_jumps:
        # tj = ((pos_1, time_1), (pos_2, time_2))
        output.append([
            [int(tj[0][0]), tj[0][1].timestamp()],
            [int(tj[1][0]), tj[1][1].timestamp()],
        ])
    return output


def process_study(input_args):
    curr_patient_index, total_patients, patient_id, unsorted_studies, patient_to_actual_times, patient_to_row, study_to_info, study_to_study_folder, output_dir, s3_bucket, s3_upload_path = input_args
    min_start = None
    max_end = None

    try:
        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Starting patient processing {curr_patient_index}/{total_patients}...")

        roomed_time = patient_to_actual_times[patient_id]["roomed_time"]
        dispo_time = patient_to_actual_times[patient_id]["dispo_time"]
        departure_time = patient_to_actual_times[patient_id]["departure_time"]

        # Remove any duplicate studies
        unsorted_studies = list(set(unsorted_studies))

        # Determine the order to parse the studies in (it should be ordered by the clock time)
        #
        study_to_start = []
        study_path_to_times = {}
        for study in unsorted_studies:
            info = study_to_info[study]
            study_path = info["study_folder"]
            if not os.path.isdir(study_path):
                print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Patient {patient_id} skipped because study path not found {study_path}")
                break

            skip_study = False
            local_starts = []
            for f in os.listdir(study_path):
                actual_type = f.split(".")[-2]  # e.g. II
                if actual_type == "clock":
                    start = get_time_from_clock_file(os.path.join(study_path, f), return_type=START)
                    end = get_time_from_clock_file(os.path.join(study_path, f), return_type=END)
                    if start is None or end is None:
                        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Study {study} skipped because clock was missing")
                        skip_study = True
                        break
                    study_path_to_times[study_path] = {
                        "start": start,
                        "end": end
                    }
                    local_starts.append(start)
            if skip_study:
                continue
            if len(local_starts) > 0:
                study_to_start.append((study, min(local_starts)))

        study_to_start.sort(key=lambda x: x[1])
        studies = [x[0] for x in study_to_start]

        # Extract waveforms
        #
        waveform_to_metadata = {}
        for study in studies:
            info = study_to_info[study]
            study_path = info["study_folder"]

            if DEBUG:
                print(
                    f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] Processing patient_id {patient_id} study {study} in dir {study_path} roomed_time = {roomed_time} departure_time = {departure_time}")

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
                    if waveform_start <= departure_time and roomed_time <= waveform_end:
                        start_offset_time = get_start_offset_time(roomed_time, waveform_start)
                        end_offset_time = get_end_offset_time(departure_time, waveform_end)

                        start_offset = get_waveform_offset(waveform_start, start_offset_time,
                                                           sample_rate=info_obj[wave_type]["sample_rate"])
                        end_offset = get_waveform_offset(waveform_start, end_offset_time,
                                                         sample_rate=info_obj[wave_type]["sample_rate"])

                        if start_offset < end_offset:
                            filename = os.path.join(study_path, f"{prefix}.{wave_type}.dat")
                            if wave_type not in waveform_to_metadata:
                                waveform_to_metadata[wave_type] = []
                            waveform_to_metadata[wave_type].append({
                                "start_offset": start_offset,  # this is where we want to trim from
                                "start_offset_time": start_offset_time,
                                "end_offset": end_offset,  # this is where we want to trim to
                                "end_offset_time": end_offset_time,
                                "filename": filename,
                                "info": info_obj,
                                "time_jumps": [] # these are any points in the waveform where a time jump occurred
                            })
                            if DEBUG:
                                print(
                                    f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Waveform {filename} start={start_offset} end={end_offset} start_offset_time={str(start_offset_time)} end_offset_time={str(end_offset_time)}")
                        else:
                            if DEBUG:
                                print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Waveform start={start_offset} is AFTER end={end_offset} ")

        notes = ""
        if len(waveform_to_metadata) == 0:
            if DEBUG:
                print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > No waveforms found")
            notes = "no waveforms found"

        # Our model is based on the assumption that there are lead II waveforms
        #
        elif "II" not in waveform_to_metadata:
            if DEBUG:
                print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Lead II not found in waveforms")
            notes = "lead II not found"

        waveform_type_to_waveform = {}
        waveform_type_to_times = {}

        waveform_start_end_times = {}
        for w, metadata_list in waveform_to_metadata.items():
            for metadata in metadata_list:
                waveform_start_end_times[metadata["start_offset_time"]] = metadata["end_offset_time"]

        for w, metadata_list in waveform_to_metadata.items():
            final_waveform, waveform_start_time, waveform_end_time, time_jumps = join_waveforms(patient_id, metadata_list, w, set(waveform_to_metadata.keys()), waveform_start_end_times)
            waveform_type_to_waveform[w] = final_waveform
            waveform_type_to_times[w] = {
                "start": waveform_start_time,
                "end": waveform_end_time,
                "time_jumps": time_jumps
            }

        data_length_sec = 0
        trim_start_sec = 0
        trim_end_sec = 0
        waveform_start_time = ""
        waveform_end_time = ""
        for j, waveform_anchor in enumerate(WAVEFORM_PRIORITIES):
            if waveform_anchor in waveform_to_metadata:
                waveform_start_time = waveform_type_to_times[waveform_anchor]["start"]
                waveform_end_time = waveform_type_to_times[waveform_anchor]["end"]

                trim_start_sec, trim_end_sec = get_skip_waveform_seconds(patient_id, waveform_type_to_waveform[waveform_anchor], waveform_anchor,
                                                                         TARGET_WAVEFORM_SAMPLE_RATES[waveform_anchor])
                if DEBUG:
                    print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Calculated recommended trim to be ({trim_start_sec}, {trim_end_sec})")

                data_length_sec = round(len(waveform_type_to_waveform[waveform_anchor]) / TARGET_WAVEFORM_SAMPLE_RATES[waveform_anchor], 1)
                for w, waveform in waveform_type_to_waveform.items():
                    if DEBUG:
                        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}]     > Waveform {w} has length {len(waveform)} corresponding to {len(waveform) / TARGET_WAVEFORM_SAMPLE_RATES[w]} secs")
                break

        # Extract numerics data
        #
        numerics, pt = process_numerics_file(patient_id, study_to_study_folder, studies, roomed_time, departure_time)

        # Save output files
        #
        if numerics is not None or len(waveform_type_to_waveform) > 0:
            patient_output_path = os.path.join(output_dir, str(patient_id)[-2:])
            Path(patient_output_path).mkdir(parents=True, exist_ok=True)
            output_save_path = os.path.join(patient_output_path, f"{patient_id}.h5")
            with h5py.File(output_save_path, "w") as f:
                f.create_dataset("schema_version", data="2")

                numeric_group = f.create_group("numerics")
                if numerics is not None:
                    for k, v in numerics.items():
                        k_group = numeric_group.create_group(k)
                        k_group.create_dataset("vals", data=np.array(v["vals"]))
                        k_group.create_dataset("times", data=np.array(v["times"]))

                waveform_group = f.create_group("waveforms")
                for waveform_type, waveform in waveform_type_to_waveform.items():
                    k_group = waveform_group.create_group(waveform_type)
                    time_jump_list = format_time_jumps(waveform_type_to_times[waveform_type]["time_jumps"])
                    curr_idx = 0
                    curr_time = waveform_start_time.timestamp()
                    for time_jump in time_jump_list:
                        time_jump_before = time_jump[0]
                        time_jump_after = time_jump[1]

                        v_group = k_group.create_group(datetime.datetime.fromtimestamp(curr_time, pytz.timezone("America/Vancouver")).strftime('%Y-%m-%dT%H:%M:%S%z'))
                        v_group.create_dataset("waveform", data=waveform[curr_idx:time_jump_before[0] + 1]) # adds 1 due to the exclusive range query
                        v_group.create_dataset("start", data=curr_time)
                        v_group.create_dataset("end", data=time_jump_before[1])
                        curr_idx = time_jump_after[0]
                        curr_time = time_jump_after[1]

                    v_group = k_group.create_group(datetime.datetime.fromtimestamp(curr_time, pytz.timezone("America/Vancouver")).strftime('%Y-%m-%dT%H:%M:%S%z'))
                    v_group.create_dataset("waveform", data=waveform[curr_idx:])
                    v_group.create_dataset("start", data=curr_time)
                    v_group.create_dataset("end", data=waveform_end_time.timestamp())

            if s3_bucket is not None and s3_upload_path is not None:
                s3 = boto3.client('s3')
                s3_patient_output_path = os.path.join(s3_upload_path, str(patient_id)[-2:])
                s3_save_path = os.path.join(s3_patient_output_path, f"{patient_id}.h5")
                with open(output_save_path, "rb") as f:
                    s3.upload_fileobj(f, s3_bucket, s3_save_path)

        # Return patient summary
        #
        obj = {
            "patient_id": patient_id,
            "data_length_sec": data_length_sec,
            "min_start": min_start,
            "max_end": max_end,
            "trim_start_sec": trim_start_sec,
            "trim_end_sec": trim_end_sec,
            "roomed_time": roomed_time,
            "dispo_time": dispo_time,
            "waveform_start_time": waveform_start_time,
            "waveform_end_time": waveform_end_time,
            "notes": notes,
            "studies": ",".join(studies),
        }
        for wt in WAVEFORM_TYPES:
            obj[wt] = 1 if wt in waveform_to_metadata else 0
        
        # Print object to allow service to continue if failures occur
        print(json.dumps(obj, default=str))

        obj["row"] = patient_to_row[patient_id]
        return obj
    except Exception as e:
        print(f"[{patient_id}] [{os.getpid()}] [{datetime.datetime.now().isoformat()}] [ERROR] Could not process patient due to error: {e}")
        print(traceback.format_exc())
        return None


def process_studies(patient_to_actual_times, patient_to_studies, patient_to_row, study_to_info, study_to_study_folder, output_dir, s3_bucket, s3_upload_path, limit):
    patient_id_to_results = {}
    
    print(f"Using WAVEFORM_TYPES = {WAVEFORM_TYPES}")

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        i = 0
        for patient_id, studies in tqdm(patient_to_studies.items(), disable=True):
            i += 1
            if limit is not None and i > limit:
                break
            
            if patient_id not in patient_to_actual_times:
                print(f"[{patient_id}] skipped because it was not in the mapping file or is not valid")
                continue

            input_args = [i, len(patient_to_studies), patient_id, studies, patient_to_actual_times,
                          patient_to_row, study_to_info, study_to_study_folder, output_dir, s3_bucket, s3_upload_path]
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
        if pd.isna(row["Roomed_time"]) or pd.isna(row["Arrival_time"]) or pd.isna(row["Departure_time"]):
            continue

        roomed_time = datetime.datetime.strptime(row["Roomed_time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)

        arrival_time = datetime.datetime.strptime(row["Arrival_time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
        arrival_time = pytz.timezone('America/Vancouver').localize(arrival_time)

        departure_time = datetime.datetime.strptime(row["Departure_time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
        departure_time = pytz.timezone('America/Vancouver').localize(departure_time)

        if pd.isna(row["Dispo_time"]):
            dispo_time = ""
        else:
            dispo_time = datetime.datetime.strptime(row["Dispo_time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
            # While the file lists this as UTC time, david_kim@ confirmed that this is actually Pacific time
            dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)

        case_id = row["CSN"]
        patient_to_actual_times[case_id] = {
            "roomed_time": roomed_time,
            "dispo_time": dispo_time,
            "arrival_time": arrival_time,
            "departure_time": departure_time
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
            # print(f"Found same study {study} used for multiple patients: {case}, {study_to_patient[study]}")
            pass

    print("---")
    print(
        f"Total patients in mapped file = {len(patient_to_studies)}")

    return patient_to_studies, study_to_info, study_to_study_folder


def write_output_file(patient_id_to_results, output_file):
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        headers = ["patient_id", "roomed_time", "dispo_time", "waveform_start_time", "waveform_end_time", "visit_length_sec", "data_length_sec",
                   "data_available_offset_sec", "data_start_offset_sec", "recommended_trim_start_sec",
                   "recommended_trim_end_sec", "min_study_start", "max_study_end"]
        for wt in sorted(WAVEFORM_TYPES):
            headers.append(f"{wt}_available")
        headers.extend(["studies", "notes"])
        
        # Add additional headers
        additional_headers = []
        try:
            random_pt = next(iter(patient_id_to_results))
            for k in list(patient_id_to_results[random_pt]["row"].keys()):
                if k not in headers:
                    headers.append(k)
                    additional_headers.append(k)
        except:
            print(f"Could not find headers to write")

        writer.writerow(headers)

        if DEBUG:
            print(headers)

        for k, v in patient_id_to_results.items():
            visit_length_sec = (v["departure_time"] - v["roomed_time"]).total_seconds()

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
                   str(v["min_start"]), str(v["max_end"])]
            
            for wt in sorted(WAVEFORM_TYPES):
                row.append(v[wt])
            row.extend([v["studies"], v["notes"]])
            
            # Append the data from the original file
            for k in additional_headers:
                row.append(v["row"][k])

            if DEBUG:
                print(row)
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrieves and consolidates numeric and waveform data from raw study folder')
    parser.add_argument('-m', '--mapping-file',
                        required=True,
                        help='The path to a matched cohort file. e.g. " /deep/group/ed-monitor-self-supervised/v3/matched-cohort.csv"')
    parser.add_argument('-e', '--exports-file',
                        required=True,
                        help='Exports file e.g. "/deep/group/ed-monitor-self-supervised/v3/matched-export.csv"')
    parser.add_argument('-o', '--output-dir',
                        required=True,
                        help='Folder where you should output the H5 files e.g. /deep/group/ed-monitor-self-supervised/v3/patient-data')
    parser.add_argument('-f', '--output-file',
                        required=True,
                        help='The path to the output consolidated summary file. e.g. "/deep/group/ed-monitor-self-supervised/v3/consolidated.csv". Will skip anything in this file.')
    parser.add_argument('-c', '--continue-from',
                        required=False,
                        default=None,
                        help='The path to the consolidated summary file from a previous run (saves time if running again)')
    parser.add_argument('-s3b', '--s3-bucket',
                        required=False,
                        default=None,
                        help='The S3 bucket where we want to upload the files to')
    parser.add_argument('-s3p', '--s3-upload-path',
                        required=False,
                        default=None,
                        help='Path to the S3 bucket where we want to upload the files to')
    parser.add_argument('-t', '--waveform-types',
                        required=False,
                        default=None,
                        help='Comma separated list of waveform types')
    parser.add_argument('-l', '--limit',
                        required=False,
                        default=None,
                        help='Maximum number of patients to produce')

    args = parser.parse_args()
    
    # Mapping file contains the original cohort information. It is primarily used here to retrieve basic information on the patient.
    mapping_file = args.mapping_file

    # Exports file contains the patient and STUDY ID relationship.
    exports_file = args.exports_file

    # Where the data files should be written to
    output_dir = args.output_dir

    # Where the output summary should be written to
    output_file = args.output_file

    # S3 bucket
    s3_bucket = args.s3_bucket

    # S3 upload path
    s3_upload_path = args.s3_upload_path

    if args.waveform_types is not None:
        WAVEFORM_TYPES = set(args.waveform_types.split(","))
    
    if args.limit is not None:
        limit = int(args.limit)
    else:
        limit = None

    print("=" * 30)
    print(f"Starting data consolidation with mapping_file={mapping_file}, exports_file={exports_file}, waveform_types={WAVEFORM_TYPES}, s3_bucket={s3_bucket}, s3_upload_path={s3_upload_path}, limit={limit}")
    print("-" * 30)

    patient_to_actual_times, patient_to_row = load_mapping_file(mapping_file)
    patient_to_studies, study_to_info, study_to_study_folder = load_exports_file(exports_file)
    
    # Remove any patients in patient_to_actual_times but not in the patient_to_studies map
    keys = list(patient_to_studies.keys())
    print(f"patient_to_studies had {len(keys)} length originally")
    for k in keys:
        if k not in patient_to_actual_times:
            del patient_to_studies[k]
    print(f"patient_to_studies now has {len(patient_to_actual_times)} length after removing missing or invalid keys")

    patient_id_to_results = process_studies(patient_to_actual_times, patient_to_studies, patient_to_row, study_to_info, study_to_study_folder, output_dir, s3_bucket, s3_upload_path, limit)

    print("==" * 30)
    write_output_file(patient_id_to_results, output_file)

    print("DONE")
