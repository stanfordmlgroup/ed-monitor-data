#!/usr/bin/env python

"""
Script to generate matched waveform, numerics data X min after waveforms start to be recorded.
Also include the next X min of numerics data.

Usage:
```
python -u /deep/u/tomjin/ed-monitor-self-supervised/preprocessing/generate_downstream_dataset.py --input-dir /deep/group/ed-monitor-self-supervised/v3/patient-data --input-file /deep/group/ed-monitor-self-supervised/v3/consolidated.csv --output-data-file /deep/group/ed-monitor-self-supervised/v4/downstream.15min.60min.head.h5 --output-summary-file /deep/group/ed-monitor-self-supervised/v4/downstream.15min.60min.head.csv --pre-minutes 15 --post-minutes 60 --max-patients 3
```

"""

import argparse
import csv
from concurrent import futures
from datetime import datetime, timedelta
from decimal import Decimal

from dateutil import parser as dt_parser
import traceback

import h5py
import numpy as np
import pandas as pd
from edm.utils.waveforms import WAVEFORMS_OF_INTERST, get_waveform
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

TYPE_II = "II"
TYPE_PLETH = "Pleth"
WAVEFORM_COLUMNS = [TYPE_II, TYPE_PLETH]
NUMERIC_COLUMNS = ['HR', 'RR', 'SpO2', 'btbRRInt_ms', 'NBPs', 'NBPd', 'Perf']
TARGET_FREQ = {
    TYPE_II: 500,
    #     TYPE_II: 125, // 125 is used for the self-supervised project
    TYPE_PLETH: 125,
}
VERBOSE = False


def get_waveform_offsets(start_time, current_time, freq, time_jumps):
    """
    Returns the offset in the waveform for the query time, taking into
    account the time_jumps that have occurred
    """

    # Find the latest time jump that is before the current_time.
    # Compute the number of seconds since the latest time jump
    current_time_to_time_jump_interval = None
    current_time_to_time_jump_time = None
    current_time_to_time_jump_pos = None
    for tj in time_jumps:
        # tj = ((pos_1, time_1), (pos_2, time_2))
        # Case #1: After gap
        # ---| |---
        #        | <- current_time
        # Case #2: Before gap
        # ---| |---
        #  |
        # Case #3: In between gap
        # ---| |---
        #     |
        interval_to_time_jump = Decimal(str(round(current_time.timestamp(), 3))) - Decimal(str(round(tj[1][1], 3)))
        if interval_to_time_jump >= 0 and (current_time_to_time_jump_time is None or interval_to_time_jump < current_time_to_time_jump_interval):
            current_time_to_time_jump_time = Decimal(str(round(tj[1][1], 3)))
            current_time_to_time_jump_pos = int(tj[1][0])
            current_time_to_time_jump_interval = interval_to_time_jump

    if current_time_to_time_jump_pos is None:
        start = ((current_time - start_time).total_seconds()) * freq
    else:
        start = (Decimal(str(round(current_time.timestamp(), 3))) - current_time_to_time_jump_time) * freq + current_time_to_time_jump_pos
    return int(start)


def get_numerics_averaged_by_second(numerics_obj, start_epoch, end_epoch, average_window_sec=60):
    start = int(start_epoch)
    end = int(end_epoch)
    range_len = end - start
    output = {}
    for col in NUMERIC_COLUMNS:
        if col not in numerics_obj:
            output[col] = np.full(int((end - start) / average_window_sec), np.NaN)
            output[f"{col}-time"] = np.full(int((end - start) / average_window_sec), np.NaN)
            output[f"{col}-length"] = 0
            continue
        second_to_vals = {}
        vals = np.array(numerics_obj[col])
        times = np.array(numerics_obj[f"{col}-time"])
        assert len(times) == len(vals)

        indices_of_interest = np.squeeze(np.argwhere(np.logical_and(times >= start, times <= end)), axis=-1)
        if len(indices_of_interest) > 1:
            vals = vals[indices_of_interest]
            times = times[indices_of_interest]
        elif len(indices_of_interest) == 1:
            vals = vals[[indices_of_interest]]
            times = times[[indices_of_interest]]
        else:
            # There are no usable numerics here
            output[col] = np.full(int((end - start) / average_window_sec), np.NaN)
            output[f"{col}-time"] = np.full(int((end - start) / average_window_sec), np.NaN)
            output[f"{col}-length"] = 0
            continue

        assert all(times[i] <= times[i+1] for i in range(len(times) - 1)), "Times are not sorted!"

        for idx, t in enumerate(times):
            # e.g. 
            # start = 1634840005
            # t     = 1634840068
            # average_window_sec = 60
            
            # offset = 63
            offset = int(t) - start

            # Determine the bucket - e.g. if we are grouping every 60 seconds,
            # we want to have buckets that begin on the 60 second marks
            # e.g. offset = 63 - 63 % 60 = 63 - 3 = 60
            # We would have buckets from 0, 60, 120, ...
            offset = offset - offset % average_window_sec
            if offset not in second_to_vals:
                second_to_vals[offset] = []
            second_to_vals[offset].append(vals[idx])

        output[col] = np.full(int((end - start) / average_window_sec), np.NaN)
        output[f"{col}-time"] = np.full(int((end - start) / average_window_sec), np.NaN)

        last_val = np.NaN
        valid_vals = 0
        curr = 0
        idx = 0
        while curr < range_len:
            actual_time = start + curr
            if curr in second_to_vals and len(second_to_vals[curr]) > 0:
                output[col][idx] = np.nanmean(second_to_vals[curr])
                valid_vals += 1
            elif not np.isnan(last_val):
                # Attempt to carry-forward the last value
                output[col][idx] = last_val
            last_val = output[col][idx]
            output[f"{col}-time"][idx] = actual_time
            curr += average_window_sec
            idx += 1
        output[f"{col}-length"] = valid_vals
    return output


def get_time_jump_window(f, waveform_type, start, seg_len):
    """
    Returns the time jump window, including the next position and next time stamp.
    Returns None, None if no timejump is applicable
    """
    if "waveforms_time_jumps" not in f:
        return None, None
    waveform_time_jumps = f["waveforms_time_jumps"][waveform_type]
    for tj in waveform_time_jumps:
        # tj = ((prev pos, prev time), (next pos, next time))
        next_pos = tj[1][0]
        next_time = tj[1][1]
        if start < next_pos < (start + seg_len):
            return next_pos, next_time
    return None, None


def get_best_waveforms(f, start_time, start_trim_sec, end_time, waveform_len_sec, stride_length_sec=10, csn=None):
    type_to_waveform = {}

    # We start looking for waveforms from the left, trying to find the first good waveform
    #
    current_time = start_time + timedelta(seconds=(start_trim_sec))
    best_waveforms = None
    best_qualities = []

    # Sanity check that the waveforms are all actually present and not just empty array (saves time)
    for waveform_type in WAVEFORM_COLUMNS:
        waveform_base = np.array(f["waveforms"][waveform_type])
        if len(waveform_base) == 0 or (waveform_base == waveform_base[0]).all():
            # The waveform is just empty so skip this patient
            raise Exception(f"Waveform {waveform_type} is empty or flat-line")

    # For each sliding window...
    while current_time <= (end_time - timedelta(seconds=waveform_len_sec)):
#         print(f"[{datetime.now().isoformat()}] [{csn}] Checking window {current_time}")

        # Quality check must pass for all waveform types
        local_waveforms = []
        local_qualities = []

        # Time jump flag - if set to true, we must move the sliding window over to the next time jump slot
        time_jumped = False
        time_jump_start_time = None

        # Get waveforms for current window
        for waveform_type in WAVEFORM_COLUMNS:
            waveform_config = WAVEFORMS_OF_INTERST[waveform_type]

            if waveform_type not in f["waveforms"]:
                raise Exception(f"Waveform {waveform_type} not available")
            waveform_base = f["waveforms"][waveform_type]
            waveforms_time_jumps = f["waveforms_time_jumps"][waveform_type] if "waveforms_time_jumps" in f else []

            start = get_waveform_offsets(start_time, current_time, waveform_config["orig_frequency"], waveforms_time_jumps)
            seg_len = int(waveform_len_sec * waveform_config["orig_frequency"])

            time_jump_next_pos, time_jump_next_time = get_time_jump_window(f, waveform_type, start, seg_len)
            if time_jump_next_pos is not None:
                # A time jump occurred, so we must move the sliding window
                time_jumped = True
                time_jump_start_time = time_jump_next_time
                break

            if len(waveform_base[start:(start + seg_len)]) < seg_len:
                # If the waveform is not of expected size, there is no point continuing
                raise Exception(f"Waveform {waveform_type} not usable")
            try:
                waveform, quality = get_waveform(waveform_base, start, seg_len,
                                                 waveform_config["orig_frequency"],
                                                 should_normalize=False,
                                                 bandpass_type=waveform_config["bandpass_type"],
                                                 bandwidth=waveform_config["bandpass_freq"],
                                                 target_fs=TARGET_FREQ[waveform_type],
                                                 waveform_type=waveform_type,
                                                 skewness_max=0.87,
                                                 msq_min=0.27)
            except Exception as et:
                # Fail fast because this represents an error case - we assumed that the windows in this range 
                # are all good so we should raise an error when we get into this exceptional case
                print(f"[{datetime.now().isoformat()}] [{csn}] A window at start={start} could not be parsed due to {et}")
                raise et

            local_waveforms.append(waveform)
            local_qualities.append(quality)

        if time_jumped:
            # We must forcibly move the sliding window to the next time jump point
            print(f"[{datetime.now().isoformat()}] [{csn}] Time jump occurred")
            current_time = datetime.fromtimestamp(time_jump_start_time, tz=current_time.tzinfo)
            continue
        # Ensure quality is met for all waveforms
        #
        elif sum(local_qualities) == len(WAVEFORM_COLUMNS):
            # Great, we have found an acceptable waveform
            best_waveforms = local_waveforms
            best_qualities = local_qualities
#             print(f"[{datetime.now().isoformat()}] [{csn}] Good window found")
            break
        else:
            # Continue searching the next window to see if the waveform is any better
            # We always keep the last waveform with the assumption that waveforms get
            # better in further windows (e.g. setup issues in initial windows) unless
            # the quality got worse

            if best_waveforms is None or sum(best_qualities) <= sum(local_qualities):
                best_waveforms = local_waveforms
                best_qualities = local_qualities

            current_time += timedelta(seconds=stride_length_sec)
#             print(f"[{datetime.now().isoformat()}] [{csn}] Bad window found, continuing search")
            continue

    if best_waveforms is None:
        raise Exception("No acceptable windows found")
    
    for k, waveform_type in enumerate(WAVEFORM_COLUMNS):
        waveform = best_waveforms[k]
#         print(f"[{datetime.now().isoformat()}] [{csn}] Best waveform had shape {waveform.shape}")

        if len(waveform) > int(waveform_len_sec * TARGET_FREQ[waveform_type]):
            waveform = waveform[int(waveform_len_sec * TARGET_FREQ[waveform_type]) - len(waveform):]
        elif len(waveform) < int(waveform_len_sec * TARGET_FREQ[waveform_type]):
            waveform = np.pad(waveform, (int(waveform_len_sec * TARGET_FREQ[waveform_type]) - len(waveform), 0))

        type_to_waveform[waveform_type] = {
            "waveform": waveform,
            "quality": best_qualities[k]
        }

    return type_to_waveform


def get_min_numerics_time(numerics_obj, roomed_time):
    min_time = None
    for col in NUMERIC_COLUMNS:
        if f"{col}-time" in numerics_obj and len(numerics_obj[f"{col}-time"]) > 0:
            if min_time is None or numerics_obj[f"{col}-time"][0] < min_time:
                min_time = numerics_obj[f"{col}-time"][0]
    if min_time is None:
        raise Exception("The patient did not have any numerics!")
    else:
        return datetime.fromtimestamp(min_time, tz=roomed_time.tzinfo)


def process_patient(input_args):
    i, df, csn, input_folder, waveform_length_sec, pre_minutes_min, post_minutes_min, pre_granularity_sec, post_granularity_sec, align_col, align_type = input_args

    filename = f"{input_folder}/{str(csn)[-2:]}/{csn}.h5"
    print(f"[{datetime.now().isoformat()}] [{i}/{df.shape[0]}] Working on patient {csn} at {filename}")
    try:

        output = {
            "csn": csn,
            "alignment_times": [],
            "numerics_before": {},
            "numerics_after": {},
            "waveforms": {},
        }
        for col in NUMERIC_COLUMNS:
            output["numerics_before"][col] = {
                "vals": [],
                "times": [],
                "lengths": [],
            }
            output["numerics_after"][col] = {
                "vals": [],
                "times": [],
                "lengths": [],
            }
        for col in WAVEFORM_COLUMNS:
            output["waveforms"][col] = {
                "waveforms": [],
                "qualities": [],
            }

        with h5py.File(filename, "r") as f:
            row = df[df["patient_id"] == csn]

            roomed_time = row["roomed_time"].item()
            roomed_time = datetime.strptime(roomed_time, '%Y-%m-%d %H:%M:%S%z')

            if align_col is not None and not pd.isna(row[align_col].item()):
                max_alignment_time = row[align_col].item()
                max_alignment_time = dt_parser.parse(max_alignment_time).replace(tzinfo=roomed_time.tzinfo)
            else:
                max_alignment_time = None

            if align_type == "waveform":
                # start_time is when the waveform monitoring starts (note that anything before the
                # recommended trim was an empty array, even though it might have technically been
                # part of the patient's visit)

                waveform_start = row["waveform_start_time"].item()
                waveform_start = datetime.strptime(waveform_start, '%Y-%m-%d %H:%M:%S.%f%z')
                recommended_trim_start_sec = int(row["recommended_trim_start_sec"].item())
                start_time = waveform_start + timedelta(seconds=(recommended_trim_start_sec))
            elif align_type == "waveform_optional" or align_type == "numerics":
                has_required_waveforms = row["II_available"].item() == 1 and row["Pleth_available"].item() == 1
                has_usable_waveforms = row["recommended_trim_start_sec"].item() != 0 or row["recommended_trim_end_sec"].item() != 0
                if has_required_waveforms and has_usable_waveforms:
                    waveform_start = row["waveform_start_time"].item()
                    waveform_start = datetime.strptime(waveform_start, '%Y-%m-%d %H:%M:%S.%f%z')
                    recommended_trim_start_sec = int(row["recommended_trim_start_sec"].item())
                else:
                    waveform_start = None
                    recommended_trim_start_sec = None

                if align_type == "waveform_optional" and waveform_start is not None:
                    start_time = waveform_start + timedelta(seconds=(recommended_trim_start_sec))
                else:
                    # start time forced to be the numerics start
                    start_time = get_min_numerics_time(f["numerics"], roomed_time)
            else:
                raise Exception(f"Unknown align_type={align_type} provided")

            alignment_time = start_time + timedelta(seconds=(pre_minutes_min * 60))

            if max_alignment_time is not None:
                # The idea is that we take either the minimum of the alignment time we derived from
                # the specified pre minutes or the max alignment time. Example:
                # Say we had pre-minutes of 10 min
                #
                # -------------------------
                #    ^
                #    calculated alignment time
                #             ^ max alignment time
                #  We choose the actual alignment time here to be the calculated alignment time.
                #
                if max_alignment_time.timestamp() < alignment_time.timestamp():
                    alignment_time = max_alignment_time

            end_time = alignment_time + timedelta(seconds=(post_minutes_min * 60))

            start_time_epoch = start_time.timestamp()
            alignment_time_epoch = alignment_time.timestamp()
            end_time_epoch = end_time.timestamp()

            print(f"[{datetime.now().isoformat()}] [{csn}] Getting best waveforms...")

            if VERBOSE:
                print(f"start_time = {start_time}")
                print(f"max_alignment_time = {max_alignment_time}")
                print(f"alignment_time = {alignment_time}")
                print(f"end_time = {end_time}")

            try:
                type_to_waveform_obj = get_best_waveforms(f, waveform_start, recommended_trim_start_sec, alignment_time, waveform_length_sec, csn=csn)
            except Exception as ec:
                if align_type == "waveform":
                    raise Exception(f"Patient did not have any usable waveforms. Technical: {ec}")
                else:
                    # Waveforms are optional for other alignment types
                    type_to_waveform_obj = {}
                    for waveform_type in WAVEFORM_COLUMNS:
                        type_to_waveform_obj[waveform_type] = {
                            "waveform": np.full(TARGET_FREQ[waveform_type] * waveform_length_sec, np.NaN),
                            "quality": 0
                        }

            numerics_map_before = get_numerics_averaged_by_second(f["numerics"], start_time_epoch, alignment_time_epoch, pre_granularity_sec)
            for col in NUMERIC_COLUMNS:
                output["numerics_before"][col]["vals"].append(numerics_map_before[col])
                output["numerics_before"][col]["times"].append(numerics_map_before[f"{col}-time"])
                output["numerics_before"][col]["lengths"].append(numerics_map_before[f"{col}-length"])

            if post_minutes_min > 0:
                numerics_map_after = get_numerics_averaged_by_second(f["numerics"], alignment_time_epoch, end_time_epoch, post_granularity_sec)
                for col in NUMERIC_COLUMNS:
                    output["numerics_after"][col]["vals"].append(numerics_map_after[col])
                    output["numerics_after"][col]["times"].append(numerics_map_after[f"{col}-time"])
                    output["numerics_after"][col]["lengths"].append(numerics_map_after[f"{col}-length"])

            output["alignment_times"].append(alignment_time_epoch)

            for waveform_type in WAVEFORM_COLUMNS:
                output["waveforms"][waveform_type]["waveforms"].append(type_to_waveform_obj[waveform_type]["waveform"])
                output["waveforms"][waveform_type]["qualities"].append(type_to_waveform_obj[waveform_type]["quality"])

        output_str = f"[{datetime.now().isoformat()}] [{i}/{df.shape[0]}]   [{csn}] "
        for k in WAVEFORM_COLUMNS:
            output_str += f"[{k}]: {type_to_waveform_obj[k]['quality']} valid | "
        print(output_str)
        return output
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] [ERROR] for patient {csn} due to {e}")
        print(traceback.format_exc())
        return None


def run(input_folder, input_file, output_data_file, output_summary_file,
        waveform_length_sec, pre_minutes_min, post_minutes_min,
        pre_granularity_sec, post_granularity_sec, align_col, align_type, limit):
    df = pd.read_csv(input_file)
    patients = df["patient_id"].tolist()

    csns = []
    alignment_times = []
    numerics_before = {}
    numerics_after = {}
    waveforms = {}
    for col in NUMERIC_COLUMNS:
        numerics_before[col] = {
            "vals": [],
            "times": [],
            "lengths": [],
        }
        numerics_after[col] = {
            "vals": [],
            "times": [],
            "lengths": [],
        }
    for col in WAVEFORM_COLUMNS:
        waveforms[col] = {
            "waveforms": [],
            "qualities": [],
        }

    fs = []
    with futures.ThreadPoolExecutor(32) as executor:
        i = 0
        for csn in tqdm(patients, disable=True):
            i += 1
            if limit is not None and i > limit:
                break

            input_args = [i, df, csn, input_folder, waveform_length_sec, pre_minutes_min, post_minutes_min, pre_granularity_sec, post_granularity_sec, align_col, align_type]
            future = executor.submit(process_patient, input_args)
            fs.append(future)

    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result = future.result(timeout=60 * 60)
        if result is not None:
            csns.extend([result["csn"] for t in result["alignment_times"]])
            alignment_times.extend(result["alignment_times"])
            for col in NUMERIC_COLUMNS:
                numerics_before[col]["vals"].extend(result["numerics_before"][col]["vals"])
                numerics_before[col]["times"].extend(result["numerics_before"][col][f"times"])
                numerics_before[col]["lengths"].extend(result["numerics_before"][col][f"lengths"])
                numerics_after[col]["vals"].extend(result["numerics_after"][col]["vals"])
                numerics_after[col]["times"].extend(result["numerics_after"][col][f"times"])
                numerics_after[col]["lengths"].extend(result["numerics_after"][col][f"lengths"])
            for col in WAVEFORM_COLUMNS:
                waveforms[col]["waveforms"].extend(result["waveforms"][col]["waveforms"])
                waveforms[col]["qualities"].extend(result["waveforms"][col]["qualities"])

    with h5py.File(output_data_file, "w") as f:
        f.create_dataset("alignment_times", data=alignment_times)
        dset_before = f.create_group("numerics_before")
        dset_after = f.create_group("numerics_after")
        for k in NUMERIC_COLUMNS:
            dset_k = dset_before.create_group(k)
            dset_k.create_dataset(f"vals", data=numerics_before[k]["vals"])
            dset_k.create_dataset(f"times", data=numerics_before[k][f"times"])
            dset_k = dset_after.create_group(k)
            dset_k.create_dataset(f"vals", data=numerics_after[k]["vals"])
            dset_k.create_dataset(f"times", data=numerics_after[k][f"times"])

        dset = f.create_group("waveforms")
        for k in WAVEFORM_COLUMNS:
            dset_k = dset.create_group(k)
            dset_k.create_dataset("waveforms", data=waveforms[k]["waveforms"])
            dset_k.create_dataset("qualities", data=waveforms[k]["qualities"])

    with open(output_summary_file, "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        headers = ["patient_id", "alignment_time"]
        for k in NUMERIC_COLUMNS:
            headers.append(f"{k}_before_length")
            headers.append(f"{k}_after_length")
        for k in WAVEFORM_COLUMNS:
            headers.append(f"{k}_length")
            headers.append(f"{k}_quality")
        writer.writerow(headers)

        i = 0
        while i < len(alignment_times):
            row = [csns[i], alignment_times[i]]
            for k in NUMERIC_COLUMNS:
                row.append(numerics_before[k]["lengths"][i])
                if post_minutes_min > 0:
                    row.append(numerics_after[k]["lengths"][i])
                else:
                    row.append("")
            for k in WAVEFORM_COLUMNS:
                row.append(len(waveforms[k]["waveforms"][i]))
                row.append(waveforms[k]["qualities"][i])
            writer.writerow(row)
            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input-dir',
                        required=True,
                        help='Where the data files are located')
    parser.add_argument('-f', '--input-file',
                        required=True,
                        help='Where the summary is located')
    parser.add_argument('-od', '--output-data-file',
                        required=True,
                        help='Where the output data file is located')
    parser.add_argument('-os', '--output-summary-file',
                        required=True,
                        help='Where the output summary file is located')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')
    parser.add_argument('-w', '--waveform-length',
                        default=10,
                        help='Waveform length to take (sec)')
    parser.add_argument('-pr', '--pre-minutes',
                        default=15,
                        help='Number of minutes since beginning of waveform recording to use as the initial window of high resolution input data')
    parser.add_argument('-po', '--post-minutes',
                        default=60,
                        help='Number of minutes of averaged data after the high resolution window to use as the prediction outcome')
    parser.add_argument('-gpr', '--pre-granularity-sec',
                        default=1,
                        help='Granularity to use pre-alignment (e.g. 1 means to produce data at a 1 sec resolution)')
    parser.add_argument('-gpo', '--post-granularity-sec',
                        default=60,
                        help='Granularity to use post-alignment (e.g. 60 means to produce data averaged over 60 sec ranges)')
    parser.add_argument('-a', '--align',
                        default=None,
                        help='Specify a column to use as the maximum alignment column. e.g. we might want to collect as much numerics before the first blood draw time')
    parser.add_argument('-at', '--align-type',
                        default='waveform',
                        help='Specifies the type of alignment. If "waveform", then alignment is relative to the start of the waveform recording and waveforms are required. '
                             + 'If "waveform_optional", then alignment is relative to the start of the waveform recording if possible, otherwise, relative to the start of the numerics. '
                             + 'If "numerics", then alignment is relative to the start of the numerics'
                        )

    args = parser.parse_args()

    # Where the data files are located
    input_dir = args.input_dir

    # Where the summary is located
    input_file = args.input_file

    # Where the output data file is located
    output_data_file = args.output_data_file

    # Where the output summary file is located
    output_summary_file = args.output_summary_file

    waveform_length_sec = int(args.waveform_length)
    pre_minutes_min = int(args.pre_minutes)
    post_minutes_min = int(args.post_minutes)
    pre_granularity_sec = int(args.pre_granularity_sec)
    post_granularity_sec = int(args.post_granularity_sec)

    limit = int(args.max_patients) if args.max_patients is not None else None

    align_col = args.align
    align_type = args.align_type

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_data_file={output_data_file}, output_summary_file={output_summary_file}")
    print(
        f"waveform_length_sec={waveform_length_sec}, pre_minutes_min={pre_minutes_min}, post_minutes_min={post_minutes_min}")
    print(
        f"pre_granularity_sec={pre_granularity_sec}, post_granularity_sec={post_granularity_sec}, align_type={align_type}")
    print("-" * 30)

    run(input_dir, input_file, output_data_file, output_summary_file, waveform_length_sec, pre_minutes_min,
        post_minutes_min, pre_granularity_sec, post_granularity_sec, align_col, align_type, limit)

    print("DONE")
