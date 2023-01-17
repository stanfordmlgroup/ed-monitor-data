#!/usr/bin/env python

"""
Script to generate matched waveform, numerics data 15 min after waveforms start to be recorded.
Also include the next 60 min of numerics data.

Usage:
```
python -u /deep/u/tomjin/ed-monitor-self-supervised/preprocessing/generate_downstream_dataset.py --input-dir /deep/group/ed-monitor-self-supervised/v3/patient-data --input-file /deep/group/ed-monitor-self-supervised/v3/consolidated.csv --output-data-file /deep/group/ed-monitor-self-supervised/v4/downstream.15min.60min.head.h5 --output-summary-file /deep/group/ed-monitor-self-supervised/v4/downstream.15min.60min.head.csv --pre-minutes 15 --post-minutes 60 --max-patients 3
```

"""

import argparse
import csv
from concurrent import futures
from datetime import datetime, timedelta
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


def get_waveform_offsets(start_time, current_time, freq):
    """
    Returns the offset in the waveform for the query time
    """
    start = ((current_time - start_time).total_seconds()) * freq
    return max(int(start), 0)


def get_numerics(numerics_obj, start_epoch, end_epoch, pre_numerics_len_sec):
    start = start_epoch
    end = end_epoch
    output = {}
    for col in NUMERIC_COLUMNS:
        output[col] = np.zeros(
            pre_numerics_len_sec)  # Roughly reserve space for 1 data point per sec (required to obtain a square matrix)
        output[col][:] = np.NaN
        output[f"{col}-time"] = np.zeros(pre_numerics_len_sec)
        output[f"{col}-length"] = 0
        if col in numerics_obj:
            vals = np.array(numerics_obj[col])
            times = np.array(numerics_obj[f"{col}-time"])
            indices_of_interest = np.argwhere(np.logical_and(times >= start, times <= end))
            # Take only the values closest to the alignment point
            indices_of_interest = indices_of_interest[-pre_numerics_len_sec:]
            offset = max(0, pre_numerics_len_sec - len(indices_of_interest))
            output[col][offset:] = np.squeeze(vals[indices_of_interest])
            output[f"{col}-time"][offset:] = np.squeeze(times[indices_of_interest]).astype(int)
            output[f"{col}-length"] = len(indices_of_interest)
    return output


def get_numerics_averaged_by_minute(numerics_obj, start_epoch, end_epoch, post_numerics_min):
    start = start_epoch
    end = end_epoch
    output = {}
    for col in NUMERIC_COLUMNS:
        output[col] = np.zeros(post_numerics_min)
        output[col][:] = np.NaN
        output[f"{col}-time"] = np.zeros(post_numerics_min)
        output[f"{col}-length"] = 0

        if col in numerics_obj:
            vals = np.array(numerics_obj[col])
            times = np.array(numerics_obj[f"{col}-time"])
            indices_of_interest = np.argwhere(np.logical_and(times >= start, times <= end))
            if VERBOSE:
                print(f"len(vals) = {len(vals)}")
                print(f"len(times) = {len(times)}")
                print(f"start = {datetime.fromtimestamp(start).isoformat('T')}")
                print(f"end = {datetime.fromtimestamp(end).isoformat('T')}")
                print(f"len(indices_of_interest) = {len(indices_of_interest)}")

            temp_min = 0
            temp_list = []
            valid_vals = 0
            for idx in indices_of_interest:
                #                 if VERBOSE:
                #                     print(f"idx = {idx}")
                if temp_min >= post_numerics_min:
                    break
                if (start + temp_min * 60) <= times[idx] < (start + (temp_min + 1) * 60):
                    temp_list.append(vals[idx])
                else:
                    output[col][temp_min] = np.mean(temp_list)
                    output[f"{col}-time"][temp_min] = start + temp_min * 60
                    if len(temp_list) > 0:
                        valid_vals += 1
                    temp_list = [vals[idx]]
                    temp_min += 1

            if len(temp_list) > 0 and temp_min < post_numerics_min:
                output[col][temp_min] = np.mean(temp_list)
                output[f"{col}-time"][temp_min] = start + temp_min * 60

            output[f"{col}-length"] = valid_vals
    return output


def get_best_waveforms(f, start_time, start_trim_sec, end_time, waveform_len_sec, stride_length_sec=10, csn=None):
    type_to_waveform = {}

    # We start looking for waveforms from the left, trying to find the first good waveform
    #
    current_time = start_time + timedelta(seconds=(start_trim_sec))
    best_waveforms = None
    best_qualities = []

    # Sanity check that the waveforms are all actually present and not just empty array (saves time)
    for waveform_type in WAVEFORM_COLUMNS:
        waveform_config = WAVEFORMS_OF_INTERST[waveform_type]
        waveform_base = np.array(f["waveforms"][waveform_type])
        if len(waveform_base) == 0 or (waveform_base == waveform_base[0]).all():
            # The waveform is just empty so skip this patient
            raise Exception(f"Waveform {waveform_type} is empty")

    # For each sliding window...
    while current_time <= (end_time - timedelta(seconds=waveform_len_sec)):
#         print(f"[{datetime.now().isoformat()}] [{csn}] Checking window {current_time}")

        # Quality check must pass for all waveform types
        local_waveforms = []
        local_qualities = []

        # Get waveforms for current window
        for waveform_type in WAVEFORM_COLUMNS:
            waveform_config = WAVEFORMS_OF_INTERST[waveform_type]

            waveform_base = f["waveforms"][waveform_type]
            start = get_waveform_offsets(start_time, current_time, waveform_config["orig_frequency"])
            seg_len = int(waveform_len_sec * waveform_config["orig_frequency"])
            if len(waveform_base[start:(start + seg_len)]) < seg_len:
                # If the waveform is not of expected size, there is no point continuing
                continue
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

        # Ensure quality is met for all waveforms
        #
        if sum(local_qualities) == len(WAVEFORM_COLUMNS):
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


def process_patient(input_args):
    i, df, csn, input_folder, waveform_length_sec, pre_minutes_min, post_minutes_min, align_col = input_args

    filename = f"{input_folder}/{csn}/{csn}.h5"
    print(f"[{datetime.now().isoformat()}] [{i}/{df.shape[0]}] Working on patient {csn} at {filename}")
    try:

        output = {
            "csn": csn,
            "alignment_times": [],
            "alignment_vals": [],  # Deprecated, kept for compatibility
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

            waveform_start = row["waveform_start_time"].item()
            waveform_start = datetime.strptime(waveform_start, '%Y-%m-%d %H:%M:%S%z')
            if align_col is not None and not pd.isna(row[align_col].item()):
                max_alignment_time = row[align_col].item()
                max_alignment_time = dt_parser.parse(max_alignment_time).replace(tzinfo=waveform_start.tzinfo)
            else:
                max_alignment_time = None

            recommended_trim_start_sec = int(row["recommended_trim_start_sec"].item())

            # start_time is when the waveform monitoring starts (note that anything before the
            # recommended trim was an empty array, even though it might have technically been
            # part of the patient's visit)
            start_time = waveform_start + timedelta(seconds=(recommended_trim_start_sec))
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

            start_time_epoch = int(start_time.timestamp())
            alignment_time_epoch = int(alignment_time.timestamp())
            end_time_epoch = int(end_time.timestamp())

            print(f"[{datetime.now().isoformat()}] [{csn}] Getting best waveforms...")

            if VERBOSE:
                print(f"start_time = {start_time}")
                print(f"max_alignment_time = {max_alignment_time}")
                print(f"alignment_time = {alignment_time}")
                print(f"end_time = {end_time}")

            try:
                type_to_waveform_obj = get_best_waveforms(f, waveform_start, recommended_trim_start_sec, alignment_time, waveform_length_sec, csn=csn)
            except Exception as ec:
                raise Exception(f"Patient did not have any useable waveforms. Technical: {ec}")

#             print(f"[{datetime.now().isoformat()}] [{csn}] Got best waveforms")

            numerics_map_before = get_numerics_averaged_by_minute(f["numerics"], start_time_epoch, alignment_time_epoch, pre_minutes_min)
            for col in NUMERIC_COLUMNS:
                output["numerics_before"][col]["vals"].append(numerics_map_before[col])
                output["numerics_before"][col]["times"].append(numerics_map_before[f"{col}-time"])
                output["numerics_before"][col]["lengths"].append(numerics_map_before[f"{col}-length"])

#             print(f"[{datetime.now().isoformat()}] [{csn}] Got before numerics")
            numerics_map_after = get_numerics_averaged_by_minute(f["numerics"], alignment_time_epoch, end_time_epoch,
                                                                 post_minutes_min)
#             print(f"[{datetime.now().isoformat()}] [{csn}] Got after numerics")
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
#         print(traceback.format_exc())
        return None
#         raise e


def run(input_folder, input_file, output_data_file, output_summary_file,
        waveform_length_sec, pre_minutes_min, post_minutes_min, align_col, limit):
    df = pd.read_csv(input_file)
    patients = df["patient_id"].tolist()

    csns = []
    alignment_times = []
    alignment_vals = []
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

            input_args = [i, df, csn, input_folder, waveform_length_sec, pre_minutes_min, post_minutes_min, align_col]
            future = executor.submit(process_patient, input_args)
            fs.append(future)

    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result = future.result(timeout=60 * 60)
        if result is not None:
            csns.extend([result["csn"] for t in result["alignment_times"]])
            alignment_times.extend(result["alignment_times"])
            alignment_vals.extend(result["alignment_vals"])
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
        f.create_dataset("alignment_vals", data=alignment_vals)
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
        headers = ["patient_id", "alignment_time", "alignment_val"]
        for k in NUMERIC_COLUMNS:
            headers.append(f"{k}_before_length")
            headers.append(f"{k}_after_length")
        for k in WAVEFORM_COLUMNS:
            headers.append(f"{k}_length")
            headers.append(f"{k}_quality")
        writer.writerow(headers)

        i = 0
        while i < len(alignment_times):
            row = [csns[i], alignment_times[i], ""]
            for k in NUMERIC_COLUMNS:
                row.append(numerics_before[k]["lengths"][i])
                row.append(numerics_after[k]["lengths"][i])
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
    parser.add_argument('-a', '--align',
                        default=None,
                        help='Specify a column to use as the maximum alignment column. e.g. we might want to collect as much numerics before the first blood draw time')

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

    limit = int(args.max_patients) if args.max_patients is not None else None

    align_col = args.align

    print("=" * 30)
    print(
        f"Starting data generation with input_dir={input_dir}, input_file={input_file}, output_data_file={output_data_file}, output_summary_file={output_summary_file}")
    print(
        f"waveform_length_sec={waveform_length_sec}, pre_minutes_min={pre_minutes_min}, post_minutes_min={post_minutes_min}")
    print("-" * 30)

    run(input_dir, input_file, output_data_file, output_summary_file, waveform_length_sec, pre_minutes_min,
        post_minutes_min, align_col, limit)

    print("DONE")
