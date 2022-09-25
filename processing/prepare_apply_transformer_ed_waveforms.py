#!/usr/bin/env python

"""
Script to prepare the ED-acquired Philips waveforms, but also applies the Transformer model to create embeddings
that are written to a single file.

This is useful to produce the embeddings for consecutive waveforms, since
the number of waveforms differ in each patient so there is no point in creating
an intermediate waveform file that the `prepare_ed_waveforms.py` script will normally 
do sine this file would be extremely large.

Example: python prepare_apply_transformer_ed_waveforms.py -i /deep/group/ed-monitor/patient_data_v9/consolidated.filtered.test.txt -d /deep/group/ed-monitor/patient_data_v9/patient-data -o /deep/group/ed-monitor/patient_data_v9/waveforms -l 15 -f 500 -n -w II -b First_trop_result_time-waveform_start_time -mo /deep/u/tomjin/aihc-aut20-selfecg/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar -m 120 -p 3
"""

import argparse
import csv
import datetime
import pickle
from concurrent import futures
from pathlib import Path
import time
from shutil import copyfile

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
torch.multiprocessing.set_start_method('spawn', force=True)

from edm.models.transformer_model import load_best_model
from edm.utils.waveforms import WAVEFORMS_OF_INTERST, get_waveform

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

PATIENCE = 10 # number of rounds before we give up trying to find a non-empty waveform


def load_pkl_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def process_records(input_args):
    j, args, total_rows, waveform_df, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, sample_before = input_args
    
    model_path = args.model_path
    remove_last_layer = bool(int(args.remove_last_layer))
    deepfeat_sz = int(args.deepfeat_sz)
    max_lead_time = int(args.max_lead_time) if args.max_lead_time is not None else None

    print(f"[{j}] Loading model from {model_path}", flush=True)
    model = load_best_model(model_path, deepfeat_sz=deepfeat_sz, remove_last_layer=remove_last_layer)
    model.eval()
    print(f"[{j}] Loaded model from {model_path}", flush=True)

    waveforms = {}
    for w in waveform_types:
        waveforms[w] = []

    for i, row in waveform_df.iterrows():
        result = process_record([j, i, total_rows, waveform_df, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, sample_before, max_lead_time, model])
        
        if result is not None:
            for w in result.keys():
                waveforms[w].extend(result[w])
    return waveforms


def process_record(input_args):
    j, i, total_rows, waveform_df, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, sample_before, max_lead_time, model = input_args

    try:
        patient_id = waveform_df.iloc[[i]]["patient_id"].item()
        recommended_trim_start_sec = waveform_df.iloc[[i]]["recommended_trim_start_sec"].item()
        recommended_trim_end_sec = waveform_df.iloc[[i]]["recommended_trim_end_sec"].item()
        sample_before_val = int(waveform_df.iloc[[i]][sample_before].item()) if sample_before is not None else None
        info = load_pkl_file(f"{patient_dir}/{patient_id}/info.pkl")

        waveforms = {}
        for waveform_type in waveform_types:

            waveform_base = np.load(f"{patient_dir}/{patient_id}/{waveform_type}.dat.npy")

            waveform_config = WAVEFORMS_OF_INTERST[waveform_type]
            fs = waveform_config["orig_frequency"]

            if waveform_type in info["supported_types"]:
                window_size = int(waveform_length * fs)

                start_offset = int(max(0, recommended_trim_start_sec * fs))
                end_offset = int(min(len(waveform_base), recommended_trim_end_sec * fs))
                if sample_before_val is not None:
                    end_offset = min(end_offset, sample_before_val * fs)

                pointer = start_offset
                lead_time = 0
                num_done = 0
                
                # Accumulate waveforms in order to call model with batches in order to make it more performant
                waveforms_so_far = []
                infos_so_far = []
                while pointer < (end_offset - window_size):

                    waveform, quality = get_waveform(waveform_base, pointer, window_size, fs, should_normalize=should_normalize, bandpass_type=waveform_config["bandpass_type"], bandwidth=waveform_config["bandpass_freq"], target_fs=waveform_target_freq, ecg_quality_check=True)

                    print(f"[{j}] [{i}/{total_rows}] {patient_id} waveform {waveform_type} of shape {waveform.shape} at pointer={pointer}, lead_time={lead_time}", flush=True)

                    waveforms_so_far.append([waveform])
                    infos_so_far.append({
                        "record_name": patient_id,
                        "pointer": pointer,
                        "quality": quality,
                        "lead_time": lead_time
                    })
                    
                    if len(waveforms_so_far) > 8:
                        input_tensor = torch.tensor(np.array(waveforms_so_far), dtype=torch.float32).to(device)
                        embeddings = model(input_tensor).detach().cpu().numpy()
                        del input_tensor

                        for k, waveform_so_far in enumerate(waveforms_so_far):
                            if waveform_type not in waveforms:
                                waveforms[waveform_type] = []
                            waveforms[waveform_type].append({
                                "record_name": infos_so_far[k]["record_name"],
                                "pointer": infos_so_far[k]["pointer"],
                                "embeddings": embeddings[k],
                                "quality": infos_so_far[k]["quality"],
                                "lead_time": infos_so_far[k]["lead_time"]
                            })
                        waveforms_so_far = []
                        infos_so_far = []

                    pointer += window_size
                    lead_time += waveform_length
                    num_done += 1
                    
                    if max_lead_time is not None and lead_time > max_lead_time:
                        print(f"[{j}] [{i}/{total_rows}] {patient_id} early termination at lead_time={lead_time}", flush=True)
                        break

                if len(waveforms_so_far) > 0:
                    input_tensor = torch.tensor(np.array(waveforms_so_far), dtype=torch.float32).to(device)
                    embeddings = model(input_tensor).detach().cpu().numpy()
                    del input_tensor

                    for k, waveform_so_far in enumerate(waveforms_so_far):
                        if waveform_type not in waveforms:
                            waveforms[waveform_type] = []
                        waveforms[waveform_type].append({
                            "record_name": infos_so_far[k]["record_name"],
                            "pointer": infos_so_far[k]["pointer"],
                            "embeddings": embeddings[k],
                            "quality": infos_so_far[k]["quality"],
                            "lead_time": infos_so_far[k]["lead_time"]
                        })
                    waveforms_so_far = []
                    infos_so_far = []

                print(f"[{j}] [{i}/{total_rows}] {patient_id} had {num_done} waveforms", flush=True)

        return waveforms
    except Exception as e:
        print(f"[{j}] [{i}/{total_rows}] Unexpected error:", e)
        return None
    

def run(args):
    print(f"START TIME: {datetime.datetime.now()}")
    waveform_types = args.waveform_types.split(",")
    input_file = args.waveform_file
    patient_dir = args.patient_dir
    should_normalize = args.normalize
    output_folder = args.output_folder
    max_patients = int(args.max_patients) if args.max_patients is not None else None
    waveform_length = int(args.length)
    waveform_target_freq = float(args.frequency)
    waveform_types = waveform_types
    sample_before = args.sample_before
    model_path = args.model_path
    remove_last_layer = bool(int(args.remove_last_layer))
    deepfeat_sz = int(args.deepfeat_sz)

    output_folder = f"{output_folder}/{waveform_length}sec-{int(waveform_target_freq)}hz-{int(should_normalize)}norm-all/transformer-{deepfeat_sz}"
    if not remove_last_layer:
        output_folder = f"{output_folder}-{deepfeat_sz}"

    completed_pts = set()
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for w in waveform_types:
        Path(f"{output_folder}/{w}").mkdir(parents=True, exist_ok=True)

        if Path(f"{output_folder}/{w}/summary.csv").is_file():
            # This script could prematurely terminate due to OOM on the GPU if not enough resources are allocated.
            # To remedy this, we will first check for the existence of the output file. 
            # We will discount any patients who have already been processed and just continue with the remaining 
            # patients to avoid reprocessing the files.
            df_existing = pd.read_csv(f"{output_folder}/{w}/summary.csv")
            print(f"Found existing file with shape {df_existing.shape} and {df_existing['patient_id'].unique().shape} existing pts")
            completed_pts.update(df_existing['patient_id'].unique().tolist())
    
    print(f"Found {len(completed_pts)} completed patients")
    
    df = pd.read_csv(input_file, sep="\t")
    print(f"Found {input_file} with shape {df.shape}")
    
    df = df[~df["patient_id"].isin(completed_pts)]
    print(f"After eliminating completed patients {input_file} has shape {df.shape}")
    
    total_rows = len(df)

    if max_patients is not None:
        df = df.head(max_patients)

    fs = []
    with futures.ProcessPoolExecutor(6) as executor:
        dfs = np.array_split(df, 6)
        j = 0
        for df_sub in dfs:
            df_sub.reset_index(inplace=True)
            if df_sub.shape[0] > 0:
                input_args = [j, args, total_rows, df_sub, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, sample_before]
                print(f"Submitting for process {j}")
                future = executor.submit(process_records, input_args)
                fs.append(future)
            j += 1

    waveforms = {}
    for w in waveform_types:
        waveforms[w] = []
    
    for future in futures.as_completed(fs):
        # Blocking call - wait for 1 hour for a single future to complete
        # (highly unlikely, most likely something is wrong)
        result = future.result(timeout=60*60)
        if result is not None:
            for w in result.keys():
                waveforms[w].extend(result[w])

    for w in waveform_types:
        output_embeddings = []
        if len(completed_pts) > 0:
            # Save copy as backup
            print(f"Appending onto existing file...")
            backup_time = str(int(time.time()))
            copyfile(f"{output_folder}/{w}/embeddings.dat.npy", f"{output_folder}/{w}/embeddings.{backup_time}.dat.npy")
            copyfile(f"{output_folder}/{w}/summary.csv", f"{output_folder}/{w}/summary.{backup_time}.csv")

            with open(f"{output_folder}/{w}/summary.csv", "a") as f:
                writer = csv.writer(f, delimiter=',', quotechar='"')
                for row in waveforms[w]:
                    writer.writerow([row["record_name"], row["pointer"], row["quality"], row["lead_time"]])
                    output_embeddings.append(row["embeddings"])
            
            existing_tensor = np.load(f"{output_folder}/{w}/embeddings.dat.npy")

            output_tensor = np.array(output_embeddings)
            existing_tensor = np.concatenate((existing_tensor, output_tensor), axis=0)
            np.save(f"{output_folder}/{w}/embeddings.dat", existing_tensor)
        else:
            with open(f"{output_folder}/{w}/summary.csv", "w") as f:
                writer = csv.writer(f, delimiter=',', quotechar='"')
                headers = ["patient_id", "pointer", "quality", "lead_time"]
                writer.writerow(headers)
                for row in waveforms[w]:
                    writer.writerow([row["record_name"], row["pointer"], row["quality"], row["lead_time"]])
                    output_embeddings.append(row["embeddings"])

            output_tensor = np.array(output_embeddings)
            np.save(f"{output_folder}/{w}/embeddings.dat", output_tensor)
    print(f"Output is written to: {output_folder}/{w}/summary.csv")
    print(f"Output is written to: {output_folder}/{w}/embeddings.dat.npy")
    print(f"END TIME: {datetime.datetime.now()}")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts waveforms from the patient dir to create a single file')
    parser.add_argument('-i', '--waveform-file',
                        required=True,
                        help='The path to the waveforms file')
    parser.add_argument('-d', '--patient-dir',
                        required=True,
                        help='The path to the patient dir')
    parser.add_argument('-o', '--output-folder',
                        required=True,
                        help='Folder where the output files will be written')
    parser.add_argument('-l', '--length',
                        default=15,
                        help='Length of the output waveform (sec)')
    parser.add_argument('-n', '--normalize', action='store_true', help='Normalize the waveform')
    parser.add_argument('-f', '--frequency',
                        default=500,
                        help='Length of the output frequency (Hz)')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')
    parser.add_argument('-m', '--max-lead-time',
                        default=None,
                        help='We pick all waveforms up to the max lead time in seconds (recommended since some waveforms may go one for several hours)')
    parser.add_argument('-w', '--waveform-types',
                        required=True,
                        help='Comma separated list of waveform types to process. Supported values: II, PLETH, RESP')
    parser.add_argument('-b', '--sample-before',
                        default=None,
                        help='Provide a field name of a column that contains the maximum offset from the waveform start time that waveforms will be sampled from. If none, samples from any part of the waveform in the recommended range')
    parser.add_argument('-mo', '--model-path',
                        required=True,
                        help='Pre-trained model location')
    parser.add_argument('-de', '--deepfeat-sz',
                        default=64,
                        help='deepfeat_sz')
    parser.add_argument('-re', '--remove-last-layer',
                        default=1,
                        help='Set to be 1 if we wish to remove last layer and create embeddings. 0 if we want the ECG classifications.')

    args = parser.parse_args()

    run(args)

    print("DONE")
