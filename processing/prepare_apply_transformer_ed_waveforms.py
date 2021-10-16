#!/usr/bin/env python

"""
Script to prepare the ED-acquired Philips waveforms, but also applies the Transformer model to create embeddings
that are written to a single file.

This is useful to produce the embeddings for consecutive waveforms, since
the number of waveforms differ in each patient.

Example: python prepare_apply_transformer_ed_waveforms.py -i /deep/group/pulmonary-embolism/v2/consolidated.filtered.test.csv -d /deep/group/pulmonary-embolism/v2/patient-data -o /deep/group/pulmonary-embolism/v2/waveforms -l 15 -f 500 -w II -p 1 -n -mo /deep/u/tomjin/aihc-aut20-selfecg/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar
"""

import argparse
import csv
import datetime
import pickle
from concurrent import futures
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from edm.models.transformer_model import load_best_model
from edm.utils.waveforms import WAVEFORMS_OF_INTERST, get_waveform

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

PATIENCE = 10 # number of rounds before we give up trying to find a non-empty waveform


def load_pkl_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def process_record(input_args):
    i, total_rows, waveform_df, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, sample_before, model = input_args

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
                while pointer < (end_offset - window_size):

                    waveform, quality = get_waveform(waveform_base, pointer, window_size, fs, should_normalize=should_normalize, bandpass_type=waveform_config["bandpass_type"], bandwidth=waveform_config["bandpass_freq"], target_fs=waveform_target_freq)

                    print(f"[{i}/{total_rows}] {patient_id} waveform {waveform_type} of shape {waveform.shape} at {pointer}")

                    input_tensor = torch.tensor(np.array([waveform]), dtype=torch.float32).to(device)
                    embeddings = model(input_tensor).detach().cpu().numpy()
                    del input_tensor

                    if waveform_type not in waveforms:
                        waveforms[waveform_type] = []

                    waveforms[waveform_type].append({
                        "record_name": patient_id,
                        "pointer": pointer,
                        "embeddings": embeddings,
                        "quality": quality,
                        "lead_time": lead_time
                    })

                    pointer += window_size
                    lead_time += waveform_length
                    num_done += 1

                print(f"[{i}/{total_rows}] {patient_id} had {num_done} waveforms")

        return waveforms
    except Exception as e:
        print("Unexpected error:", e)
        return {}
    

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

    print(f"Loading model from {model_path}")
    model = load_best_model(model_path, deepfeat_sz=deepfeat_sz, remove_last_layer=remove_last_layer)
    model.eval()
    print(f"Loaded model from {model_path}")

    output_folder = f"{output_folder}/{waveform_length}sec-{int(waveform_target_freq)}hz-{int(should_normalize)}norm-all/transformer-{deepfeat_sz}"
    if not remove_last_layer:
        output_folder = f"{output_folder}-{deepfeat_sz}"

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for w in waveform_types:
        Path(f"{output_folder}/{w}").mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_file)
    print(f"Found {input_file} with shape {df.shape}")
    total_rows = len(df)

    fs = []
    with futures.ProcessPoolExecutor(16) as executor:
        for i, row in tqdm(df.iterrows(), disable=True):
            input_args = [i, total_rows, df, patient_dir, should_normalize, waveform_length, waveform_target_freq, waveform_types, sample_before, model]
            future = executor.submit(process_record, input_args)
            fs.append(future)
            if max_patients is not None and i >= (max_patients - 1):
                break

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
        with open(f"{output_folder}/{w}/summary.csv", "w") as f:
            writer = csv.writer(f, delimiter=',', quotechar='"')
            headers = ["patient_id", "pointer", "quality", "lead_time"]
            writer.writerow(headers)
            for row in waveforms[w]:
                writer.writerow([row["record_name"], row["pointer"], row["quality"], row["lead_time"]])
                output_embeddings.append(row["waveform"])

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
