#!/usr/bin/env python

"""
Applies the pretrained Transformer model onto the 12-lead ECG file to retrieve embeddings. 

Example: python apply_pretrained_transformer_12leads.py -i /deep/group/ed-monitor/admission-ecgs/v1/processed.dat.npy -s /deep/group/ed-monitor/admission-ecgs/v1/processed.csv -o /deep/group/ed-monitor/patient_data_v9/waveforms/12-lead -m /deep/u/tomjin/aihc-aut20-selfecg/prna/outputs-wide-64-12lead/saved_models/ctn/fold_1/ctn.tar -c /deep/group/ed-monitor/patient_data_v9/consolidated.filtered.csv -p 5
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from edm.models.transformer_model import load_best_model
from biosppy.signals.tools import filter_signal

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def apply_filter(signal, filter_bandwidth, fs=500):
    # Calculate filter order
    order = int(0.3 * fs)
    # Filter signal
    signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                 order=order, frequency=filter_bandwidth,
                                 sampling_rate=fs)
    return signal


def normalize(seq, smooth=1e-8):
    ''' Normalize each sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq) + smooth) - 1


def process_record(input_args):
    i, total_rows, csn, waveform, remove_last_layer, model, lead = input_args
    print(f"[{i}/{total_rows}] {csn} waveform...")

    try:
        if lead is None:
            data = waveform[:, :5000]
            assert data.shape[0] == 12
        elif lead == "II":
            data = waveform[1:2, :5000]
            assert data.shape[0] == 1
        else:
            raise NotImplementedError()
        
        assert data.shape[1] == 5000
        data = apply_filter(data, [3, 45])
        data = normalize(data)

        input_tensor = torch.tensor(np.array([data]), dtype=torch.float32).to(device)
        if remove_last_layer:
            embeddings = model(input_tensor).detach().cpu().numpy()
        else:
            embeddings = model(input_tensor, None).detach().cpu().numpy()
        embeddings = np.mean(embeddings, axis=0)
        del input_tensor

        return {
            "record_name": csn,
            "embeddings": embeddings
        }
    except Exception as e:
        print("Unexpected error:", e)
        return None


def run(args):
    waveform_file = args.waveform_file
    summary_file = args.summary_file
    consolidated_file = args.consolidated_file
    output_folder = args.output_folder
    model_path = args.model_path
    remove_last_layer = bool(int(args.remove_last_layer))
    deepfeat_sz = int(args.deepfeat_sz)
    max_patients = int(args.max_patients) if args.max_patients is not None else None
    lead = args.lead

    if remove_last_layer:
        if lead is None:
            output_folder = f"{output_folder}/transformer-{deepfeat_sz}"
        else:
            output_folder = f"{output_folder}/transformer-{deepfeat_sz}-{lead}"
    else:
        if lead is None:
            output_folder = f"{output_folder}/transformer-{deepfeat_sz}-logits"
        else:
            output_folder = f"{output_folder}/transformer-{deepfeat_sz}-{lead}-logits"

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Loading file from {waveform_file}")
    waveforms = np.load(waveform_file, allow_pickle=True)

    print(f"Done loading input file with shape {waveforms.shape}")
    print(f"Output folder will be: {output_folder}")

    df_summary = pd.read_csv(f"{summary_file}")
    df_consolidated = pd.read_csv(f"{consolidated_file}")
    patient_ids = set(df_consolidated["patient_id"].tolist())

    print(f"Loading model from {model_path}")
    
    if lead is None:
        model = load_best_model(model_path, deepfeat_sz=deepfeat_sz, remove_last_layer=remove_last_layer, leads=12)
    else:
        model = load_best_model(model_path, deepfeat_sz=deepfeat_sz, remove_last_layer=remove_last_layer)
    model.eval()
    print(f"Loaded model from {model_path}")

    total_rows = len(df_consolidated)

    csn_to_index = {}
    df_summary = df_summary.sort_values(by=['date', 'time'])
    for i, row in df_summary.iterrows():
        try:
            csn = int(row["csn"])

            # TODO: Each CSN could have multiple ECGs taken. Here, we take the first occurrence but we could 
            # expand this to use the average of all ECGs.
            if csn not in csn_to_index:
                csn_to_index[csn] = i
        except:
            pass

    print(f"Found csn_to_index = {len(csn_to_index)}")

    output = []
    for i, patient_id in tqdm(enumerate(patient_ids), disable=True):
        if patient_id in csn_to_index:
            input_args = [i, total_rows, patient_id, waveforms[csn_to_index[patient_id]], remove_last_layer, model, lead]
            result = process_record(input_args)
            if result is not None:
                output.append(result)
            if max_patients is not None and i >= (max_patients - 1):
                break

    output_embeddings = []
    with open(f"{output_folder}/embeddings_summary.csv", "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        headers = ["record_name"]
        writer.writerow(headers)
        for row in output:
            new_row = [row["record_name"]]
            writer.writerow(new_row)
            output_embeddings.append(row["embeddings"])

    output_tensor = np.array(output_embeddings)
    np.save(f"{output_folder}/embeddings.dat", output_tensor)
    print(f"Output is written to: {output_folder}/embeddings_summary.csv")
    print(f"Output is written to: {output_folder}/embeddings.dat.npy")


#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates embeddings by applying the pre-trained model onto the waveforms')
    parser.add_argument('-i', '--waveform-file',
                        required=True,
                        help='The path to the 12-lead waveforms containing')
    parser.add_argument('-s', '--summary-file',
                        required=True,
                        help='The path to the waveforms summary file')
    parser.add_argument('-c', '--consolidated-file',
                        required=True,
                        help='The path to the consolidated file containing the patients of interest')
    parser.add_argument('-o', '--output-folder',
                        required=True,
                        help='The output folder')
    parser.add_argument('-m', '--model-path',
                        default="",
                        help='Pre-trained model location')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')
    parser.add_argument('-d', '--deepfeat-sz',
                        default=64,
                        help='deepfeat_sz')
    parser.add_argument('-r', '--remove-last-layer',
                        default=1,
                        help='Set to be 1 if we wish to remove last layer and create embeddings. 0 if we want the ECG classifications.')
    parser.add_argument('-l', '--lead',
                        default=None,
                        help='Set this value if you want to use a single-lead from the 12-lead ECG.')

    args = parser.parse_args()

    run(args)

    print("DONE")
