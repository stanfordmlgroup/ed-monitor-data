#!/usr/bin/env python

"""
Applies the pretrained Transformer model onto the consolidated waveform file to retrieve embeddings. 

Example: python apply_pretrained_transformer.py -i /deep/group/lactate/v1/waveforms/15sec-500hz-1norm-1wpp/II -m /deep/u/tomjin/aihc-aut20-selfecg/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from edm.models.transformer_model import load_best_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def process_record(input_args):
    i, total_rows, waveform_df, patient_id, waveforms, remove_last_layer, model = input_args
    print(f"[{i}/{total_rows}] {patient_id} waveform...")
    
    try:
        waveforms_for_patient = []
        for j, row in waveform_df[waveform_df["record_name"] == patient_id].iterrows():
            waveforms_for_patient.append([waveforms[j]])

        input_tensor = torch.tensor(np.array(waveforms_for_patient), dtype=torch.float32).to(device)
        if remove_last_layer:
            embeddings = model(input_tensor).detach().cpu().numpy()
        else:
            embeddings = model(input_tensor, None).detach().cpu().numpy()
        embeddings = np.mean(embeddings, axis=0)
        del input_tensor

        return {
            "record_name": patient_id,
            "subject_id": waveform_df[waveform_df["record_name"] == patient_id]["subject_id"] if "subject_id" in waveform_df.columns else "",
            "embeddings": embeddings
        }
    except Exception as e:
        print("Unexpected error:", e)
        return None
    

def run(args):
    input_folder = args.waveform_folder
    model_path = args.model_path
    remove_last_layer = bool(int(args.remove_last_layer))
    deepfeat_sz = int(args.deepfeat_sz)
    max_patients = int(args.max_patients) if args.max_patients is not None else None
    
    if remove_last_layer:
        output_folder = f"{input_folder}/transformer-{deepfeat_sz}"
    else:
        output_folder = f"{input_folder}/transformer-{deepfeat_sz}-logits"

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Loading file from {input_folder}")
    print(f"Output folder will be: {output_folder}")

    df = pd.read_csv(f"{input_folder}/summary.csv")
    waveforms_numpy = np.load(f"{input_folder}/waveforms.dat.npy", allow_pickle=True)
    patient_ids = set(df["record_name"].tolist())
    
    print(f"Loading model from {model_path}")
    
    model = load_best_model(model_path, deepfeat_sz=deepfeat_sz, remove_last_layer=remove_last_layer)
    model.eval()
    print(f"Loaded model from {model_path}")
    
    print(f"Found {input_folder} with shape {df.shape}")
    total_rows = len(df["record_name"].unique())

    waveforms = []
    for i, patient_id in tqdm(enumerate(patient_ids), disable=True):
        input_args = [i, total_rows, df, patient_id, waveforms_numpy, remove_last_layer, model]
        result = process_record(input_args)
        if result is not None:
            waveforms.append(result)
        if max_patients is not None and i >= (max_patients - 1):
            break

    output_embeddings = []
    with open(f"{output_folder}/embeddings_summary.csv", "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        headers = []
        if "subject_id" in df.columns:
            headers.append("subject_id")
        if "record_name" in df.columns:
            headers.append("record_name")
        writer.writerow(headers)
        for row in waveforms:
            new_row = []
            if "subject_id" in df.columns:
                new_row.append(row["subject_id"])
            if "record_name" in df.columns:
                new_row.append(row["record_name"])
                
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
    parser = argparse.ArgumentParser(description='Extracts waveforms from the patient dir to create a single file')
    parser.add_argument('-i', '--waveform-folder',
                        required=True,
                        help='The path to the consolidated waveforms folder containing the summary file and the waveform NumPy')
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

    args = parser.parse_args()

    run(args)

    print("DONE")
