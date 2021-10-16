#!/usr/bin/env python

"""
Applies the pretrained PCLR model onto the consolidated waveform file to retrieve embeddings. 

Example: python apply_pretrained_pclr.py -i /deep/group/pulmonary-embolism/v2/waveforms/15sec-500hz-1norm-3wpp/II -m /deep/group/pulmonary-embolism/v2/pretrained-models/pclr/PCLR_lead_II.h5
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

pd.set_option('display.max_columns', 500)

def process_record(input_args):
    i, total_rows, waveform_df, patient_id, waveforms, model = input_args
    print(f"[{i}/{total_rows}] {patient_id} waveform...")
    
    try:
        waveforms_for_patient = []
        for j, row in waveform_df[waveform_df["record_name"] == patient_id].iterrows():
            waveforms_for_patient.append(np.expand_dims(waveforms[j], axis=1))

        embeddings = model(tf.convert_to_tensor(np.array(waveforms_for_patient), dtype=tf.float32)).numpy()
        embeddings = np.mean(embeddings, axis=0)

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
    max_patients = int(args.max_patients) if args.max_patients is not None else None
    
    output_folder = f"{input_folder}/pclr"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Loading file from {input_folder}")

    df = pd.read_csv(f"{input_folder}/summary.csv")
    waveforms_numpy = np.load(f"{input_folder}/waveforms.dat.npy")
    patient_ids = set(df["record_name"].tolist())
    
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    print(f"Found {input_folder} with shape {df.shape}")
    total_rows = len(df)

    waveforms = []
    for i, patient_id in tqdm(enumerate(patient_ids), disable=True):
        input_args = [i, total_rows, df, patient_id, waveforms_numpy, model]
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
    print(f"Output is written to: {output_folder}/summary.csv")
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
                        default=15,
                        help='Pre-trained model location')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')

    args = parser.parse_args()

    run(args)

    print("DONE")
