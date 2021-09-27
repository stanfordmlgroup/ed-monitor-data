#!/usr/bin/env python

"""
Applies the pretrained Transformer model onto the consolidated waveform file to retrieve embeddings. 

Example: python apply_pretrained_transformer.py -i /deep/group/lactate/v1/waveforms/15sec-500hz-1norm-1wpp/II -m /deep/u/tomjin/aihc-aut20-selfecg/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar
"""

import datetime
import pandas as pd
import numpy as np
import os
import sys
import pytz
import re
import csv
import matplotlib.pyplot as plt
import math
import pickle
from biosppy.signals.tools import filter_signal
from concurrent import futures
import argparse
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import wfdb
import torch
from scipy import signal
from scipy.signal import decimate, resample
from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

    
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning.metrics.functional as metrics
import pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler
import csv  
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, normalize, Normalizer
from sklearn.compose import ColumnTransformer
from pathlib import Path
import datetime
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# Transformer parameters
# Copied from the PRNA model
#
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers
dropout_rate = 0.2
model_name = 'ctn'
nb_demo = 2
nb_feats = 20
classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006',
                  '713426002', '445118002', '39732003', '164909002', '251146004',
                  '698252002', '10370003', '284470004', '427172004', '164947007',
                  '111975006', '164917005', '47665007', '427393009',
                  '426177001', '426783006', '427084000', '164934002',
                  '59931005'])


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encodings to embeddings
        # x: embedding vects, [B x L x d_model]
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    '''
    Transformer encoder processes convolved ECG samples
    Stacks a number of TransformerEncoderLayers
    '''

    def __init__(self, d_model, h, d_ff, num_layers, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe = PositionalEncoding(d_model, dropout=0.1)

        encode_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.h,
            dim_feedforward=self.d_ff,
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.pe(out)
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = out.mean(0)  # global pooling
        return out


# 15 second model
class CTN(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_feats, nb_demo, classes):
        super(CTN, self).__init__()

        self.encoder = nn.Sequential(  # downsampling factor = 20
            nn.Conv1d(1, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        self.transformer = Transformer(d_model, nhead, d_ff, num_layers, dropout=0.1)
        self.fc1 = nn.Linear(d_model, deepfeat_sz)
        # self.fc2 = nn.Linear(deepfeat_sz+nb_feats+nb_demo, len(classes))
        self.fc2 = nn.Linear(deepfeat_sz, len(classes))
        self.dropout = nn.Dropout(dropout_rate)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.apply(_weights_init)

    def forward(self, x, wide_feats):
        z = self.encoder(x)  # encoded sequence is batch_sz x nb_ch x seq_len
        out = self.transformer(z)  # transformer output is batch_sz x d_model
        out = self.dropout(F.relu(self.fc1(out)))
        # out = self.fc2(torch.cat([wide_feats, out], dim=1))
        out = self.fc2(out)
        return out


def load_best_model(model_loc, deepfeat_sz, remove_last_layer=True):
    model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_feats, nb_demo, classes).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if model_loc is not None:
        if torch.cuda.is_available():
            checkpoint = torch.load(model_loc)
        else:
            checkpoint = torch.load(model_loc, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loading best model: best_loss', checkpoint['best_loss'], 'best_auroc', checkpoint['best_auroc'],
              'at epoch',
              checkpoint['epoch'])
    else:
        print("NOT using a pre-trained model")

    if remove_last_layer:
        model = torch.nn.Sequential(*(list(list(model.children())[0].children())[:-2]))
    return model


def process_record(input_args):
    i, total_rows, waveform_df, patient_id, waveforms, model = input_args
    print(f"[{i}/{total_rows}] {patient_id} waveform...")
    
    try:
        waveforms_for_patient = []
        for j, row in waveform_df[waveform_df["record_name"] == patient_id].iterrows():
            waveforms_for_patient.append([waveforms[j]])

        input_tensor = torch.tensor(np.array(waveforms_for_patient), dtype=torch.float32).to(device)
#         print(f"input={input_tensor.shape}")
        embeddings = model(input_tensor).detach().cpu().numpy()
#         print(f"output1={embeddings.shape}")
        embeddings = np.mean(embeddings, axis=0)
#         print(f"output2={embeddings.shape}")

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
    deepfeat_sz = int(args.deepfeat_sz)
    max_patients = int(args.max_patients) if args.max_patients is not None else None
    
    output_folder = f"{input_folder}/transformer"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Loading file from {input_folder}")

    df = pd.read_csv(f"{input_folder}/summary.csv")
    waveforms_numpy = np.load(f"{input_folder}/waveforms.dat.npy")
    patient_ids = set(df["record_name"].tolist())
    
    print(f"Loading model from {model_path}")
    
    model = load_best_model(model_path, deepfeat_sz=deepfeat_sz, remove_last_layer=True)
    model.eval()
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
                        default=15,
                        help='Pre-trained model location')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')
    parser.add_argument('-d', '--deepfeat-sz',
                        default=64,
                        help='deepfeat_sz')

    args = parser.parse_args()

    run(args)

    print("DONE")
