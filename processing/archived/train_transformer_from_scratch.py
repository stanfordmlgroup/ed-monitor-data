#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_curve, auc
from edm.jobs.transformer_job import TransformerJob

def run(args):
    include_numerics = args.include_numerics

    if not include_numerics:
        print("NOT INCLUDING NUMERICS")
        tj = TransformerJob(
            df_train_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.train.txt",
            df_val_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.val.txt",
            df_test_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.test.txt",
            summary_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/summary.csv",
            embeddings_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/waveforms.dat.npy",
            # save_predictions_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer-64-scratch",
            run_bootstrap_ci=False,
            save_model=False
        )
    else:
        print("INCLUDING NUMERICS")
        tj = TransformerJob(
            df_train_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.train.txt",
            df_val_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.val.txt",
            df_test_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.test.txt",
            summary_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/summary.csv",
            embeddings_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/waveforms.dat.npy",
            additional_cols=["Age", "Gender", "SpO2", "RR", "HR", "Temp", "SBP", "DBP", "Athero", "HTN", "HLD", "DM", "Obese", "Smoking"],
            ordinal_cols=["Gender"],
            run_bootstrap_ci=False,
            save_model=False
        )

    auroc_train, auroc_val, auroc_test = tj.run(epochs=100, patience=10)
    print(auroc_train, auroc_val, auroc_test)


#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains a Transformer-based model from scratch')
    parser.add_argument('-n', '--include_numerics', action='store_true')

    args = parser.parse_args()

    run(args)

    print("DONE")

