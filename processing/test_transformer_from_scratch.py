#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc
from edm.jobs.transformer_job import TransformerJob

def run(args):
    include_numerics = args.include_numerics
    model = args.model # e.g. "/deep/u/tomjin/ed-monitor-data/submissions/checkpoints/epoch=8-step=575.ckpt"
    output = args.output # e.g. '/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer-64-scratch/test_output.pkl'
    print(f"RUNNING TEST...")

    if not include_numerics:
        print("NOT INCLUDING NUMERICS")

        tj = TransformerJob(
            df_train_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.train.txt",
            df_val_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.val.txt",
            df_test_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.test.txt",
            summary_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/summary.csv",
            embeddings_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/waveforms.dat.npy",
            save_predictions_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer-64-scratch-test",
            run_bootstrap_ci=True,
            verbose=0
        )

    else:
        print("INCLUDING NUMERICS")

        tj = TransformerJob(
            df_train_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.train.txt",
            df_val_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.val.txt",
            df_test_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.test.txt",
            summary_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/summary.csv",
            embeddings_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/waveforms.dat.npy",
            save_predictions_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer-64-scratch-test",
            additional_cols=["Age", "Gender", "SpO2", "RR", "HR", "Temp", "SBP", "DBP", "Athero", "HTN", "HLD", "DM", "Obese", "Smoking"],
            ordinal_cols=["Gender"],
            run_bootstrap_ci=True,
            verbose=0
        )

    final_patient_ids, final_preds, final_y = tj.test(model)

    a = {
        "final_patient_ids": final_patient_ids,
        "final_preds": final_preds,
        "final_y": final_y
    }

    with open(output, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests a Transformer-based model from scratch')
    parser.add_argument('-n', '--include_numerics', action='store_true')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    run(args)

    print("DONE")

