#!/usr/bin/env python

"""
Tunes the hyperparameters of a model

Example: python tune_hyperparameters.py -f /deep/group/ed-monitor/patient_data_v9 -e /deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-10wpp/II/transformer
"""    

import os
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from edm.jobs.lr_job import LogisticRegressionJob
from edm.jobs.mlp_job import MlpJob


def run(args):
    input_folder = args.input_folder
    embeddings_folder = args.embeddings_folder
    
    dropout_rates = [0, 0.2, 0.5]
    num_inner_layers = [2, 3, 4]
    inner_dims = [64, 128, 256, 512]

    best_combination = ""
    best_test_perf = 0
    best_val_perf = 0
    for dropout_rate in dropout_rates:
        for num_inner_layer in num_inner_layers:
            for inner_dim in inner_dims:
                print(f"Working on combination of dropout_rate={dropout_rate}, num_inner_layer={num_inner_layer}, inner_dim={inner_dim}")
                
                mlp = MlpJob(
                    df_train_path=f"{input_folder}/consolidated.filtered.train.txt",
                    df_val_path=f"{input_folder}/consolidated.filtered.val.txt",
                    df_test_path=f"{input_folder}/consolidated.filtered.test.txt",
                    summary_path=f"{embeddings_folder}/embeddings_summary.csv",
                    embeddings_path=f"{embeddings_folder}/embeddings.dat.npy",
                    verbose=0
                )

                auroc_train, auroc_val, auroc_test = mlp.run(batch_size=128, 
                                                             learning_rate=0.00001, 
                                                             dropout_rate=dropout_rate, 
                                                             num_inner_layers=num_inner_layer, 
                                                             epochs=100, 
                                                             inner_dim=inner_dim
                )
                
                print(f"{auroc_train}, {auroc_val}, {auroc_test}")

                if auroc_val > best_val_perf:
                    best_val_perf = auroc_val
                    best_test_perf = auroc_test
                    best_combination = f"{dropout_rate}, {num_inner_layer}, {inner_dim}"
    print()
    print("===")
    print(f"best_combination = {best_combination} with best_val_perf={best_val_perf} with best_test_perf={best_test_perf}")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tunes the hyperparameters')
    parser.add_argument('-f', '--input-folder',
                        required=True,
                        help='Folder containing the patient root folder')
    parser.add_argument('-e', '--embeddings-folder',
                        required=True,
                        help='Folder containing the embeddings')

    args = parser.parse_args()

    run(args)

    print("DONE")
