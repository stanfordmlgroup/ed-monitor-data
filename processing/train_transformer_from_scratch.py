#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from edm.jobs.transformer_job import TransformerJob

tj = TransformerJob(
    df_train_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.train.txt",
    df_val_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.val.txt",
    df_test_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.test.txt",
    summary_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/summary.csv",
    embeddings_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/waveforms.dat.npy",
    save_predictions_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer-64-scratch",
    run_bootstrap_ci=False,
    save_model=True
)

auroc_train, auroc_val, auroc_test = tj.run(epochs=100, patience=10)
print(auroc_train, auroc_val, auroc_test)
