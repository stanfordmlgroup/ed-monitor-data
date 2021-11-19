#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import roc_curve, auc
from edm.jobs.transformer_job import TransformerJob

tj = TransformerJob(
    df_train_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.12lead.train.txt",
    df_val_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.12lead.val.txt",
    df_test_path="/deep/group/ed-monitor/patient_data_v9/consolidated.filtered.12lead.test.txt",
    summary_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/summary.csv",
    embeddings_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/waveforms.dat.npy",
    save_predictions_path="/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer-64-scratch-test",
    run_bootstrap_ci=True,
    verbose=0x
)

final_patient_ids, final_preds, final_y = tj.test("/deep/u/tomjin/ed-monitor-data/submissions/checkpoints/epoch=3-step=227.ckpt")

a = {
    "final_patient_ids": final_patient_ids,
    "final_preds": final_preds,
    "final_y": final_y
}

with open('/deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer-64-scratch/test_output.pkl', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
