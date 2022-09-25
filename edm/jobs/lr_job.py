import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from edm.utils.measures import perf_measure, calculate_output_statistics, calculate_confidence_intervals
from edm.utils.embeddings import get_embedding_df
from pathlib import Path
import csv

class LogisticRegressionJob():
    
    def __init__(self, df_train_path, df_val_path, df_test_path, summary_path, embeddings_path, save_predictions_path=None):
        """
        :param df_train_path: Example - /deep/group/ed-monitor/patient_data_v9/consolidated.filtered.train.txt
        :param df_val_path: Example - /deep/group/ed-monitor/patient_data_v9/consolidated.filtered.val.txt
        :param df_test_path: Example - /deep/group/ed-monitor/patient_data_v9/consolidated.filtered.test.txt
        :param summary_path: Example - /deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer/embeddings_summary.csv
        :param embeddings_path: Example - /deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer/embeddings.dat.npy
        """
        self.df_train_path = df_train_path
        self.df_val_path = df_val_path
        self.df_test_path = df_test_path
        self.summary_path = summary_path
        self.embeddings_path = embeddings_path
        self.save_predictions_path = save_predictions_path

    def run(self, max_iter=1000, class_weight="balanced", run_bootstrap_ci=True, random_state=42):

        df_train = pd.read_csv(self.df_train_path, sep="\t", na_values='?')
        df_val = pd.read_csv(self.df_val_path, sep="\t", na_values='?')
        df_test = pd.read_csv(self.df_test_path, sep="\t", na_values='?')
        summary_df = pd.read_csv(self.summary_path)
        waveforms = np.load(self.embeddings_path)

        print(f"Read df_train with shape = {df_train.shape}, pos = {df_train[df_train['outcome'] == 1].shape}, neg = {df_train[df_train['outcome'] == 0].shape}")
        print(f"Read df_val with shape = {df_val.shape}, pos = {df_val[df_val['outcome'] == 1].shape}, neg = {df_val[df_val['outcome'] == 0].shape}")
        print(f"Read df_test with shape = {df_test.shape}, pos = {df_test[df_test['outcome'] == 1].shape}, neg = {df_test[df_test['outcome'] == 0].shape}")

        df_train_x = get_embedding_df(df_train, summary_df, waveforms)
        print(f"Produced embedding for df_train_x with shape = {df_train_x.shape}")

        df_val_x = get_embedding_df(df_val, summary_df, waveforms)
        print(f"Produced embedding for df_val_x with shape = {df_val_x.shape}")

        df_test_x = get_embedding_df(df_test, summary_df, waveforms)
        print(f"Produced embedding for df_test_x with shape = {df_test_x.shape}")

        df_train_y = df_train_x["outcome"]
        df_train_x = df_train_x.drop(["patient_id", "outcome"], axis=1)

        df_val_y = df_val_x["outcome"]
        df_val_x = df_val_x.drop(["patient_id", "outcome"], axis=1)

        df_test_y = df_test_x["outcome"]
        df_test_x_ids = df_test_x["patient_id"].tolist()
        df_test_x = df_test_x.drop(["patient_id", "outcome"], axis=1)

        print(f"Starting model training...")
        clf = LogisticRegression(random_state=random_state, class_weight=class_weight, max_iter=max_iter)
        clf.fit(df_train_x, df_train_y)

        print()
        print()
        print(f"=== TRAIN ===")
        y_train_pred = clf.predict_proba(df_train_x)[:, 1]
        auroc_train = calculate_output_statistics(df_train_y.tolist(), y_train_pred)
        calculate_confidence_intervals(df_train_y.tolist(), y_train_pred, ci_type="delong")

        print()
        print()
        print(f"=== VAL ===")
        y_val_pred = clf.predict_proba(df_val_x)[:, 1]
        auroc_val = calculate_output_statistics(df_val_y.tolist(), y_val_pred)
        calculate_confidence_intervals(df_val_y.tolist(), y_val_pred, ci_type="delong")

        print()
        print()
        print(f"=== TEST ===")
        y_test_pred = clf.predict_proba(df_test_x)[:, 1]
        auroc_test = calculate_output_statistics(df_test_y.tolist(), y_test_pred)
        calculate_confidence_intervals(df_test_y.tolist(), y_test_pred, ci_type="delong")
        if run_bootstrap_ci:
            calculate_confidence_intervals(df_test_y.tolist(), y_test_pred, ci_type="bootstrap")

        if self.save_predictions_path is not None:
            Path(f"{self.save_predictions_path}").mkdir(parents=True, exist_ok=True)
            with open(f"{self.save_predictions_path}/test.csv", "w") as fp:
                writer = csv.writer(fp, delimiter=",")
                writer.writerow(["patient_id", "preds", "actual"])
                final_y = df_test_y.tolist()
                for ind in range(len(df_test_x_ids)):
                    writer.writerow([df_test_x_ids[ind], y_test_pred[ind], final_y[ind]])

        return auroc_train, auroc_val, auroc_test
