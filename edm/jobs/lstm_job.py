import os
import sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from edm.utils.measures import perf_measure, calculate_output_statistics
from edm.utils.embeddings import get_embedding_df, clean_additional_columns
from edm.modules.lstm_module import train_lstm


class LstmJob():
    
    def __init__(self, df_train_path, df_val_path, df_test_path, monitoring_data_path, outcome_col,
                 additional_cols=[], monitoring_cols=[], ordinal_cols=[], impute=True, normalize=True, 
                 save_model=False, save_predictions_path=None, run_bootstrap_ci=True, verbose=0):
        """
        :param df_train_path: Example - /deep/group/lactate/v3/monitoring-collectiontime1.train.txt
        :param df_val_path: Example - /deep/group/lactate/v3/monitoring-collectiontime1.val.txt
        :param df_test_path: Example - /deep/group/lactate/v3/monitoring-collectiontime1.test.txt
        :param monitoring_data_path: Example - /deep/group/lactate/v3/monitoring-collectiontime1.head.pkl
        :param additional_cols: Additional columns to add as part of the training process. For instance, these could be vital signs.
        :param ordinal_cols: Ordinal columns.
        :param monitoring_cols: Any additional columns which are for monitoring.
        :param impute: True if we should impute the additional columns when there are nan.
        :param normalize: True if we should scale the additional columns.
        :param save_model: True if you want to save the actual trained model (for possible use later on).
        :param save_predictions_path: Save the predictions to this path.
        :param verbose: Verbose level 0 means complete silence - no logs at all. 1 means some basic messages will be logged, incl proress bar. 2 means everything will be logged.
        """
        self.df_train_path = df_train_path
        self.df_val_path = df_val_path
        self.df_test_path = df_test_path
        self.outcome_col = outcome_col
        self.monitoring_data_path = monitoring_data_path
        
        self.monitoring_cols = monitoring_cols
        self.additional_cols = additional_cols
        self.ordinal_cols = ordinal_cols
        self.impute = impute
        self.normalize = normalize
        self.save_model = save_model
        self.save_predictions_path = save_predictions_path
        self.run_bootstrap_ci = run_bootstrap_ci
        self.verbose = verbose

    # features=['HR', 'RR', 'btbRRInt_ms', 'MAP', 'SpO2']
    def get_input_array(self, df, b, max_datapoints=6*60, features=['HR', 'RR', 'btbRRInt_ms', 'MAP', 'SpO2']):
        # Returns n x seq_len x features
        output_matrix = np.zeros((len(df), max_datapoints, len(features)))
        seq_lens = []
        for i, row in tqdm(df.iterrows()):
            seq_lens.append(min(max_datapoints, len(b[row["CSN"]]["HR"]["left_vals"])))
            hr_times = b[row["CSN"]]["HR"]["left_times"]
            if len(hr_times) > max_datapoints:
                hr_times = hr_times[-max_datapoints:]
            for j, feature in enumerate(features):
                if feature == "HR":
                    vals = b[row["CSN"]][feature]["left_vals"]
                    if len(vals) > max_datapoints:
                        vals = vals[-max_datapoints:]
                else:
                    feat_times = b[row["CSN"]][feature]["left_times"]
                    feat_vals = b[row["CSN"]][feature]["left_vals"]

                    #      -----------
                    #    x   x    xxx
                    if len(feat_vals) == 0:
                        # This feature is not recorded so there's nothing to see here
                        continue

                    k = 0 # pointer for heart rate times
                    t = 0 # pointer for current feature times
                    v = 0 # pointer for target array
                    vals = np.zeros((max_datapoints))
                    prev_val = feat_vals[0]
                    while k < len(hr_times):
                        if hr_times[k] >= feat_times[t] and t < (len(feat_times) - 1):
                            prev_val = feat_vals[t]
                            t += 1
                        vals[v] = prev_val
                        k += 1
                        v += 1

                output_matrix[i, :len(vals), j] = vals

        return output_matrix, seq_lens
    
    def get_metadata_array(self, df, max_datapoints=6*60, additional_cols=["Age", "SpO2", "RR", "HR", "Temp", "SBP", "DBP", "Athero", "HTN", "HLD", "DM", "Obese", "Smoking"], ordinal_cols=["Gender"]):
        col_features = [f for f in additional_cols]
        col_features.extend(ordinal_cols)
        output_matrix = np.zeros((len(df), max_datapoints, len(col_features)))
        for i, row in tqdm(df.iterrows()):
            for j, feature in enumerate(col_features):
                val = row[feature]
                output_matrix[i, :, j] = np.full((max_datapoints), val)

        return output_matrix
        
    def __preprocess(self):
        df_train = pd.read_csv(self.df_train_path, sep="\t", na_values='?')
        df_val = pd.read_csv(self.df_val_path, sep="\t", na_values='?')
        df_test = pd.read_csv(self.df_test_path, sep="\t", na_values='?')
        
        if len(self.additional_cols) > 0 or len(self.ordinal_cols) > 0:
            df_train, df_val, df_test = clean_additional_columns(df_train, df_val, df_test, cols_to_clean=self.additional_cols, ordinal_cols=self.ordinal_cols, impute=True, normalize=True)

        with open(self.monitoring_data_path, 'rb') as handle:
            b = pickle.load(handle)

        if self.verbose >= 1:
            print(f"Read df_train with shape = {df_train.shape}, pos = {df_train[df_train[self.outcome_col] == 1].shape}, neg = {df_train[df_train[self.outcome_col] == 0].shape}")
            print(f"Read df_val with shape = {df_val.shape}, pos = {df_val[df_val[self.outcome_col] == 1].shape}, neg = {df_val[df_val[self.outcome_col] == 0].shape}")
            print(f"Read df_test with shape = {df_test.shape}, pos = {df_test[df_test[self.outcome_col] == 1].shape}, neg = {df_test[df_test[self.outcome_col] == 0].shape}")

        train_mat, train_seq_lens = self.get_input_array(df_train, b, features=self.monitoring_cols)
        if len(self.additional_cols) > 0 or len(self.ordinal_cols) > 0:
            train_metadata_mat = self.get_metadata_array(df_train, additional_cols=self.additional_cols, ordinal_cols=self.ordinal_cols)
        if self.verbose >= 1:
            print(f"Produced embedding for train_mat with shape = {train_mat.shape}")

        val_mat, val_seq_lens = self.get_input_array(df_val, b, features=self.monitoring_cols)
        if len(self.additional_cols) > 0 or len(self.ordinal_cols) > 0:
            val_metadata_mat = self.get_metadata_array(df_val, additional_cols=self.additional_cols, ordinal_cols=self.ordinal_cols)
        if self.verbose >= 1:
            print(f"Produced embedding for val_mat with shape = {val_mat.shape}")

        test_mat, test_seq_lens = self.get_input_array(df_test, b, features=self.monitoring_cols)
        if len(self.additional_cols) > 0 or len(self.ordinal_cols) > 0:
            test_metadata_mat = self.get_metadata_array(df_test, additional_cols=self.additional_cols, ordinal_cols=self.ordinal_cols)
        if self.verbose >= 1:
            print(f"Produced embedding for test_mat with shape = {test_mat.shape}")

        scaler = MinMaxScaler()
        train_mat = scaler.fit_transform(train_mat.reshape(-1, train_mat.shape[-1])).reshape(train_mat.shape)
        val_mat = scaler.transform(val_mat.reshape(-1, val_mat.shape[-1])).reshape(val_mat.shape)
        test_mat = scaler.transform(test_mat.reshape(-1, test_mat.shape[-1])).reshape(test_mat.shape)

        if len(self.additional_cols) > 0 or len(self.ordinal_cols) > 0:
            train_mat = np.concatenate((train_mat, train_metadata_mat), axis=2)
            val_mat = np.concatenate((val_mat, val_metadata_mat), axis=2)
            test_mat = np.concatenate((test_mat, test_metadata_mat), axis=2)
        
            if self.verbose >= 1:
                print(f"Augmented train_mat with metadata to produce shape = {train_mat.shape}")
                print(f"Augmented train_mat with metadata to produce shape = {val_mat.shape}")
                print(f"Augmented train_mat with metadata to produce shape = {test_mat.shape}")
                # print(train_mat[0])

        df_train_y = df_train[self.outcome_col].tolist()
        df_val_y = df_val[self.outcome_col].tolist()
        df_test_y = df_test[self.outcome_col].tolist()
        
        df_train_ids = df_train["CSN"].tolist()
        df_val_ids = df_val["CSN"].tolist()
        df_test_ids = df_test["CSN"].tolist()
        
        return train_mat, val_mat, test_mat, df_train_ids, df_val_ids, df_test_ids, df_train_y, df_val_y, df_test_y, train_seq_lens, val_seq_lens, test_seq_lens

    def run(self, batch_size=128, learning_rate=0.00001, dropout_rate=0, num_inner_layers=2, epochs=100, inner_dim=128, patience=None):
        train_mat, val_mat, test_mat, df_train_ids, df_val_ids, df_test_ids, df_train_y, df_val_y, df_test_y, train_seq_lens, val_seq_lens, test_seq_lens = self.__preprocess()

        if self.verbose >= 1:
            print(f"Starting model training...")
        return train_lstm(train_mat, val_mat, test_mat, df_train_ids, df_val_ids, df_test_ids,
                          df_train_y, df_val_y, df_test_y, train_seq_lens, val_seq_lens, test_seq_lens,
                          batch_size=batch_size, patience=patience, inner_dim=inner_dim, learning_rate=learning_rate, dropout_rate=dropout_rate,
                          num_inner_layers=num_inner_layers, epochs=epochs, save_predictions_path=self.save_predictions_path, 
                          save_model=self.save_model, run_bootstrap_ci=self.run_bootstrap_ci, verbose=self.verbose)

#     def test(self, model_path, batch_size=128, dropout_rate=0, num_inner_layers=2, inner_dim=128):
#         # Note that the original train/val files still need to be provided during testing to ensure we properly
#         # clean the columns (e.g. normalize with respect to the train file)
#         _, _, df_test_x = self.__preprocess()

#         if self.verbose >= 1:
#             print(f"Starting model testing...")
#         return test_lstm(df_test_x, model_path, batch_size=batch_size, embed_dim=df_test_x.shape[1] - 2,
#                         inner_dim=inner_dim, dropout_rate=dropout_rate,
#                         num_inner_layers=num_inner_layers, save_predictions_path=self.save_predictions_path,
#                         verbose=self.verbose)
