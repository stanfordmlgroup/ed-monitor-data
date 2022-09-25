import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from edm.utils.measures import perf_measure, calculate_output_statistics
from edm.utils.embeddings import get_embedding_df, clean_additional_columns
from edm.modules.transformer_module import train_transformer, test_transformer


class TransformerJob():
    
    def __init__(self, df_train_path, df_val_path, df_test_path, summary_path, embeddings_path, 
                 additional_cols=[], ordinal_cols=[], impute=True, normalize=True, 
                 save_model=False, save_predictions_path=None, run_bootstrap_ci=True, verbose=0):
        """
        :param df_train_path: Example - /deep/group/ed-monitor/patient_data_v9/consolidated.filtered.train.txt
        :param df_val_path: Example - /deep/group/ed-monitor/patient_data_v9/consolidated.filtered.val.txt
        :param df_test_path: Example - /deep/group/ed-monitor/patient_data_v9/consolidated.filtered.test.txt
        :param summary_path: Example - /deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer/embeddings_summary.csv
        :param embeddings_path: Example - /deep/group/ed-monitor/patient_data_v9/waveforms/15sec-500hz-1norm-1wpp/II/transformer/embeddings.dat.npy
        :param additional_cols: Additional columns to add as part of the training process. For instance, these could be vital signs.
        :param ordinal_cols: Any additional columns which are string type.
        :param impute: True if we should impute the additional columns when there are nan.
        :param normalize: True if we should scale the additional columns.
        :param save_model: True if you want to save the actual trained model (for possible use later on).
        :param save_predictions_path: Save the predictions to this path.
        :param verbose: Verbose level 0 means complete silence - no logs at all. 1 means some basic messages will be logged, incl proress bar. 2 means everything will be logged.
        """
        self.df_train_path = df_train_path
        self.df_val_path = df_val_path
        self.df_test_path = df_test_path
        self.summary_path = summary_path
        self.embeddings_path = embeddings_path
        
        self.additional_cols = additional_cols
        self.ordinal_cols = ordinal_cols
        self.impute = impute
        self.normalize = normalize
        self.save_model = save_model
        self.save_predictions_path = save_predictions_path
        self.run_bootstrap_ci = run_bootstrap_ci
        self.verbose = verbose

    def __preprocess(self):
        df_train = pd.read_csv(self.df_train_path, sep="\t", na_values='?')
        df_val = pd.read_csv(self.df_val_path, sep="\t", na_values='?')
        df_test = pd.read_csv(self.df_test_path, sep="\t", na_values='?')
        summary_df = pd.read_csv(self.summary_path)
        waveforms = np.load(self.embeddings_path)

        if self.verbose >= 1:
            print(f"Read df_train with shape = {df_train.shape}, pos = {df_train[df_train['outcome'] == 1].shape}, neg = {df_train[df_train['outcome'] == 0].shape}")
            print(f"Read df_val with shape = {df_val.shape}, pos = {df_val[df_val['outcome'] == 1].shape}, neg = {df_val[df_val['outcome'] == 0].shape}")
            print(f"Read df_test with shape = {df_test.shape}, pos = {df_test[df_test['outcome'] == 1].shape}, neg = {df_test[df_test['outcome'] == 0].shape}")

        df_train_x = get_embedding_df(df_train, summary_df, waveforms, additional_cols=self.additional_cols)
        if self.verbose >= 1:
            print(f"Produced embedding for df_train_x with shape = {df_train_x.shape}")

        df_val_x = get_embedding_df(df_val, summary_df, waveforms, additional_cols=self.additional_cols)
        if self.verbose >= 1:
            print(f"Produced embedding for df_val_x with shape = {df_val_x.shape}")

        df_test_x = get_embedding_df(df_test, summary_df, waveforms, additional_cols=self.additional_cols)
        if self.verbose >= 1:
            print(f"Produced embedding for df_test_x with shape = {df_test_x.shape}")

        if len(self.additional_cols) > 0:
            df_train_x, df_val_x, df_test_x = clean_additional_columns(df_train_x, df_val_x, df_test_x, cols_to_clean=self.additional_cols, ordinal_cols=self.ordinal_cols, impute=self.impute, normalize=self.normalize)
            if self.verbose >= 1:
                print(f"Cleaned columns with additional_cols={self.additional_cols} impute={self.impute} normalize={self.normalize}")

        return df_train_x, df_val_x, df_test_x

    def run(self, batch_size=128, learning_rate=0.00001, dropout_rate=0, num_inner_layers=2, epochs=100, inner_dim=128, patience=None):
        df_train_x, df_val_x, df_test_x = self.__preprocess()

        if self.verbose >= 1:
            print(f"Starting model training...")
        
        nb_patient_feats = len(self.additional_cols) + len(self.ordinal_cols)
        return train_transformer(df_train_x, df_val_x, df_test_x, batch_size=batch_size, embed_dim=df_train_x.shape[1] - 2, 
                                 nb_patient_feats=nb_patient_feats, patience=patience, inner_dim=inner_dim, 
                                 learning_rate=learning_rate, dropout_rate=dropout_rate, num_inner_layers=num_inner_layers, 
                                 epochs=epochs, save_predictions_path=self.save_predictions_path, 
                                 save_model=self.save_model, run_bootstrap_ci=self.run_bootstrap_ci, verbose=self.verbose)

    def test(self, model_path, batch_size=128, dropout_rate=0, num_inner_layers=2, inner_dim=128):
        # Note that the original train/val files still need to be provided during testing to ensure we properly
        # clean the columns (e.g. normalize with respect to the train file)
        _, _, df_test_x = self.__preprocess()

        print(f"Starting model testing...")
        nb_patient_feats = len(self.additional_cols) + len(self.ordinal_cols)
        return test_transformer(df_test_x, model_path, batch_size=batch_size, embed_dim=df_test_x.shape[1] - 2,
                        inner_dim=inner_dim, dropout_rate=dropout_rate, nb_patient_feats=nb_patient_feats,
                        num_inner_layers=num_inner_layers, save_predictions_path=self.save_predictions_path,
                        verbose=self.verbose)
