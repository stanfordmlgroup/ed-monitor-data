import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def get_cohort_df(df, outcome_col="outcome", additional_cols=[]):
    """
    Returns a dataframe that contains additional columns from the cohort file to use as input into the downstream model.
    
    :param df: The dataframe of the patients (contains visit information, demographics, etc.)
    :outcome_col: The column where the outcome is defined
    :additional_cols: The columns which should be added to the output embedding
    """

    output = []
    
    for i, row in df.iterrows():
        patient_id = int(row["patient_id"])
        outcome = row[outcome_col]
        new_row = [patient_id, outcome]

        for c in additional_cols:
            new_row.append(row[c])

        output.append(new_row)
    
    headers = ["patient_id", "outcome"]
    for c in additional_cols:
        headers.append(c)
    
    return pd.DataFrame(output, columns=headers)

def get_embedding_df(df, summary_df, waveforms, outcome_col="outcome", additional_cols=[]):
    """
    Returns a dataframe that concatenates the embeddings and any additional columns to use
    as input into the downstream model.
    
    :param df: The dataframe of the patients (contains visit information, demographics, etc.)
    :param summary_df: The dataframe that summarizes the relative positions of the patients in the waveform object
    :param waveforms: The embeddings, in the order defined by the summary_df
    :outcome_col: The column where the outcome is defined
    :additional_cols: The columns which should be added to the output embedding
    """

    patient_id_to_index = {}
    for i, row in summary_df.iterrows():
        patient_id_to_index[int(row["record_name"])] = i

    embed_size = waveforms.shape[1]
    
    output = []
    
    for i, row in tqdm(df.iterrows()):
        patient_id = int(row["patient_id"])
        outcome = row[outcome_col]
        if patient_id in patient_id_to_index:
            new_row = [patient_id, outcome]
            
            for c in additional_cols:
                new_row.append(row[c])
            
            new_row.extend(waveforms[patient_id_to_index[patient_id]])    
            output.append(new_row)
    
    headers = ["patient_id", "outcome"]
    for c in additional_cols:
        headers.append(c)

    for i in range(embed_size):
        headers.append(f"embed_{i}")
    
    return pd.DataFrame(output, columns=headers)

def clean_additional_columns(df_train, df_val, df_test, cols_to_clean=[], ordinal_cols=[], impute=True, normalize=True, save_path=None):
    """
    Imputes and/or normalizes the additional columns.
    This is required after the embedding dataframes are produced for train/val/test because the transformation
    should be based on the train set only.
    """
    cols_to_not_clean = []
    for c in df_train.columns:
        if c not in cols_to_clean:
            cols_to_not_clean.append(c)

    df_train_nosub = df_train[cols_to_not_clean]
    df_val_nosub = df_val[cols_to_not_clean]
    df_test_nosub = df_test[cols_to_not_clean]

    df_train_sub = df_train[cols_to_clean]
    df_val_sub = df_val[cols_to_clean]
    df_test_sub = df_test[cols_to_clean]
    
    if len(ordinal_cols) > 0:
        for c in ordinal_cols:
            assert c in cols_to_clean

        non_ordinal_cols = []
        for c in cols_to_clean:
            if c not in ordinal_cols:
                non_ordinal_cols.append(c)

        df_train_sub_ordinal = df_train_sub[ordinal_cols]
        df_val_sub_ordinal = df_val_sub[ordinal_cols]
        df_test_sub_ordinal = df_test_sub[ordinal_cols]

        df_train_sub_non_ordinal = df_train_sub[non_ordinal_cols]
        df_val_sub_non_ordinal = df_val_sub[non_ordinal_cols]
        df_test_sub_non_ordinal = df_test_sub[non_ordinal_cols]

        enc = OrdinalEncoder(handle_unknown='ignore')
        df_train_sub_ordinal = pd.DataFrame(
            enc.fit_transform(df_train_sub_ordinal), 
            index=df_train_sub_ordinal.index,
            columns=df_train_sub_ordinal.columns
        )
        df_val_sub_ordinal = pd.DataFrame(
            enc.transform(df_val_sub_ordinal), 
            index=df_val_sub_ordinal.index,
            columns=df_val_sub_ordinal.columns
        )
        df_test_sub_ordinal = pd.DataFrame(
            enc.transform(df_test_sub_ordinal), 
            index=df_test_sub_ordinal.index,
            columns=df_test_sub_ordinal.columns
        )

        df_train_sub = pd.concat([df_train_sub_ordinal, df_train_sub_non_ordinal], axis=1)
        df_val_sub = pd.concat([df_val_sub_ordinal, df_val_sub_non_ordinal], axis=1)
        df_test_sub = pd.concat([df_test_sub_ordinal, df_test_sub_non_ordinal], axis=1)
        
        if save_path is not None:
            with open(f"{save_path}/ordinal-enc.obj", "wb") as filehandler:
                pickle.dump(enc, filehandler)

    if impute:
        imp = IterativeImputer(missing_values=np.nan, max_iter=10, random_state=0)
        df_train_sub = pd.DataFrame(
            imp.fit_transform(df_train_sub), 
            index=df_train_sub.index,
            columns=df_train_sub.columns
        )
        df_val_sub = pd.DataFrame(
            imp.transform(df_val_sub), 
            index=df_val_sub.index,
            columns=df_val_sub.columns
        )
        df_test_sub = pd.DataFrame(
            imp.transform(df_test_sub), 
            index=df_test_sub.index,
            columns=df_test_sub.columns
        )
        if save_path is not None:
            with open(f"{save_path}/imputer-enc.obj", "wb") as filehandler:
                pickle.dump(imp, filehandler)

    if normalize:
        sc = StandardScaler()
        df_train_sub = pd.DataFrame(
            sc.fit_transform(df_train_sub),
            index=df_train_sub.index,
            columns=df_train_sub.columns
        )
        df_val_sub = pd.DataFrame(
            sc.transform(df_val_sub),
            index=df_val_sub.index,
            columns=df_val_sub.columns
        )
        df_test_sub = pd.DataFrame(
            sc.transform(df_test_sub),
            index=df_test_sub.index,
            columns=df_test_sub.columns
        )
        if save_path is not None:
            with open(f"{save_path}/scaler-enc.obj", "wb") as filehandler:
                pickle.dump(sc, filehandler)

    df_train_concat = pd.concat([df_train_nosub, df_train_sub], axis=1)
    df_val_concat = pd.concat([df_val_nosub, df_val_sub], axis=1)
    df_test_concat = pd.concat([df_test_nosub, df_test_sub], axis=1)

    return df_train_concat, df_val_concat, df_test_concat
