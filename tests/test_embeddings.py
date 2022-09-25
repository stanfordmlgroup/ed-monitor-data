import unittest
import pandas as pd
import numpy as np
import edm.utils.embeddings as embeddings

"""
python -m unittest tests.test_embeddings
"""

class TestEmbeddings(unittest.TestCase):

    def test_get_cohort_df(self):
        df = pd.DataFrame({
            'patient_id': [101, 102, 103],
            'outcome': [0, 1, 0],
            'HR': [70, 100, 80],
            'RR': [np.nan, 12, 30]
        })
        
        output_df = embeddings.get_cohort_df(df, additional_cols=["HR", "RR"])
        self.assertEqual(output_df.shape[0], 3)
        self.assertEqual(output_df.shape[1], 4)
        print(output_df)

    def test_get_embedding_df(self):
        df = pd.DataFrame({
            'patient_id': [101, 102, 103],
            'outcome': [0, 1, 0],
            'HR': [70, 100, 80],
            'RR': [np.nan, 12, 30]
        })
        samples_df = pd.DataFrame({'record_name' : [101, 102, 103]})
        waveforms = np.zeros((3, 12)) # 3 samples x 12 columns
        
        output_df = embeddings.get_embedding_df(df, samples_df, waveforms, additional_cols=["HR", "RR"])
        self.assertEqual(output_df.shape[0], 3)
        self.assertEqual(output_df.shape[1], 16)
        # print(output_df)

    def test_clean_additional_columns(self):
        df_train = pd.DataFrame({
            'patient_id': [101, 102, 103],
            'outcome': [0, 1, 1],
            'Sex': ['M', 'F', 'F'],
            'HR': [70, 100, 80],
            'RR': [np.nan, 12, 30]
        })
        
        df_val = pd.DataFrame({
            'patient_id': [104, 105],
            'outcome': [0, 1],
            'Sex': ['M', 'F'],
            'HR': [72, 130],
            'RR': [np.nan, 12]
        })
        
        df_test = pd.DataFrame({
            'patient_id': [106, 107],
            'outcome': [0, 1],
            'Sex': ['M', 'F'],
            'HR': [72, 130],
            'RR': [12, 12]
        })
        
        df_train, df_val, df_test = embeddings.clean_additional_columns(df_train, df_val, df_test, cols_to_clean=["Sex", "RR"], ordinal_cols=["Sex"])
        self.assertEqual(df_train.shape[0], 3)
        self.assertEqual(df_train.shape[1], 5)
        self.assertEqual(df_val.shape[0], 2)
        self.assertEqual(df_val.shape[1], 5)
        self.assertEqual(df_test.shape[0], 2)
        self.assertEqual(df_test.shape[1], 5)
        self.assertEqual(round(df_train.iloc[2]["RR"], 6), 1.224745)
        self.assertEqual(round(df_test.iloc[0]["RR"], 6), -1.224745)
        # print(df_train)
        # print(df_val)
        # print(df_test)

if __name__ == '__main__':
    unittest.main()
