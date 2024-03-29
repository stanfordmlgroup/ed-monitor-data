import unittest
from processing.generate_downstream_dataset import run

import h5py
import math
import pandas as pd
import numpy as np


class TestGenerateDownstreamDataset(unittest.TestCase):

    def test_process(self):
        run(input_folder="resources", input_file="resources/consolidated.visits_ssl_2022_05_23.XX12915362YY.csv",
            output_data_file="resources/output.h5", output_summary_file="resources/output.csv",
            waveform_length_sec=60, pre_minutes_min=15, post_minutes_min=60,
            pre_granularity_sec=60, post_granularity_sec=60, align_col=None, limit=None
        )

        # Validate output CSV file
        #
        df = pd.read_csv("resources/output.csv")
        self.assertEqual(1, df.shape[0])

        expected = [15,60,15,60,15,60,15,60,2,4,2,4,14,58,30000,1,7500,1]
        actual = list(df.iloc[0])
        self.compare_lists(expected, actual[2:])

        expected = ["patient_id", "alignment_time", "HR_before_length", "HR_after_length", "RR_before_length", "RR_after_length", "SpO2_before_length", "SpO2_after_length", "btbRRInt_ms_before_length", "btbRRInt_ms_after_length", "NBPs_before_length", "NBPs_after_length", "NBPd_before_length", "NBPd_after_length", "Perf_before_length", "Perf_after_length", "II_length", "II_quality", "Pleth_length", "Pleth_quality"]
        actual = list(df.columns)
        self.compare_lists(expected, actual)

        # Validate output H5 file
        #
        with h5py.File("resources/output.h5", "r") as f:
            print()
            print(f['numerics_after'].keys())
            print("---")
            self.assertListEqual(
                ['alignment_times', 'numerics_after', 'numerics_before', 'waveforms'],
                list(f.keys())
            )

            expected_numeric_cols = ['HR', 'NBPd', 'NBPs', 'Perf', 'RR', 'SpO2', 'btbRRInt_ms']
            self.assertListEqual(
                expected_numeric_cols,
                list(f['numerics_before'].keys())
            )
            self.assertListEqual(
                expected_numeric_cols,
                list(f['numerics_after'].keys())
            )
            for numeric in expected_numeric_cols:
                sbp_times = list(f['numerics_before'][numeric]["times"][:][0])
                sbp_vals = list(f['numerics_before'][numeric]["vals"][:][0])
                self.assertEqual(15, len(sbp_times))
                self.assertEqual(15, len(sbp_vals))

                sbp_times = list(f['numerics_after'][numeric]["times"][:][0])
                sbp_vals = list(f['numerics_after'][numeric]["vals"][:][0])
                self.assertEqual(60, len(sbp_times))
                self.assertEqual(60, len(sbp_vals))

            ii = f['waveforms']['II']["waveforms"][:][0]
            self.assertEqual(30000, len(ii))
            self.assertEqual(0.0169, round(ii[0], 4))
            self.assertEqual(0.0182, round(ii[-1], 4))

            ppg = f['waveforms']['Pleth']["waveforms"][:][0]
            self.assertEqual(7500, len(ppg))
            self.assertEqual(0.5111, round(ppg[0], 4))
            self.assertEqual(69.1695, round(ppg[-1], 4))

            hr_vals = list(f['numerics_after']['HR']["vals"][:][0])
            self.assertEqual(79.14, round(hr_vals[0], 2))
            self.assertEqual(71.55, round(hr_vals[-1], 2))

            sbp_vals = np.array(list(f['numerics_after']['NBPs']["vals"][:][0]))
            self.compare_lists([134, 133, 123, 120], sbp_vals[[6, 21, 36, 51]])

    def compare_lists(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for idx, v in enumerate(expected):
            if (isinstance(v, int) or isinstance(v, float)) and math.isnan(v):
                self.assertTrue(math.isnan(actual[idx]))
            else:
                self.assertEqual(v, actual[idx])


if __name__ == '__main__':
    unittest.main()
