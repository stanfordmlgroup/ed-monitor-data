import unittest
from processing.generate_lactate_dataset import run

import h5py
import math
import pandas as pd
import numpy as np


class TestGenerateLactateDataset(unittest.TestCase):

    def test_process(self):
        run(input_folder="resources", input_file="resources/lactate.csv",
            output_file="resources/output.csv", limit=None
        )

        # Validate output CSV file
        #
        df = pd.read_csv("resources/output.csv")
        print(df.iloc[0])
        self.assertEqual(1, df.shape[0])
        actual = list(df.iloc[0])
        expected = ['XX12915362YY', 175.66491252324647, 0.3610893954696408, 2718.116165433352, -11.831598709639705, 76.53584161309175, 0.4047971331016039, 9.542876203962326, 0.7449896434996773, 18.27068965517241, 0.1287854352421093, 9.384567796439375, 0.4129534419422679, 98.81340333612395, 0.1310665796023685, 0.22494536438067, -0.0090692074124444, 785.1979511415782, -4.351937910690245, 2372.220945933969, 257.0682774046929, 128.0, 11.99999999999998, 0.0, 0.0, 82.5, 19.0, 0.0, 0.0, 1.7829633985973117, -0.1494710407667326, 0.0197323782242168, -0.0037035677574331]
        self.compare_lists(expected, actual)

    def compare_lists(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for idx, v in enumerate(expected):
            if (isinstance(v, int) or isinstance(v, float)) and math.isnan(v):
                self.assertTrue(math.isnan(actual[idx]))
            else:
                self.assertEqual(v, actual[idx])

if __name__ == '__main__':
    unittest.main()
