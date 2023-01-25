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
        print(df.head(1))
        # self.assertEqual(1, df.shape[0])


if __name__ == '__main__':
    unittest.main()
