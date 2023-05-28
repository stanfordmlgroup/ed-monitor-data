import unittest

import pandas as pd

from processing.match import run


class TestMatch(unittest.TestCase):

    def test_process(self):
        run(cohort_file="resources/visits.csv",
            experiment_folders=["resources/2022_08_01_2022_09_28"],
            cohort_output_file="resources/cohort-output.csv",
            export_output_file="resources/export-output.csv",
            positive_column=None
        )

        expected_df = pd.read_csv("resources/expected-cohort-output.csv")
        df = pd.read_csv("resources/cohort-output.csv")
        self.assertTrue(expected_df.equals(df))

        expected_df = pd.read_csv("resources/expected-export-output.csv")
        df = pd.read_csv("resources/export-output.csv")
        self.assertTrue(expected_df.equals(df))


if __name__ == '__main__':
    unittest.main()
