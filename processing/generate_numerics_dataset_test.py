import unittest
from datetime import datetime

import h5py
import numpy as np

from preprocess.waveform import WaveformPreprocess


class TestGenerateDownstreamDataset(unittest.TestCase):

    def test_process(self):
        self.assertEqual(500, 500)


if __name__ == '__main__':
    unittest.main()
