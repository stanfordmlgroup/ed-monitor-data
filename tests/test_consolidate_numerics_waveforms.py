"""
The following tests are representative of the test suite for the consolidate_numerics_waveforms.py
Actual test cases and their corresponding data files are not released for confidentiality reasons.
"""

import datetime
import os
import unittest

import json
import pytz
import matplotlib.pyplot as plt

from processing.consolidate_numerics_waveforms import load_numerics_file, process_numerics_file, \
    process_study, make_waveform_lengths_consistent, NULL_WAVEFORM_VALUE, get_skip_waveform_seconds, \
    get_overlap_interval, handle_waveform_overlap_and_gap, get_start_offset_time, get_end_offset_time

import h5py
import pandas as pd
import numpy as np

SAMPLE_PATIENT_ID = 1234567890
SHOW_PLOTS = False


class TestConsolidate(unittest.TestCase):

    def test_load_numerics_file(self):
        study_to_study_folder = {
            "STUDY-000000": "resources/STUDY-000000",
            "STUDY-000001": "resources/STUDY-000001",
        }
        output_files = load_numerics_file(study_to_study_folder, "STUDY-000000")
        self.assertEqual(1, len(output_files))
        self.assertEqual(31132, output_files[0].shape[0])

    def test_get_overlap_interval(self):
        self.assertEqual(0.008, get_overlap_interval(set(["II", "Pleth"])))
        self.assertEqual(0.016, get_overlap_interval(set(["II", "Pleth", "Resp"])))
        self.assertEqual(0.008, get_overlap_interval(set(["Pleth"])))
        self.assertEqual(0.002, get_overlap_interval(set(["II"])))
        self.assertRaises(Exception, lambda x: get_overlap_interval(set()))

    def test_get_start_offset_time(self):
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.173Z"), get_start_offset_time(self.localize_iso_time("2000-10-21T10:44:23.173Z"), self.localize_iso_time("2000-10-21T10:44:23.173Z")))
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.273Z"), get_start_offset_time(self.localize_iso_time("2000-10-21T10:44:23.173Z"), self.localize_iso_time("2000-10-21T10:44:23.273Z")))
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.185Z"), get_start_offset_time(self.localize_iso_time("2000-10-21T10:44:23.173Z"), self.localize_iso_time("2000-10-21T10:44:23.073Z")))

    def test_get_end_offset_time(self):
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.173Z"), get_end_offset_time(self.localize_iso_time("2000-10-21T10:44:23.173Z"), self.localize_iso_time("2000-10-21T10:44:23.173Z")))
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.161Z"), get_end_offset_time(self.localize_iso_time("2000-10-21T10:44:23.173Z"), self.localize_iso_time("2000-10-21T10:44:23.273Z")))
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.073Z"), get_end_offset_time(self.localize_iso_time("2000-10-21T10:44:23.173Z"), self.localize_iso_time("2000-10-21T10:44:23.073Z")))

    def test_handle_waveform_overlap_and_gap(self):
        #
        # No overlap or gap
        #
        prev_time = None
        metadata = {
            "time_jumps": [],
            "start_offset_time": self.localize_iso_time("2000-10-21T10:44:23.173Z")
        }
        final_waveform = np.zeros(1000)
        sample_rate = 500
        available_waveforms = set(["II"])
        output_waveform = handle_waveform_overlap_and_gap(SAMPLE_PATIENT_ID, prev_time, metadata, final_waveform, sample_rate, available_waveforms)
        self.assertEqual(1000, len(output_waveform))
        self.assertEqual(0, len(metadata["time_jumps"]))

        #
        # Jump ahead with perfect alignment
        #
        prev_time = self.localize_iso_time("2000-10-21T10:44:23.160Z")
        metadata = {
            "time_jumps": [],
            "start_offset_time": self.localize_iso_time("2000-10-21T10:44:23.160Z")
        }
        final_waveform = np.zeros(1000)
        sample_rate = 500
        available_waveforms = set(["II", "Pleth", "Resp"])
        output_waveform = handle_waveform_overlap_and_gap(SAMPLE_PATIENT_ID, prev_time, metadata, final_waveform, sample_rate, available_waveforms)
        self.assertEqual(1000, len(output_waveform))
        self.assertEqual(0, len(metadata["time_jumps"]))

        #
        # Jump ahead with exactly one cycle of difference
        #
        prev_time = self.localize_iso_time("2000-10-21T10:44:23.160Z")
        metadata = {
            "time_jumps": [],
            "start_offset_time": self.localize_iso_time("2000-10-21T10:44:23.176Z")
        }
        final_waveform = np.ones(1000)
        sample_rate = 500
        available_waveforms = set(["II", "Pleth", "Resp"])
        output_waveform = handle_waveform_overlap_and_gap(SAMPLE_PATIENT_ID, prev_time, metadata, final_waveform, sample_rate, available_waveforms)
        self.assertEqual(1008, len(output_waveform))
        self.assertTrue(np.all(output_waveform[:1000] == 1))
        self.assertTrue(np.all(output_waveform[1000:1008] == NULL_WAVEFORM_VALUE))
        self.assertEqual(0, len(metadata["time_jumps"]))

        #
        # Jump ahead with less than one cycle of difference
        #
        prev_time = self.localize_iso_time("2000-10-21T10:44:23.160Z")
        metadata = {
            "time_jumps": [],
            "start_offset_time": self.localize_iso_time("2000-10-21T10:44:23.161Z")
        }
        final_waveform = np.ones(1000)
        sample_rate = 125
        available_waveforms = set(["II", "Pleth", "Resp"])
        output_waveform = handle_waveform_overlap_and_gap(SAMPLE_PATIENT_ID, prev_time, metadata, final_waveform, sample_rate, available_waveforms)
        self.assertEqual(1000, len(output_waveform))
        self.assertTrue(np.all(output_waveform[:1000] == 1))
        self.assertEqual(1, len(metadata["time_jumps"]))
        self.assertEqual(999, metadata["time_jumps"][0][0][0])
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.160Z"), metadata["time_jumps"][0][0][1])
        self.assertEqual(1000, metadata["time_jumps"][0][1][0])
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.161Z"), metadata["time_jumps"][0][1][1])

        #
        # Jump ahead with more than one cycle of difference
        #
        prev_time = self.localize_iso_time("2000-10-21T10:44:23.160Z")
        metadata = {
            "time_jumps": [],
            "start_offset_time": self.localize_iso_time("2000-10-21T10:44:23.179Z")
        }
        final_waveform = np.ones(1000)
        sample_rate = 500
        available_waveforms = set(["II", "Pleth", "Resp"])
        output_waveform = handle_waveform_overlap_and_gap(SAMPLE_PATIENT_ID, prev_time, metadata, final_waveform, sample_rate, available_waveforms)
        self.assertEqual(1008, len(output_waveform))
        self.assertTrue(np.all(output_waveform[:1000] == 1))
        self.assertTrue(np.all(output_waveform[1000:1008] == NULL_WAVEFORM_VALUE))
        self.assertEqual(1, len(metadata["time_jumps"]))
        self.assertEqual(1007, metadata["time_jumps"][0][0][0])
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.176Z"), metadata["time_jumps"][0][0][1])
        self.assertEqual(1008, metadata["time_jumps"][0][1][0])
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.179Z"), metadata["time_jumps"][0][1][1])

        #
        # Fall behind with exactly one cycle of difference
        #
        prev_time = self.localize_iso_time("2000-10-21T10:44:23.160Z")
        metadata = {
            "time_jumps": [],
            "start_offset_time": self.localize_iso_time("2000-10-21T10:44:23.144Z")
        }
        final_waveform = np.ones(1000)
        sample_rate = 500
        available_waveforms = set(["II", "Pleth", "Resp"])
        output_waveform = handle_waveform_overlap_and_gap(SAMPLE_PATIENT_ID, prev_time, metadata, final_waveform, sample_rate, available_waveforms)
        self.assertEqual(992, len(output_waveform))
        self.assertTrue(np.all(output_waveform[:992] == 1))
        self.assertEqual(0, len(metadata["time_jumps"]))

        #
        # Fall behind with more than one cycle of difference
        #
        prev_time = self.localize_iso_time("2000-10-21T10:44:23.160Z")
        metadata = {
            "time_jumps": [],
            "start_offset_time": self.localize_iso_time("2000-10-21T10:44:23.143Z")
        }
        final_waveform = np.ones(1000)
        sample_rate = 500
        available_waveforms = set(["II", "Pleth", "Resp"])
        output_waveform = handle_waveform_overlap_and_gap(SAMPLE_PATIENT_ID, prev_time, metadata, final_waveform, sample_rate, available_waveforms)
        self.assertEqual(984, len(output_waveform))
        self.assertTrue(np.all(output_waveform[:984] == 1))
        self.assertEqual(1, len(metadata["time_jumps"]))
        self.assertEqual(983, metadata["time_jumps"][0][0][0])
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.128Z"), metadata["time_jumps"][0][0][1])
        self.assertEqual(984, metadata["time_jumps"][0][1][0])
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.143Z"), metadata["time_jumps"][0][1][1])

        #
        # Fall behind with less than one cycle of difference
        #
        prev_time = self.localize_iso_time("2000-10-21T10:44:23.160Z")
        metadata = {
            "time_jumps": [],
            "start_offset_time": self.localize_iso_time("2000-10-21T10:44:23.153Z")
        }
        final_waveform = np.ones(1000)
        sample_rate = 500
        available_waveforms = set(["II", "Pleth", "Resp"])
        output_waveform = handle_waveform_overlap_and_gap(SAMPLE_PATIENT_ID, prev_time, metadata, final_waveform, sample_rate, available_waveforms)
        self.assertEqual(992, len(output_waveform))
        self.assertTrue(np.all(output_waveform[:992] == 1))
        self.assertEqual(1, len(metadata["time_jumps"]))
        self.assertEqual(991, metadata["time_jumps"][0][0][0])
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.144Z"), metadata["time_jumps"][0][0][1])
        self.assertEqual(992, metadata["time_jumps"][0][1][0])
        self.assertEqual(self.localize_iso_time("2000-10-21T10:44:23.153Z"), metadata["time_jumps"][0][1][1])

    def test_get_skip_waveform_seconds(self):
        waveform_type_to_waveform = {
            "II": np.concatenate((np.full(500 * 5 * 60, -100), np.ones(500 * 5 * 60), np.full(500 * 5 * 30, -100)))
        }
        trim_start_sec, trim_end_sec = get_skip_waveform_seconds(SAMPLE_PATIENT_ID, waveform_type_to_waveform["II"], "II", 500)
        self.assertEquals(304, trim_start_sec)
        self.assertEquals(595, trim_end_sec)

    def test_process_numerics_file(self):
        study_to_study_folder = {
            "STUDY-000000": "resources/STUDY-000000",
            "STUDY-000001": "resources/STUDY-000001",
        }
        studies = ["STUDY-000000", "STUDY-000001"]

        roomed_time = datetime.datetime.strptime("2000-10-21T13:45:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("2000-10-21T19:59:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        output_vals, patient_id = process_numerics_file(SAMPLE_PATIENT_ID, study_to_study_folder, studies, roomed_time, dispo_time)
        self.assertEqual(16, len(output_vals))
        self.assertEqual(1634849495.215, output_vals["HR-time"][0])
        self.assertEqual(71.0, output_vals["HR"][0])
        self.assertEqual(SAMPLE_PATIENT_ID, patient_id)

    def test_make_waveform_lengths_consistent(self):
        waveform_to_metadata = {
            "II": {
            },
            "Pleth": {
            }
        }

        # II   :      ---
        # Pleth: ---
        waveform_type_to_times = {
            "II": {
                "start": self.localize_iso_time("2000-10-21T13:40:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:45:00Z")
            },
            "Pleth": {
                "start": self.localize_iso_time("2000-10-21T13:30:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:35:00Z")
            }
        }
        waveform_type_to_waveform = {
            "II": np.ones(500 * 5 * 60),
            "Pleth": np.ones(125 * 5 * 60)
        }
        make_waveform_lengths_consistent(waveform_type_to_times, waveform_to_metadata, waveform_type_to_waveform)
        self.assertEqual(500 * 5 * 60, len(waveform_type_to_waveform["II"]))
        self.assertTrue(np.all(waveform_type_to_waveform["II"] == 1))
        self.assertEqual(125 * 5 * 60, len(waveform_type_to_waveform["Pleth"]))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"] == NULL_WAVEFORM_VALUE))


        # II   :    ---
        # Pleth:  ---
        waveform_type_to_times = {
            "II": {
                "start": self.localize_iso_time("2000-10-21T13:40:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:45:00Z")
            },
            "Pleth": {
                "start": self.localize_iso_time("2000-10-21T13:36:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:41:00Z")
            }
        }
        waveform_type_to_waveform = {
            "II": np.ones(500 * 5 * 60),
            "Pleth": np.ones(125 * 5 * 60)
        }
        make_waveform_lengths_consistent(waveform_type_to_times, waveform_to_metadata, waveform_type_to_waveform)
        self.assertEqual(500 * 5 * 60, len(waveform_type_to_waveform["II"]))
        self.assertTrue(np.all(waveform_type_to_waveform["II"] == 1))
        self.assertEqual(125 * 5 * 60, len(waveform_type_to_waveform["Pleth"]))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"][:125 * 1 * 60] == 1))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"][125 * 1 * 60:] == NULL_WAVEFORM_VALUE))


        # II   : ------
        # Pleth:   ---
        waveform_type_to_times = {
            "II": {
                "start": self.localize_iso_time("2000-10-21T13:40:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:45:00Z")
            },
            "Pleth": {
                "start": self.localize_iso_time("2000-10-21T13:41:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:44:00Z")
            }
        }
        waveform_type_to_waveform = {
            "II": np.ones(500 * 5 * 60),
            "Pleth": np.ones(125 * 3 * 60)
        }
        make_waveform_lengths_consistent(waveform_type_to_times, waveform_to_metadata, waveform_type_to_waveform)
        self.assertEqual(500 * 5 * 60, len(waveform_type_to_waveform["II"]))
        self.assertTrue(np.all(waveform_type_to_waveform["II"] == 1))
        self.assertEqual(125 * 5 * 60, len(waveform_type_to_waveform["Pleth"]))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"][:125 * 1 * 60] == NULL_WAVEFORM_VALUE))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"][125 * 1 * 60:125 * 4 * 60] == 1))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"][125 * 4 * 60:] == NULL_WAVEFORM_VALUE))


        # II   :     ---
        # Pleth:   -------
        waveform_type_to_times = {
            "II": {
                "start": self.localize_iso_time("2000-10-21T13:40:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:45:00Z")
            },
            "Pleth": {
                "start": self.localize_iso_time("2000-10-21T13:39:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:46:00Z")
            }
        }
        waveform_type_to_waveform = {
            "II": np.ones(500 * 5 * 60),
            "Pleth": np.ones(125 * 7 * 60)
        }
        make_waveform_lengths_consistent(waveform_type_to_times, waveform_to_metadata, waveform_type_to_waveform)
        self.assertEqual(500 * 5 * 60, len(waveform_type_to_waveform["II"]))
        self.assertTrue(np.all(waveform_type_to_waveform["II"] == 1))
        self.assertEqual(125 * 5 * 60, len(waveform_type_to_waveform["Pleth"]))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"] == 1))


        # II   :  ---
        # Pleth:    ---
        waveform_type_to_times = {
            "II": {
                "start": self.localize_iso_time("2000-10-21T13:40:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:45:00Z")
            },
            "Pleth": {
                "start": self.localize_iso_time("2000-10-21T13:44:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:49:00Z")
            }
        }
        waveform_type_to_waveform = {
            "II": np.ones(500 * 5 * 60),
            "Pleth": np.ones(125 * 5 * 60)
        }
        make_waveform_lengths_consistent(waveform_type_to_times, waveform_to_metadata, waveform_type_to_waveform)
        self.assertEqual(500 * 5 * 60, len(waveform_type_to_waveform["II"]))
        self.assertTrue(np.all(waveform_type_to_waveform["II"] == 1))
        self.assertEqual(125 * 5 * 60, len(waveform_type_to_waveform["Pleth"]))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"][:125 * 4 * 60] == NULL_WAVEFORM_VALUE))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"][125 * 4 * 60:] == 1))

        # II   :  ---
        # Pleth:      ---
        waveform_type_to_times = {
            "II": {
                "start": self.localize_iso_time("2000-10-21T13:40:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:45:00Z")
            },
            "Pleth": {
                "start": self.localize_iso_time("2000-10-21T13:46:00Z"),
                "end": self.localize_iso_time("2000-10-21T13:51:00Z")
            }
        }
        waveform_type_to_waveform = {
            "II": np.ones(500 * 5 * 60),
            "Pleth": np.ones(125 * 5 * 60)
        }
        make_waveform_lengths_consistent(waveform_type_to_times, waveform_to_metadata, waveform_type_to_waveform)
        self.assertEqual(500 * 5 * 60, len(waveform_type_to_waveform["II"]))
        self.assertTrue(np.all(waveform_type_to_waveform["II"] == 1))
        self.assertEqual(125 * 5 * 60, len(waveform_type_to_waveform["Pleth"]))
        self.assertTrue(np.all(waveform_type_to_waveform["Pleth"] == NULL_WAVEFORM_VALUE))

    def test_process_study_inconsistent(self):
        study_to_study_folder = {
            "STUDY-000002": "resources/STUDY-000002",
            "STUDY-000003": "resources/STUDY-000003",
        }
        studies = ["STUDY-000002", "STUDY-000003"]

        roomed_time = datetime.datetime.strptime("2002-06-29T16:08:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("2002-06-29T18:39:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000002": {
                "export_start_time": self.localize_time("06/29/02 02:36:09"),
                "export_end_time": self.localize_time("06/29/02 17:42:36"),
                "study_folder": "resources/STUDY-000002"
            },
            "STUDY-000003": {
                "export_start_time": self.localize_time("06/29/02 17:42:36"),
                "export_end_time": self.localize_time("06/29/02 22:41:50"),
                "study_folder": "resources/STUDY-000003"
            }
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            print("HR measurements:")
            print(f['numerics']['HR'].shape)
            print("II shape:")
            print(f['waveforms']['II'].shape)
            print("II seconds:")
            print(f['waveforms']['II'].shape[0] / 500)
            print("Pleth seconds:")
            print(f['waveforms']['Pleth'].shape[0] / 125)
            ii_processed = f['waveforms']['II'][:]
            ppg_processed = f['waveforms']['Pleth'][:]
            resp_processed = f['waveforms']['Resp'][:]

            self.assertEqual(3358, round(len(ii_processed) / 500))
            self.assertEqual(3358, round(len(ppg_processed) / 125))
            self.assertEqual(3358, round(len(resp_processed) / 62.5))

    def test_process_study(self):
        study_to_study_folder = {
            "STUDY-000000": "resources/STUDY-000000",
            "STUDY-000001": "resources/STUDY-000001",
        }
        studies = ["STUDY-000000", "STUDY-000001"]

        roomed_time = datetime.datetime.strptime("2000-10-21T13:45:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("2000-10-21T19:59:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000000": {
                "export_start_time": self.localize_time("10/21/00 13:51:19"),
                "export_end_time": self.localize_time("10/21/00 21:12:56"),
                "study_folder": "resources/STUDY-000000"
            },
            "STUDY-000001": {
                "export_start_time": self.localize_time("10/20/00 14:46:04"),
                "export_end_time": self.localize_time("10/21/00 13:51:19"),
                "study_folder": "resources/STUDY-000001"
            }
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])
        self.assertEqual("2000-10-21T13:45:00.007000-07:00", obj["waveform_start_time"].isoformat())
        self.assertEqual("2000-10-21T19:58:59.999000-07:00", obj["waveform_end_time"].isoformat())

        actual_obj = json.dumps(obj, indent=4, sort_keys=True, default=str)
        expected_obj = ""
        with open("expected_output/test_process_study.json", "r") as f:
            for row in f:
                expected_obj += row
        self.assertEqual(expected_obj, actual_obj)

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            with h5py.File(f"expected_output/test_process_study.h5", "r") as f_expected:
                self.assertEqual(sorted(list(f_expected.keys())), sorted(list(f.keys())))
                self.assertEqual(sorted(list(f_expected['numerics'].keys())), sorted(list(f['numerics'].keys())))
                self.assertEqual(sorted(list(f_expected['waveforms'].keys())), sorted(list(f['waveforms'].keys())))

                for k in f_expected['numerics'].keys():
                    self.assertEqual(f_expected['numerics'][k].shape, f['numerics'][k].shape)
                    self.assertTrue(np.all(f_expected['numerics'][k][:] == f['numerics'][k][:]))

                for k in f_expected['waveforms'].keys():
                    self.assertEqual(f_expected['waveforms'][k].shape, f['waveforms'][k].shape)
                    self.assertTrue(np.all(f_expected['waveforms'][k][:] == f['waveforms'][k][:]))

                for k in f_expected['waveforms_time_jumps'].keys():
                    self.assertEqual(f_expected['waveforms_time_jumps'][k].shape, f['waveforms_time_jumps'][k].shape)
                    self.assertTrue(np.all(f_expected['waveforms_time_jumps'][k][:] == f['waveforms_time_jumps'][k][:]))
                    for jump in f_expected['waveforms_time_jumps'][k][:]:
                        print(f"{k}: {jump[0][0]:.6f} : {jump[0][1]:.6f}")
                        print(f"{k}: {jump[1][0]:.6f} : {jump[1][1]:.6f}")

                # e.g. A single time jump contains a pair of tuples, representing the end of the first segment and
                #      start of the second segment. First element of the tuple is position in the waveform array
                #      and second element is the corresponding time in epoch timestamp
                # II: (192431, 1634849484.871000), (192432, 1634849484.879000)
                self.assertEqual(192431, f_expected['waveforms_time_jumps']["II"][:][0][0][0])
                self.assertEqual(1634849484.871000, f_expected['waveforms_time_jumps']["II"][:][0][0][1])
                self.assertEqual(192432, f_expected['waveforms_time_jumps']["II"][:][0][1][0])
                self.assertEqual(1634849484.879000, f_expected['waveforms_time_jumps']["II"][:][0][1][1])
                self.assertEqual(48107, f_expected['waveforms_time_jumps']["Pleth"][:][0][0][0])
                self.assertEqual(1634849484.871000, f_expected['waveforms_time_jumps']["Pleth"][:][0][0][1])
                self.assertEqual(48108, f_expected['waveforms_time_jumps']["Pleth"][:][0][1][0])
                self.assertEqual(1634849484.879000, f_expected['waveforms_time_jumps']["Pleth"][:][0][1][1])
                self.assertEqual(24053, f_expected['waveforms_time_jumps']["Resp"][:][0][0][0])
                self.assertEqual(1634849484.871000, f_expected['waveforms_time_jumps']["Resp"][:][0][0][1])
                self.assertEqual(24054, f_expected['waveforms_time_jumps']["Resp"][:][0][1][0])
                self.assertEqual(1634849484.879000, f_expected['waveforms_time_jumps']["Resp"][:][0][1][1])

                ii_processed = f['waveforms']['II'][:]
                ppg_processed = f['waveforms']['Pleth'][:]
                resp_processed = f['waveforms']['Resp'][:]
                if SHOW_PLOTS:
                    a4_dims = (12, 4)
                    fig, (ax1, ax2) = plt.subplots(2, figsize=a4_dims)
                    start_sec = 60
                    len_sec = 10
                    ax1.plot(ii_processed[start_sec * 500:start_sec * 500 + 500 * len_sec])
                    ax1.set_ylabel("II")
                    ax2.plot(ppg_processed[start_sec * 125:start_sec * 125 + 125 * len_sec])
                    ax2.set_ylabel("PPG")
                    plt.show()

            print("---")
            recommended_trim_start_sec = obj["trim_start_sec"]
            recommended_trim_end_sec = obj["trim_end_sec"]
            self.assertEqual(5, recommended_trim_start_sec)
            self.assertEqual(7026, recommended_trim_end_sec)

            if SHOW_PLOTS:
                a4_dims = (12, 4)
                fig, (ax1, ax2) = plt.subplots(2, figsize=a4_dims)
                ax1.plot(ii_processed)
                ax1.set_ylabel("II")
                ax2.plot(ii_processed[recommended_trim_start_sec * 500:recommended_trim_end_sec * 500])
                ax2.set_ylabel("II (with trim)")
                plt.show()

            self.assertEqual(22439.984, len(ii_processed) / 500)
            self.assertEqual(22439.984, len(ppg_processed) / 125)
            self.assertEqual(22439.984, len(resp_processed) / 62.5)


    def test_process_study_single_study_with_uneven_waveform_lengths(self):
        study_to_study_folder = {
            "STUDY-000004": "resources/STUDY-000004"
        }
        studies = ["STUDY-000004"]

        roomed_time = datetime.datetime.strptime("2000-07-04T20:46:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("2000-07-04T23:32:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000004": {
                "export_start_time": self.localize_time("07/04/00 20:43:15"),
                "export_end_time": self.localize_time("07/05/00 00:10:07"),
                "study_folder": "resources/STUDY-000004"
            }
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            ii_processed = f['waveforms']['II'][:]
            ppg_processed = f['waveforms']['Pleth'][:]
            resp_processed = f['waveforms']['Resp'][:]
            print("---")

            self.assertEqual(9880.88, len(ii_processed) / 500)
            self.assertEqual(9880.88, len(ppg_processed) / 125)
            self.assertEqual(9880.88, len(resp_processed) / 62.5)


    def test_process_study_single_study_with_waveform_off_by_large_amount(self):
        study_to_study_folder = {
            "STUDY-000005": "resources/STUDY-000005"
        }
        studies = ["STUDY-000005"]

        roomed_time = datetime.datetime.strptime("2000-09-01T17:39:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("2000-09-02T00:47:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000005": {
                "export_start_time": self.localize_time("09/01/00 01:17:14"),
                "export_end_time": self.localize_time("09/03/00 00:38:39"),
                "study_folder": "resources/STUDY-000005"
            }
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            ii_processed = f['waveforms']['II'][:]
            ppg_processed = f['waveforms']['Pleth'][:]
            resp_processed = f['waveforms']['Resp'][:]
            print("---")

            self.assertEqual(25679.968, len(ii_processed) / 500)
            self.assertEqual(25679.968, len(ppg_processed) / 125)
            self.assertEqual(25679.968, len(resp_processed) / 62.5)

    def test_process_study_single_study_with_diff_sample_rate(self):
        study_to_study_folder = {
            "STUDY-000006": "resources/STUDY-000006"
        }
        studies = ["STUDY-000006"]

        roomed_time = datetime.datetime.strptime("2002-04-09T20:50:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("2002-04-09T22:47:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000006": {
                "export_start_time": self.localize_time("04/08/02 23:22:43"),
                "export_end_time": self.localize_time("04/09/02 23:42:05"),
                "study_folder": "resources/STUDY-000006"
            }
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            ii_processed = f['waveforms']['II'][:]
            ppg_processed = f['waveforms']['Pleth'][:]
            resp_processed = f['waveforms']['Resp'][:]
            print("---")

            if SHOW_PLOTS:
                a4_dims = (12, 4)
                fig, (ax1, ax2) = plt.subplots(2, figsize=a4_dims)
                start_sec = 60
                len_sec = 10
                ax1.plot(ii_processed[start_sec * 500:start_sec * 500 + 500 * len_sec])
                ax1.set_ylabel("II")
                ax2.plot(ppg_processed[start_sec * 125:start_sec * 125 + 125 * len_sec])
                ax2.set_ylabel("PPG")
                plt.show()

            self.assertEqual(7019.984, len(ii_processed) / 500)
            self.assertEqual(7019.984, len(ppg_processed) / 125)
            self.assertEqual(7019.984, len(resp_processed) / 62.5)

    def test_process_study_single_study_with_no_waveforms(self):
        study_to_study_folder = {
            "STUDY-000007": "resources/STUDY-000007"
        }
        studies = ["STUDY-000007"]

        roomed_time = datetime.datetime.strptime("1998-10-31T12:59:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("1998-10-31T16:31:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000007": {
                "export_start_time": self.localize_time("10/30/98 20:05:20"),
                "export_end_time": self.localize_time("10/31/98 21:07:18"),
                "study_folder": "resources/STUDY-000007"
            }
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            self.assertTrue("II" not in f["waveforms"])
            self.assertTrue("Pleth" not in f["waveforms"])
            self.assertTrue("Resp" not in f["waveforms"])

    def test_process_study_single_study_with_no_II_waveforms(self):
        study_to_study_folder = {
            "STUDY-000008": "resources/STUDY-000008"
        }
        studies = ["STUDY-000008"]

        roomed_time = datetime.datetime.strptime("1998-10-10T13:11:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("1998-10-10T16:35:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000008": {
                "export_start_time": self.localize_time("10/08/98 13:22:09"),
                "export_end_time": self.localize_time("10/12/98 19:35:58"),
                "study_folder": "resources/STUDY-000008"
            }
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])
        self.assertEqual("1998-10-10T13:28:20.151000-07:00", obj["waveform_start_time"].isoformat())
        self.assertEqual("1998-10-10T16:32:40.631000-07:00", obj["waveform_end_time"].isoformat())

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            self.assertTrue("II" not in f["waveforms"])
            self.assertTrue("Pleth" in f["waveforms"])
            self.assertTrue("Resp" not in f["waveforms"])
            spo2 = f['numerics']['SpO2'][:]
            spo2_times = f['numerics']['SpO2-time'][:]
            ppg = f['waveforms']['Pleth'][:]
            self.assertEqual(5742, len(spo2))
            self.assertTrue(all(spo2_times[i] <= spo2_times[i+1] for i in range(len(spo2_times) - 1)))
            self.assertEqual(1382560, len(ppg))

            if SHOW_PLOTS:
                a4_dims = (12, 4)
                fig, (ax2) = plt.subplots(1, figsize=a4_dims)
                start_sec = 60
                len_sec = 10
                ax2.plot(ppg[start_sec * 125:start_sec * 125 + 125 * len_sec])
                ax2.set_ylabel("PPG")
                plt.show()

    def test_process_study_large_gap(self):
        # Large gap due to Pleth file that continued even when II didn't
        study_to_study_folder = {
            "STUDY-000009": "resources/STUDY-000009"
        }
        studies = ["STUDY-000009"]

        roomed_time = datetime.datetime.strptime("1998-08-27T20:33:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("1998-08-28T00:47:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000009": {
                "export_start_time": self.localize_time("08/26/98 20:20:28"),
                "export_end_time": self.localize_time("08/28/98 13:45:22"),
                "study_folder": "resources/STUDY-000009"
            }
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            ii_processed = f['waveforms']['II'][:]
            ppg_processed = f['waveforms']['Pleth'][:]
            resp_processed = f['waveforms']['Resp'][:]
            print("---")

            if SHOW_PLOTS:
                a4_dims = (12, 4)
                fig, (ax1, ax2) = plt.subplots(2, figsize=a4_dims)
                start_sec = 60
                len_sec = 10
                ax1.plot(ii_processed[start_sec * 500:start_sec * 500 + 500 * len_sec])
                ax1.set_ylabel("II")
                ax2.plot(ppg_processed[start_sec * 125:start_sec * 125 + 125 * len_sec])
                ax2.set_ylabel("PPG")
                plt.show()

            self.assertEqual(12420, round(len(ii_processed) / 500))
            self.assertEqual(12420, round(len(ppg_processed) / 125))
            self.assertEqual(12420, round(len(resp_processed) / 62.5))


    def test_process_study_double_study(self):
        study_to_study_folder = {
            "STUDY-000010": "resources/STUDY-000010",
            "STUDY-000011": "resources/STUDY-000011"
        }
        studies = ["STUDY-000010", "STUDY-000011"]

        roomed_time = datetime.datetime.strptime("1998-08-26T22:54:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("1998-08-27T02:43:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000010": {
                "export_start_time": self.localize_time("08/26/98 13:57:23"),
                "export_end_time": self.localize_time("08/26/98 22:59:42"),
                "study_folder": "resources/STUDY-000010"
            },
            'STUDY-000011': {
                "export_start_time": self.localize_time("08/26/98 22:59:42"),
                "export_end_time": self.localize_time("08/27/98 17:58:26"),
                "study_folder": "resources/STUDY-000011"
            },
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            ii_processed = f['waveforms']['II'][:]
            ppg_processed = f['waveforms']['Pleth'][:]
            resp_processed = f['waveforms']['Resp'][:]
            print("---")

            if SHOW_PLOTS:
                a4_dims = (12, 4)
                fig, (ax1, ax2) = plt.subplots(2, figsize=a4_dims)
                start_sec = 60
                len_sec = 10
                ax1.plot(ii_processed[start_sec * 500:start_sec * 500 + 500 * len_sec])
                ax1.set_ylabel("II")
                ax2.plot(ppg_processed[start_sec * 125:start_sec * 125 + 125 * len_sec])
                ax2.set_ylabel("PPG")
                plt.show()

            self.assertEqual(13739.968, len(ii_processed) / 500)
            self.assertEqual(13739.968, len(ppg_processed) / 125)
            self.assertEqual(13739.968, len(resp_processed) / 62.5)

    def test_process_study_double_empty_clock_file(self):
        study_to_study_folder = {
            "STUDY-000012": "resources/STUDY-000012",
            "STUDY-000013": "resources/STUDY-000013"
        }
        studies = ["STUDY-000012", "STUDY-000013"]

        roomed_time = datetime.datetime.strptime("2002-04-07T21:44:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("2002-04-08T00:24:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000012": {
                "export_start_time": self.localize_time("04/07/02 18:26:06"),
                "export_end_time": self.localize_time("04/07/02 21:51:47"),
                "study_folder": "resources/STUDY-000012"
            },
            'STUDY-000013': {
                "export_start_time": self.localize_time("04/07/02 21:51:47"),
                "export_end_time": self.localize_time("04/08/02 01:36:07"),
                "study_folder": "resources/STUDY-000013"
            },
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            ii_processed = f['waveforms']['II'][:]
            ppg_processed = f['waveforms']['Pleth'][:]
            resp_processed = f['waveforms']['Resp'][:]
            print("---")

            if SHOW_PLOTS:
                a4_dims = (12, 4)
                fig, (ax1, ax2) = plt.subplots(2, figsize=a4_dims)
                start_sec = 60
                len_sec = 10
                ax1.plot(ii_processed[start_sec * 500:start_sec * 500 + 500 * len_sec])
                ax1.set_ylabel("II")
                ax2.plot(ppg_processed[start_sec * 125:start_sec * 125 + 125 * len_sec])
                ax2.set_ylabel("PPG")
                plt.show()

            self.assertEqual(9127, round(len(ii_processed) / 500))
            self.assertEqual(9127, round(len(ppg_processed) / 125))
            self.assertEqual(9127, round(len(resp_processed) / 62.5))


    def test_process_study_with_overlapping_studies(self):
        """
        There are two studies which have overlapping times
        """
        study_to_study_folder = {
            "STUDY-000014": "resources/STUDY-000014",
            "STUDY-000015": "resources/STUDY-000015",
        }
        studies = ["STUDY-000014", "STUDY-000015"]

        roomed_time = datetime.datetime.strptime("2002-05-30T19:31:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        roomed_time = pytz.timezone('America/Vancouver').localize(roomed_time)
        dispo_time = datetime.datetime.strptime("2002-05-30T23:52:00Z", "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
        dispo_time = pytz.timezone('America/Vancouver').localize(dispo_time)
        curr_patient_index = 0
        total_patients = 1
        patient_to_actual_times = {
            SAMPLE_PATIENT_ID: {
                "roomed_time": roomed_time,
                "dispo_time": dispo_time
            }
        }
        df = pd.read_csv("resources/matched-cohort.csv")
        patient_to_row = {
            SAMPLE_PATIENT_ID: df.iloc[0]
        }

        study_to_info = {
            "STUDY-000014": {
                "export_start_time": self.localize_time("05/30/02 02:06:50"),
                "export_end_time": self.localize_time("05/30/02 19:34:08"),
                "study_folder": "resources/STUDY-000014"
            },
            "STUDY-000015": {
                "export_start_time": self.localize_time("05/30/02 19:34:08"),
                "export_end_time": self.localize_time("05/31/02 01:51:12"),
                "study_folder": "resources/STUDY-000015"
            }
        }
        output_dir = "output"
        input_args = [
            curr_patient_index, total_patients, SAMPLE_PATIENT_ID, studies, patient_to_actual_times, patient_to_row,
            study_to_info, study_to_study_folder, output_dir
        ]
        obj = process_study(input_args)
        self.assertEqual(SAMPLE_PATIENT_ID, obj["patient_id"])

        with h5py.File(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5", "r") as f:
            print(f.keys())
            print(f['numerics'].keys())
            print(f['waveforms'].keys())
            print("---")
            print("HR measurements:")
            print(f['numerics']['HR'].shape)
            print("II shape:")
            print(f['waveforms']['II'].shape)
            print("II seconds:")
            print(f['waveforms']['II'].shape[0] / 500)
            print("Pleth seconds:")
            print(f['waveforms']['Pleth'].shape[0] / 125)
            hr = f['numerics']['HR'][:]
            hr_times = f['numerics']['HR-time'][:]
            spo2 = f['numerics']['SpO2'][:]
            spo2_times = f['numerics']['SpO2-time'][:]
            ii_processed = f['waveforms']['II'][:]
            ppg_processed = f['waveforms']['Pleth'][:]
            resp_processed = f['waveforms']['Resp'][:]
            print("---")

            if SHOW_PLOTS:
                a4_dims = (12, 4)
                fig, (ax1, ax2) = plt.subplots(2, figsize=a4_dims)
                ax1.plot(ii_processed)
                ax1.set_ylabel("II")
                ax2.plot(ppg_processed)
                ax2.set_ylabel("PPG")
                plt.show()

            diff = (obj["waveform_end_time"] - obj["waveform_start_time"]).total_seconds()
            self.assertEqual(diff, len(ii_processed) / 500)
            self.assertEqual(diff, len(ppg_processed) / 125)
            self.assertEqual(diff, len(resp_processed) / 62.5)

            self.assertEqual(2623, len(hr))
            self.assertTrue(all(hr_times[i] <= hr_times[i+1] for i in range(len(hr_times) - 1)))
            self.assertEqual(9503, len(spo2))
            self.assertTrue(all(spo2_times[i] <= spo2_times[i+1] for i in range(len(spo2_times) - 1)))

    def localize_iso_time(self, t):
        try:
            t = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S%z')
        except:
            t = datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f%z')
        return t.astimezone(pytz.timezone('America/Vancouver'))

    def localize_time(self, t):
        t = datetime.datetime.strptime(t, '%m/%d/%y %H:%M:%S')
        return t.astimezone(pytz.timezone('America/Vancouver'))

    def tearDown(self):
        if os.path.exists(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5"):
            os.remove(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}/{SAMPLE_PATIENT_ID}.h5")
            os.rmdir(f"output/{str(SAMPLE_PATIENT_ID)[-2:]}")


if __name__ == '__main__':
    unittest.main()
