#!/usr/bin/env python

"""
Run after the prepare_ed_numerics_from_matched_cohort.py script to create a file that summarizes which CSNs were 
processed into a pickle file. 

Files are written out to:
- /deep/group/physiologic-states/v1/processed/<hash>/<CSN>.pkl
where <hash> is the last two characters of the CSN

Example: python summarize_ed_numerics.py -i /deep/group/physiologic-states/v1/processed
"""    

import os
import csv
import argparse
import datetime
import pickle
from tqdm import tqdm

def run(args):
    print(f"START TIME: {datetime.datetime.now()}")
    input_folder = args.input_folder

    rows = []
    for root, dirs, files in tqdm(os.walk(input_folder)):
        for name in files:
            if name.endswith(".pkl"):
                try:
                    with open(os.path.join(root, name), 'rb') as handle:
                        b = pickle.load(handle)
                        hr_len = len(b["HR"])
                        rr_len = len(b["RR"])
                        spo2_len = len(b["SpO2"])
                        nbps_len = len(b["NBPs"])
                        nbpd_len = len(b["NBPd"])
                        btb_len = len(b["btbRRInt_ms"])
                        rows.append([name.replace(".pkl", ""), str(hr_len), str(rr_len), str(spo2_len), str(nbps_len), str(nbpd_len), str(btb_len)])
                except:
                    print(f"Could not load file {name}")

    with open(f"{input_folder}/summary.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["csn", "HR_len", "RR_len", "SpO2_len", "NBPs_len", "NBPd_len", "btb_len"])
        for line in rows:
            writer.writerow(line)

    print(f"END TIME: {datetime.datetime.now()}")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarizes what ED numerics files are available')
    parser.add_argument('-i', '--input-folder',
                        required=True,
                        help='Folder containing the individual hash folders')

    args = parser.parse_args()

    run(args)

    print("DONE")
